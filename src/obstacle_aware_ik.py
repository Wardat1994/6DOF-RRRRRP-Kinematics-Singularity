import numpy as np

from src.kinematics import get_end_effector_pose
from src.jacobian import translational_jacobian

from src.adaptive_damping import (
    adaptive_dls_pseudoinverse,
)

from src.multi_objective_ik import (
    normalize_vector,
    singularity_activation,
    joint_limit_activation_level,
    protected_joint_limits,
    JOINT_MIN,
    JOINT_MAX,
)

from src.singularity_avoidance import (
    translational_sigma_min,
    sigma_min_gradient,
)

from src.joint_limit_avoidance import (
    joint_limit_avoidance_direction,
    joint_limit_cost,
    joint_limit_margin,
)

from src.obstacle_avoidance import (
    minimum_obstacle_clearance,
    obstacle_activation,
    obstacle_clearance_gradient,
)


# =====================================================
# SINGLE OBSTACLE-AWARE IK STEP
# =====================================================

def obstacle_aware_ik_step(
    q,
    d6,
    target_position,
    obstacle_center,
    obstacle_radius,

    link_radius=0.05,

    step_size=0.5,

    singularity_gain=0.005,
    joint_limit_gain=0.005,
    obstacle_gain=0.01,

    joint_limit_activation=0.80,
    joint_limit_max_multiplier=5.0,

    safety_margin=0.05,

    obstacle_safety_distance=0.10,
    obstacle_influence_distance=0.40,

    sigma_threshold=0.10,

    lambda_min=1e-4,
    lambda_max=0.20,

    max_revolute_step=np.radians(5.0),
    max_prismatic_step=0.02
):
    """
    Perform one obstacle-aware multi-objective
    inverse-kinematics iteration.

    Primary task
    ------------
    Cartesian end-effector position tracking.

    Secondary objectives
    --------------------
    1. Singularity avoidance.
    2. Joint-limit avoidance.
    3. Obstacle avoidance.

    Safety layer
    ------------
    Protected joint safety bounds.

    Notes
    -----
    Obstacle avoidance is implemented as a
    secondary damped task-priority objective.

    Therefore, this first research implementation
    does not provide a formal collision-free
    guarantee.
    """

    q = np.asarray(
        q,
        dtype=float
    ).copy()

    target_position = np.asarray(
        target_position,
        dtype=float
    )

    obstacle_center = np.asarray(
        obstacle_center,
        dtype=float
    )

    obstacle_radius = float(
        obstacle_radius
    )

    link_radius = float(
        link_radius
    )

    # =================================================
    # VALIDATION
    # =================================================

    if q.shape != (5,):
        raise ValueError(
            "q must contain five revolute joints."
        )

    if target_position.shape != (3,):
        raise ValueError(
            "target_position must have shape (3,)."
        )

    if obstacle_center.shape != (3,):
        raise ValueError(
            "obstacle_center must have shape (3,)."
        )

    if obstacle_radius < 0.0:
        raise ValueError(
            "obstacle_radius must be non-negative."
        )

    if link_radius < 0.0:
        raise ValueError(
            "link_radius must be non-negative."
        )

    if singularity_gain < 0.0:
        raise ValueError(
            "singularity_gain must be non-negative."
        )

    if joint_limit_gain < 0.0:
        raise ValueError(
            "joint_limit_gain must be non-negative."
        )

    if obstacle_gain < 0.0:
        raise ValueError(
            "obstacle_gain must be non-negative."
        )

    if (
        obstacle_influence_distance
        <= obstacle_safety_distance
    ):
        raise ValueError(
            "obstacle_influence_distance must be "
            "greater than obstacle_safety_distance."
        )

    # =================================================
    # CURRENT CARTESIAN ERROR
    # =================================================

    _, current_position, _ = (
        get_end_effector_pose(
            q,
            d6
        )
    )

    error = (
        target_position
        - current_position
    )

    error_norm = float(
        np.linalg.norm(
            error
        )
    )

    # =================================================
    # TRANSLATIONAL JACOBIAN
    # =================================================

    J = translational_jacobian(
        q,
        d6
    )

    (
        J_dls,
        damping,
        sigma_min
    ) = adaptive_dls_pseudoinverse(
        J,
        sigma_threshold=(
            sigma_threshold
        ),
        lambda_min=lambda_min,
        lambda_max=lambda_max
    )

    # =================================================
    # PRIMARY TASK
    # CARTESIAN POSITION TRACKING
    # =================================================

    primary_delta = (
        step_size
        * J_dls
        @ error
    )

    # =================================================
    # SECONDARY TASK 1
    # SINGULARITY AVOIDANCE
    # =================================================

    sigma_activation = (
        singularity_activation(
            sigma_min,
            sigma_threshold
        )
    )

    singularity_gradient = (
        sigma_min_gradient(
            q,
            d6
        )
    )

    singularity_direction = (
        normalize_vector(
            singularity_gradient
        )
    )

    singularity_command = (
        singularity_gain
        * sigma_activation
        * singularity_direction
    )

    # =================================================
    # SECONDARY TASK 2
    # JOINT-LIMIT AVOIDANCE
    # =================================================

    limit_activation = (
        joint_limit_activation_level(
            q,
            d6,
            activation_threshold=(
                joint_limit_activation
            )
        )
    )

    limit_direction_raw = (
        joint_limit_avoidance_direction(
            q,
            d6,
            activation_threshold=(
                joint_limit_activation
            )
        )
    )

    limit_direction = (
        normalize_vector(
            limit_direction_raw
        )
    )

    effective_limit_gain = (
        joint_limit_gain
        * limit_activation
        * (
            1.0
            + (
                joint_limit_max_multiplier
                - 1.0
            )
            * limit_activation
        )
    )

    joint_limit_command = (
        effective_limit_gain
        * limit_direction
    )

    # =================================================
    # SECONDARY TASK 3
    # OBSTACLE AVOIDANCE
    # =================================================

    obstacle_result = (
        minimum_obstacle_clearance(
            q=q,
            d6=d6,
            obstacle_center=(
                obstacle_center
            ),
            obstacle_radius=(
                obstacle_radius
            ),
            link_radius=(
                link_radius
            )
        )
    )

    clearance = float(
        obstacle_result[
            "clearance"
        ]
    )

    obstacle_activation_value = (
        obstacle_activation(
            clearance=clearance,
            safety_distance=(
                obstacle_safety_distance
            ),
            influence_distance=(
                obstacle_influence_distance
            )
        )
    )

    # Only calculate the expensive gradient when
    # the obstacle is inside the influence region.
    if obstacle_activation_value > 0.0:

        obstacle_gradient = (
            obstacle_clearance_gradient(
                q=q,
                d6=d6,
                obstacle_center=(
                    obstacle_center
                ),
                obstacle_radius=(
                    obstacle_radius
                ),
                link_radius=(
                    link_radius
                )
            )
        )

        obstacle_direction = (
            normalize_vector(
                obstacle_gradient
            )
        )

    else:

        obstacle_gradient = np.zeros(
            6,
            dtype=float
        )

        obstacle_direction = np.zeros(
            6,
            dtype=float
        )

    obstacle_command = (
        obstacle_gain
        * obstacle_activation_value
        * obstacle_direction
    )

    # =================================================
    # COMBINED SECONDARY OBJECTIVES
    # =================================================

    secondary_command = (
        singularity_command
        + joint_limit_command
        + obstacle_command
    )

    # =================================================
    # DAMPED TASK-PRIORITY PROJECTION
    # =================================================

    projector = (
        np.eye(6)
        - J_dls @ J
    )

    secondary_delta = (
        projector
        @ secondary_command
    )

    # =================================================
    # TOTAL UPDATE
    # =================================================

    total_delta = (
        primary_delta
        + secondary_delta
    )

    # Limit revolute increments
    total_delta[:5] = np.clip(
        total_delta[:5],
        -max_revolute_step,
        max_revolute_step
    )

    # Limit prismatic increment
    total_delta[5] = np.clip(
        total_delta[5],
        -max_prismatic_step,
        max_prismatic_step
    )

    generalized_state = np.concatenate([
        q,
        [float(d6)]
    ])

    proposed_state = (
        generalized_state
        + total_delta
    )

    # =================================================
    # PROTECTED JOINT SAFETY BOUNDS
    # =================================================

    safe_min, safe_max = (
        protected_joint_limits(
            safety_margin=(
                safety_margin
            )
        )
    )

    protected_state = np.clip(
        proposed_state,
        safe_min,
        safe_max
    )

    protected_state = np.clip(
        protected_state,
        JOINT_MIN,
        JOINT_MAX
    )

    q_new = (
        protected_state[:5].copy()
    )

    d6_new = float(
        protected_state[5]
    )

    # =================================================
    # RETURN
    # =================================================

    return {
        "q": q_new,
        "d6": d6_new,

        "error": error,
        "error_norm": error_norm,

        "sigma_min": sigma_min,
        "damping": damping,

        "sigma_activation": (
            sigma_activation
        ),

        "joint_limit_activation": (
            limit_activation
        ),

        "obstacle_activation": (
            obstacle_activation_value
        ),

        "obstacle_clearance": (
            clearance
        ),

        "closest_obstacle_link": (
            obstacle_result[
                "link_number"
            ]
        ),

        "obstacle_collision": bool(
            clearance <= 0.0
        ),

        "primary_delta": (
            primary_delta
        ),

        "singularity_command": (
            singularity_command
        ),

        "joint_limit_command": (
            joint_limit_command
        ),

        "obstacle_command": (
            obstacle_command
        ),

        "secondary_command": (
            secondary_command
        ),

        "secondary_delta": (
            secondary_delta
        ),

        "total_delta": (
            total_delta
        ),

        "singularity_gradient": (
            singularity_gradient
        ),

        "obstacle_gradient": (
            obstacle_gradient
        ),

        "joint_limit_cost": (
            joint_limit_cost(
                q,
                d6,
                joint_limit_activation
            )
        ),

        "joint_limit_margin": (
            joint_limit_margin(
                q,
                d6
            )
        ),
    }


# =====================================================
# COMPLETE OBSTACLE-AWARE IK SOLVER
# =====================================================

def solve_obstacle_aware_ik(
    target_position,
    q_initial,
    d6_initial,

    obstacle_center,
    obstacle_radius,

    link_radius=0.05,

    step_size=0.5,

    singularity_gain=0.005,
    joint_limit_gain=0.005,
    obstacle_gain=0.01,

    joint_limit_activation=0.80,
    joint_limit_max_multiplier=5.0,

    safety_margin=0.05,

    obstacle_safety_distance=0.10,
    obstacle_influence_distance=0.40,

    tolerance=1e-4,
    max_iterations=1000,

    sigma_threshold=0.10,

    lambda_min=1e-4,
    lambda_max=0.20
):
    """
    Solve Cartesian position IK using:

        Adaptive DLS
        +
        Singularity Avoidance
        +
        Joint-Limit Avoidance
        +
        Obstacle Avoidance
        +
        Protected Joint Safety Bounds

    Convergence requires:

        Cartesian error <= tolerance

    and:

        obstacle clearance >= safety distance
    """

    target_position = np.asarray(
        target_position,
        dtype=float
    )

    q = np.asarray(
        q_initial,
        dtype=float
    ).copy()

    d6 = float(
        d6_initial
    )

    obstacle_center = np.asarray(
        obstacle_center,
        dtype=float
    )

    # =================================================
    # VALIDATION
    # =================================================

    if target_position.shape != (3,):
        raise ValueError(
            "target_position must have shape (3,)."
        )

    if q.shape != (5,):
        raise ValueError(
            "q_initial must contain five revolute joints."
        )

    if obstacle_center.shape != (3,):
        raise ValueError(
            "obstacle_center must have shape (3,)."
        )

    if tolerance <= 0.0:
        raise ValueError(
            "tolerance must be positive."
        )

    if max_iterations < 1:
        raise ValueError(
            "max_iterations must be at least 1."
        )

    # =================================================
    # HISTORIES
    # =================================================

    position_history = []
    error_history = []

    sigma_min_history = []
    damping_history = []

    joint_limit_margin_history = []
    joint_limit_cost_history = []

    obstacle_clearance_history = []
    obstacle_activation_history = []
    closest_obstacle_link_history = []

    singularity_activity_history = []
    joint_limit_activity_history = []
    obstacle_activity_history = []

    secondary_activity_history = []

    q_history = []
    d6_history = []

    converged = False

    # =================================================
    # ITERATIVE SOLVER
    # =================================================

    for iteration in range(
        max_iterations
    ):

        _, current_position, _ = (
            get_end_effector_pose(
                q,
                d6
            )
        )

        error = (
            target_position
            - current_position
        )

        error_norm = float(
            np.linalg.norm(
                error
            )
        )

        current_sigma = (
            translational_sigma_min(
                q,
                d6
            )
        )

        current_margin = (
            joint_limit_margin(
                q,
                d6
            )
        )

        current_cost = (
            joint_limit_cost(
                q,
                d6,
                joint_limit_activation
            )
        )

        current_obstacle = (
            minimum_obstacle_clearance(
                q=q,
                d6=d6,
                obstacle_center=(
                    obstacle_center
                ),
                obstacle_radius=(
                    obstacle_radius
                ),
                link_radius=(
                    link_radius
                )
            )
        )

        current_clearance = float(
            current_obstacle[
                "clearance"
            ]
        )

        current_obstacle_activation = (
            obstacle_activation(
                clearance=(
                    current_clearance
                ),
                safety_distance=(
                    obstacle_safety_distance
                ),
                influence_distance=(
                    obstacle_influence_distance
                )
            )
        )

        # ---------------------------------------------
        # SAVE CURRENT STATE
        # ---------------------------------------------

        position_history.append(
            current_position.copy()
        )

        error_history.append(
            error_norm
        )

        sigma_min_history.append(
            current_sigma
        )

        joint_limit_margin_history.append(
            current_margin
        )

        joint_limit_cost_history.append(
            current_cost
        )

        obstacle_clearance_history.append(
            current_clearance
        )

        obstacle_activation_history.append(
            current_obstacle_activation
        )

        closest_obstacle_link_history.append(
            current_obstacle[
                "link_number"
            ]
        )

        q_history.append(
            q.copy()
        )

        d6_history.append(
            d6
        )

        # ---------------------------------------------
        # CONVERGENCE
        # ---------------------------------------------

        obstacle_safe = (
            current_clearance
            >= obstacle_safety_distance
        )

        if (
            error_norm <= tolerance
            and obstacle_safe
        ):

            converged = True
            break

        # ---------------------------------------------
        # CONTROL STEP
        # ---------------------------------------------

        result = obstacle_aware_ik_step(
            q=q,
            d6=d6,

            target_position=(
                target_position
            ),

            obstacle_center=(
                obstacle_center
            ),

            obstacle_radius=(
                obstacle_radius
            ),

            link_radius=(
                link_radius
            ),

            step_size=(
                step_size
            ),

            singularity_gain=(
                singularity_gain
            ),

            joint_limit_gain=(
                joint_limit_gain
            ),

            obstacle_gain=(
                obstacle_gain
            ),

            joint_limit_activation=(
                joint_limit_activation
            ),

            joint_limit_max_multiplier=(
                joint_limit_max_multiplier
            ),

            safety_margin=(
                safety_margin
            ),

            obstacle_safety_distance=(
                obstacle_safety_distance
            ),

            obstacle_influence_distance=(
                obstacle_influence_distance
            ),

            sigma_threshold=(
                sigma_threshold
            ),

            lambda_min=(
                lambda_min
            ),

            lambda_max=(
                lambda_max
            )
        )

        q = result["q"]

        d6 = result["d6"]

        damping_history.append(
            result[
                "damping"
            ]
        )

        singularity_activity_history.append(
            float(
                np.linalg.norm(
                    result[
                        "singularity_command"
                    ]
                )
            )
        )

        joint_limit_activity_history.append(
            float(
                np.linalg.norm(
                    result[
                        "joint_limit_command"
                    ]
                )
            )
        )

        obstacle_activity_history.append(
            float(
                np.linalg.norm(
                    result[
                        "obstacle_command"
                    ]
                )
            )
        )

        secondary_activity_history.append(
            float(
                np.linalg.norm(
                    result[
                        "secondary_delta"
                    ]
                )
            )
        )

    # =================================================
    # FINAL STATE
    # =================================================

    _, final_position, _ = (
        get_end_effector_pose(
            q,
            d6
        )
    )

    final_error_vector = (
        target_position
        - final_position
    )

    final_error = float(
        np.linalg.norm(
            final_error_vector
        )
    )

    final_sigma = (
        translational_sigma_min(
            q,
            d6
        )
    )

    final_margin = (
        joint_limit_margin(
            q,
            d6
        )
    )

    final_cost = (
        joint_limit_cost(
            q,
            d6,
            joint_limit_activation
        )
    )

    final_obstacle = (
        minimum_obstacle_clearance(
            q=q,
            d6=d6,
            obstacle_center=(
                obstacle_center
            ),
            obstacle_radius=(
                obstacle_radius
            ),
            link_radius=(
                link_radius
            )
        )
    )

    final_clearance = float(
        final_obstacle[
            "clearance"
        ]
    )

    # =================================================
    # RETURN
    # =================================================

    return {
        "q": q,
        "d6": d6,

        "target_position": (
            target_position
        ),

        "final_position": (
            final_position
        ),

        "final_error_vector": (
            final_error_vector
        ),

        "final_error": (
            final_error
        ),

        "final_sigma_min": (
            final_sigma
        ),

        "final_joint_limit_margin": (
            final_margin
        ),

        "final_joint_limit_cost": (
            final_cost
        ),

        "final_obstacle_clearance": (
            final_clearance
        ),

        "final_closest_obstacle_link": (
            final_obstacle[
                "link_number"
            ]
        ),

        "collision": bool(
            final_clearance <= 0.0
        ),

        "converged": (
            converged
        ),

        "iterations": (
            iteration + 1
        ),

        "position_history": np.asarray(
            position_history
        ),

        "error_history": np.asarray(
            error_history
        ),

        "sigma_min_history": np.asarray(
            sigma_min_history
        ),

        "damping_history": np.asarray(
            damping_history
        ),

        "joint_limit_margin_history": np.asarray(
            joint_limit_margin_history
        ),

        "joint_limit_cost_history": np.asarray(
            joint_limit_cost_history
        ),

        "obstacle_clearance_history": np.asarray(
            obstacle_clearance_history
        ),

        "obstacle_activation_history": np.asarray(
            obstacle_activation_history
        ),

        "closest_obstacle_link_history": np.asarray(
            closest_obstacle_link_history
        ),

        "singularity_activity_history": np.asarray(
            singularity_activity_history
        ),

        "joint_limit_activity_history": np.asarray(
            joint_limit_activity_history
        ),

        "obstacle_activity_history": np.asarray(
            obstacle_activity_history
        ),

        "secondary_activity_history": np.asarray(
            secondary_activity_history
        ),

        "q_history": np.asarray(
            q_history
        ),

        "d6_history": np.asarray(
            d6_history
        ),
    }
