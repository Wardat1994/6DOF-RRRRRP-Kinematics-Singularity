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

from src.obstacle_safety_filter import (
    enforce_safe_step,
    minimum_clearance_along_step,
)


# =====================================================
# SINGLE SAFETY-CONSTRAINED OBSTACLE-AWARE IK STEP
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

    use_safety_filter=True,
    safety_filter_buffer=0.005,
    safety_filter_samples=20,
    safety_filter_backtracking_factor=0.5,
    safety_filter_max_backtracking_steps=12,

    sigma_threshold=0.10,

    lambda_min=1e-4,
    lambda_max=0.20,

    max_revolute_step=np.radians(5.0),
    max_prismatic_step=0.02
):
    """
    Perform one safety-constrained obstacle-aware
    multi-objective IK iteration.

    Primary task
    ------------
    Cartesian position tracking.

    Secondary objectives
    --------------------
    1. Singularity avoidance.
    2. Joint-limit avoidance.
    3. Obstacle avoidance.

    Safety mechanisms
    -----------------
    1. Protected internal joint limits.
    2. Linearized obstacle-clearance safety filter.
    3. Nonlinear sampled path validation.
    4. Backtracking when the proposed motion is unsafe.

    Notes
    -----
    The obstacle safety verification is sampling-based.
    It improves trajectory safety but should not be
    interpreted as a formal continuous-time collision
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

    if safety_filter_buffer < 0.0:
        raise ValueError(
            "safety_filter_buffer must be non-negative."
        )

    if safety_filter_samples < 1:
        raise ValueError(
            "safety_filter_samples must be at least 1."
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
        sigma_threshold=sigma_threshold,
        lambda_min=lambda_min,
        lambda_max=lambda_max
    )

    # =================================================
    # PRIMARY TASK
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
    # COMBINED SECONDARY COMMAND
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
    # NOMINAL TOTAL UPDATE
    # =================================================

    nominal_total_delta = (
        primary_delta
        + secondary_delta
    )

    nominal_total_delta[:5] = np.clip(
        nominal_total_delta[:5],
        -max_revolute_step,
        max_revolute_step
    )

    nominal_total_delta[5] = np.clip(
        nominal_total_delta[5],
        -max_prismatic_step,
        max_prismatic_step
    )

    current_state = np.concatenate([
        q,
        [float(d6)]
    ])

    # =================================================
    # PROTECTED JOINT BOUNDS FIRST
    # =================================================

    safe_min, safe_max = (
        protected_joint_limits(
            safety_margin=(
                safety_margin
            )
        )
    )

    nominal_state = (
        current_state
        + nominal_total_delta
    )

    bounded_state = np.clip(
        nominal_state,
        safe_min,
        safe_max
    )

    bounded_state = np.clip(
        bounded_state,
        JOINT_MIN,
        JOINT_MAX
    )

    bounded_nominal_delta = (
        bounded_state
        - current_state
    )

    # =================================================
    # OBSTACLE SAFETY FILTER
    # =================================================

    if use_safety_filter:

        safety_result = (
            enforce_safe_step(
                q=q,
                d6=d6,

                delta=(
                    bounded_nominal_delta
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

                safety_distance=(
                    obstacle_safety_distance
                ),

                safety_buffer=(
                    safety_filter_buffer
                ),

                samples=(
                    safety_filter_samples
                ),

                backtracking_factor=(
                    safety_filter_backtracking_factor
                ),

                max_backtracking_steps=(
                    safety_filter_max_backtracking_steps
                )
            )
        )

        filtered_delta = (
            safety_result[
                "delta"
            ].copy()
        )

    else:

        filtered_delta = (
            bounded_nominal_delta.copy()
        )

        path_result = (
            minimum_clearance_along_step(
                q=q,
                d6=d6,

                delta=(
                    filtered_delta
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

                samples=(
                    safety_filter_samples
                )
            )
        )

        safety_result = {
            "accepted": True,
            "scale": 1.0,
            "backtracking_steps": 0,
            "minimum_path_clearance": (
                path_result[
                    "minimum_clearance"
                ]
            ),
            "minimum_path_fraction": (
                path_result[
                    "minimum_fraction"
                ]
            ),
            "linear_filter_corrected": False,
            "predicted_clearance": (
                clearance
            ),
        }

    # =================================================
    # RE-APPLY JOINT BOUNDS AFTER FILTER
    # =================================================
    #
    # The linear safety correction may change the
    # generalized command. Therefore, joint safety
    # bounds are applied again.
    # =================================================

    filtered_state = (
        current_state
        + filtered_delta
    )

    filtered_state = np.clip(
        filtered_state,
        safe_min,
        safe_max
    )

    filtered_state = np.clip(
        filtered_state,
        JOINT_MIN,
        JOINT_MAX
    )

    actual_delta = (
        filtered_state
        - current_state
    )

    # =================================================
    # FINAL NONLINEAR PATH SAFETY VERIFICATION
    # =================================================

    actual_path = (
        minimum_clearance_along_step(
            q=q,
            d6=d6,

            delta=(
                actual_delta
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

            samples=(
                safety_filter_samples
            )
        )
    )

    actual_safe = (
        actual_path[
            "minimum_clearance"
        ]
        >= obstacle_safety_distance
    )

    final_scale = 1.0

    final_backtracking_steps = 0

    # =================================================
    # SECOND BACKTRACKING LAYER
    # =================================================
    #
    # Joint clipping can slightly change the motion
    # originally approved by the obstacle filter.
    #
    # If that happens, backtrack the actual bounded
    # step until it is safe.
    # =================================================

    if (
        use_safety_filter
        and not actual_safe
    ):

        accepted = False

        scale = 1.0

        for backtracking_step in range(
            safety_filter_max_backtracking_steps
            + 1
        ):

            candidate_delta = (
                scale
                * actual_delta
            )

            candidate_path = (
                minimum_clearance_along_step(
                    q=q,
                    d6=d6,

                    delta=(
                        candidate_delta
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

                    samples=(
                        safety_filter_samples
                    )
                )
            )

            if (
                candidate_path[
                    "minimum_clearance"
                ]
                >= obstacle_safety_distance
            ):

                actual_delta = (
                    candidate_delta
                )

                actual_path = (
                    candidate_path
                )

                final_scale = float(
                    scale
                )

                final_backtracking_steps = (
                    backtracking_step
                )

                accepted = True

                break

            scale *= (
                safety_filter_backtracking_factor
            )

        if not accepted:

            actual_delta = np.zeros(
                6,
                dtype=float
            )

            actual_path = (
                minimum_clearance_along_step(
                    q=q,
                    d6=d6,

                    delta=(
                        actual_delta
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

                    samples=(
                        safety_filter_samples
                    )
                )
            )

            final_scale = 0.0

            final_backtracking_steps = (
                safety_filter_max_backtracking_steps
            )

    # =================================================
    # FINAL STATE
    # =================================================

    final_state = (
        current_state
        + actual_delta
    )

    final_state = np.clip(
        final_state,
        safe_min,
        safe_max
    )

    final_state = np.clip(
        final_state,
        JOINT_MIN,
        JOINT_MAX
    )

    q_new = (
        final_state[:5].copy()
    )

    d6_new = float(
        final_state[5]
    )

    final_step_safe = (
        actual_path[
            "minimum_clearance"
        ]
        >= obstacle_safety_distance
    )

    filter_applied = bool(
        use_safety_filter
    )

    filter_corrected = bool(
        safety_result[
            "linear_filter_corrected"
        ]
        or not np.allclose(
            actual_delta,
            bounded_nominal_delta
        )
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

        "nominal_total_delta": (
            nominal_total_delta
        ),

        "bounded_nominal_delta": (
            bounded_nominal_delta
        ),

        # Actual applied safe motion
        "total_delta": (
            actual_delta
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

        # Safety information
        "safety_filter_applied": (
            filter_applied
        ),

        "safety_filter_corrected": (
            filter_corrected
        ),

        "safety_filter_accepted": (
            bool(
                safety_result[
                    "accepted"
                ]
            )
            and final_step_safe
        ),

        "safety_filter_scale": (
            float(
                safety_result[
                    "scale"
                ]
            )
            * final_scale
        ),

        "safety_filter_backtracking_steps": (
            int(
                safety_result[
                    "backtracking_steps"
                ]
            )
            + final_backtracking_steps
        ),

        "minimum_path_clearance": float(
            actual_path[
                "minimum_clearance"
            ]
        ),

        "minimum_path_fraction": float(
            actual_path[
                "minimum_fraction"
            ]
        ),

        "predicted_clearance": float(
            safety_result[
                "predicted_clearance"
            ]
        ),
    }


# =====================================================
# COMPLETE SAFETY-CONSTRAINED OBSTACLE-AWARE IK
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

    use_safety_filter=True,
    safety_filter_buffer=0.005,
    safety_filter_samples=20,
    safety_filter_backtracking_factor=0.5,
    safety_filter_max_backtracking_steps=12,

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
        Protected Joint Bounds
        +
        Obstacle Clearance Safety Filter

    Strong convergence criterion
    ----------------------------
    The solver reports convergence only when:

        Cartesian error <= tolerance

    AND:

        Current obstacle clearance
        >= obstacle_safety_distance

    AND:

        Minimum sampled trajectory clearance
        >= obstacle_safety_distance
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
    # INITIAL SAFETY
    # =================================================

    initial_obstacle = (
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

    initial_clearance = float(
        initial_obstacle[
            "clearance"
        ]
    )

    if (
        use_safety_filter
        and initial_clearance
        < obstacle_safety_distance
    ):
        raise ValueError(
            "Initial robot configuration violates "
            "the requested obstacle safety distance."
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

    safety_filter_corrected_history = []
    safety_filter_accepted_history = []
    safety_filter_scale_history = []
    safety_filter_backtracking_history = []

    minimum_path_clearance_history = []

    q_history = []
    d6_history = []

    converged = False

    minimum_trajectory_clearance = (
        initial_clearance
    )

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

        minimum_trajectory_clearance = min(
            minimum_trajectory_clearance,
            current_clearance
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
        # STRONG CONVERGENCE CONDITION
        # ---------------------------------------------

        current_safe = (
            current_clearance
            >= obstacle_safety_distance
        )

        trajectory_safe = (
            minimum_trajectory_clearance
            >= obstacle_safety_distance
        )

        if (
            error_norm <= tolerance
            and current_safe
            and trajectory_safe
        ):

            converged = True
            break

        # ---------------------------------------------
        # CONTROL STEP
        # ---------------------------------------------

        result = (
            obstacle_aware_ik_step(
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

                use_safety_filter=(
                    use_safety_filter
                ),

                safety_filter_buffer=(
                    safety_filter_buffer
                ),

                safety_filter_samples=(
                    safety_filter_samples
                ),

                safety_filter_backtracking_factor=(
                    safety_filter_backtracking_factor
                ),

                safety_filter_max_backtracking_steps=(
                    safety_filter_max_backtracking_steps
                ),

                sigma_threshold=(
                    sigma_threshold
                ),

                lambda_min=lambda_min,
                lambda_max=lambda_max
            )
        )

        q = result["q"]
        d6 = result["d6"]

        minimum_trajectory_clearance = min(
            minimum_trajectory_clearance,
            result[
                "minimum_path_clearance"
            ]
        )

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

        safety_filter_corrected_history.append(
            bool(
                result[
                    "safety_filter_corrected"
                ]
            )
        )

        safety_filter_accepted_history.append(
            bool(
                result[
                    "safety_filter_accepted"
                ]
            )
        )

        safety_filter_scale_history.append(
            float(
                result[
                    "safety_filter_scale"
                ]
            )
        )

        safety_filter_backtracking_history.append(
            int(
                result[
                    "safety_filter_backtracking_steps"
                ]
            )
        )

        minimum_path_clearance_history.append(
            float(
                result[
                    "minimum_path_clearance"
                ]
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

    minimum_trajectory_clearance = min(
        minimum_trajectory_clearance,
        final_clearance
    )

    trajectory_safe = bool(
        minimum_trajectory_clearance
        >= obstacle_safety_distance
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

        "initial_obstacle_clearance": (
            initial_clearance
        ),

        "final_obstacle_clearance": (
            final_clearance
        ),

        "minimum_trajectory_clearance": (
            float(
                minimum_trajectory_clearance
            )
        ),

        "trajectory_safety_satisfied": (
            trajectory_safe
        ),

        "final_closest_obstacle_link": (
            final_obstacle[
                "link_number"
            ]
        ),

        "collision": bool(
            minimum_trajectory_clearance <= 0.0
        ),

        "converged": (
            bool(
                converged
                and trajectory_safe
            )
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

        "safety_filter_corrected_history": (
            np.asarray(
                safety_filter_corrected_history,
                dtype=bool
            )
        ),

        "safety_filter_accepted_history": (
            np.asarray(
                safety_filter_accepted_history,
                dtype=bool
            )
        ),

        "safety_filter_scale_history": (
            np.asarray(
                safety_filter_scale_history,
                dtype=float
            )
        ),

        "safety_filter_backtracking_history": (
            np.asarray(
                safety_filter_backtracking_history,
                dtype=int
            )
        ),

        "minimum_path_clearance_history": (
            np.asarray(
                minimum_path_clearance_history,
                dtype=float
            )
        ),

        "q_history": np.asarray(
            q_history
        ),

        "d6_history": np.asarray(
            d6_history
        ),
    }
