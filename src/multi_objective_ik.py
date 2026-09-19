import numpy as np

from src.kinematics import get_end_effector_pose
from src.jacobian import translational_jacobian
from src.adaptive_damping import adaptive_dls_pseudoinverse

from src.inverse_kinematics import (
    Q_MIN,
    Q_MAX,
    D6_MIN,
    D6_MAX,
)

from src.singularity_avoidance import (
    translational_sigma_min,
    sigma_min_gradient,
)

from src.joint_limit_avoidance import (
    normalized_joint_position,
    joint_limit_avoidance_direction,
    joint_limit_cost,
    joint_limit_margin,
)


# =====================================================
# GENERALIZED JOINT LIMITS
# =====================================================

JOINT_MIN = np.concatenate([
    Q_MIN,
    [D6_MIN]
])

JOINT_MAX = np.concatenate([
    Q_MAX,
    [D6_MAX]
])

JOINT_CENTER = (
    JOINT_MIN + JOINT_MAX
) / 2.0

JOINT_HALF_RANGE = (
    JOINT_MAX - JOINT_MIN
) / 2.0


# =====================================================
# NORMALIZE VECTOR
# =====================================================

def normalize_vector(vector):
    """
    Normalize a generalized joint-space vector.

    Returns a zero vector when its norm is
    numerically negligible.
    """

    vector = np.asarray(
        vector,
        dtype=float
    )

    norm = float(
        np.linalg.norm(vector)
    )

    if norm <= 1e-12:
        return np.zeros_like(
            vector
        )

    return (
        vector / norm
    )


# =====================================================
# SINGULARITY ACTIVATION
# =====================================================

def singularity_activation(
    sigma_min,
    sigma_threshold
):
    """
    Continuous activation for singularity avoidance.

    sigma_min >= threshold:
        activation = 0

    sigma_min -> 0:
        activation -> 1
    """

    if sigma_threshold <= 0.0:
        raise ValueError(
            "sigma_threshold must be positive."
        )

    activation = (
        sigma_threshold
        - sigma_min
    ) / sigma_threshold

    return float(
        np.clip(
            activation,
            0.0,
            1.0
        )
    )


# =====================================================
# JOINT-LIMIT ACTIVATION
# =====================================================

def joint_limit_activation_level(
    q,
    d6,
    activation_threshold=0.80
):
    """
    Return a normalized joint-limit risk level.

    0:
        all joints are inside the safe region.

    1:
        at least one joint has reached
        its physical limit.
    """

    if not 0.0 <= activation_threshold < 1.0:
        raise ValueError(
            "activation_threshold must be in [0, 1)."
        )

    normalized = (
        normalized_joint_position(
            q,
            d6
        )
    )

    maximum_absolute_position = float(
        np.max(
            np.abs(normalized)
        )
    )

    if (
        maximum_absolute_position
        <= activation_threshold
    ):
        return 0.0

    activation = (
        maximum_absolute_position
        - activation_threshold
    ) / (
        1.0
        - activation_threshold
    )

    return float(
        np.clip(
            activation,
            0.0,
            1.0
        )
    )


# =====================================================
# PROTECTED JOINT LIMITS
# =====================================================

def protected_joint_limits(
    safety_margin=0.05
):
    """
    Construct protected limits inside the
    physical hard limits.

    safety_margin = 0.05 reserves 5% of each
    joint's normalized range near both limits.
    """

    if not 0.0 <= safety_margin < 0.5:
        raise ValueError(
            "safety_margin must be in [0, 0.5)."
        )

    allowed_fraction = (
        1.0 - safety_margin
    )

    safe_min = (
        JOINT_CENTER
        - allowed_fraction
        * JOINT_HALF_RANGE
    )

    safe_max = (
        JOINT_CENTER
        + allowed_fraction
        * JOINT_HALF_RANGE
    )

    return (
        safe_min,
        safe_max
    )


# =====================================================
# SINGLE MULTI-OBJECTIVE IK STEP
# =====================================================

def multi_objective_ik_step(
    q,
    d6,
    target_position,
    step_size=0.5,
    singularity_gain=0.005,
    joint_limit_gain=0.005,
    joint_limit_activation=0.80,
    safety_margin=0.05,
    joint_limit_max_multiplier=5.0,
    sigma_threshold=0.10,
    lambda_min=1e-4,
    lambda_max=0.20,
    max_revolute_step=np.radians(5.0),
    max_prismatic_step=0.02
):
    """
    Perform one multi-objective IK iteration.

    Primary task
    ------------
    Cartesian end-effector position tracking.

    Secondary tasks
    ---------------
    1. Activated singularity avoidance.
    2. Adaptive joint-limit avoidance.

    Safety layer
    ------------
    Protected internal joint limits maintain
    a normalized safety margin from the
    physical hard limits.
    """

    q = np.asarray(
        q,
        dtype=float
    ).copy()

    target_position = np.asarray(
        target_position,
        dtype=float
    )

    if q.shape != (5,):
        raise ValueError(
            "q must contain five revolute joints."
        )

    if target_position.shape != (3,):
        raise ValueError(
            "target_position must have shape (3,)."
        )

    if singularity_gain < 0.0:
        raise ValueError(
            "singularity_gain must be non-negative."
        )

    if joint_limit_gain < 0.0:
        raise ValueError(
            "joint_limit_gain must be non-negative."
        )

    if joint_limit_max_multiplier < 1.0:
        raise ValueError(
            "joint_limit_max_multiplier must be >= 1."
        )

    if not 0.0 <= safety_margin < 0.5:
        raise ValueError(
            "safety_margin must be in [0, 0.5)."
        )

    if not (
        0.0
        <= joint_limit_activation
        < 1.0
    ):
        raise ValueError(
            "joint_limit_activation must be in [0, 1)."
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
        np.linalg.norm(error)
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
    # CARTESIAN POSITION TRACKING
    # =================================================

    primary_delta = (
        step_size
        * J_dls
        @ error
    )

    # =================================================
    # SECONDARY TASK 1
    # ACTIVATED SINGULARITY AVOIDANCE
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
    # ADAPTIVE JOINT-LIMIT AVOIDANCE
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

    # Gain increases nonlinearly as a joint
    # approaches its physical limit.
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
    # COMBINED SECONDARY OBJECTIVE
    # =================================================

    secondary_command = (
        singularity_command
        + joint_limit_command
    )

    # Damped task-priority / null-space projection
    null_space_projector = (
        np.eye(6)
        - J_dls @ J
    )

    secondary_delta = (
        null_space_projector
        @ secondary_command
    )

    # =================================================
    # TOTAL GENERALIZED JOINT UPDATE
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
    # PROTECTED SAFETY LIMITS
    # =================================================

    safe_min, safe_max = (
        protected_joint_limits(
            safety_margin=safety_margin
        )
    )

    protected_state = np.clip(
        proposed_state,
        safe_min,
        safe_max
    )

    # Final physical hard-limit protection
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

    return {
        "q": q_new,
        "d6": d6_new,

        "error": error,
        "error_norm": error_norm,

        "sigma_min": sigma_min,
        "damping": damping,

        "sigma_activation": sigma_activation,
        "joint_limit_activation": limit_activation,

        "effective_joint_limit_gain": (
            effective_limit_gain
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

        "primary_delta": primary_delta,

        "singularity_command": (
            singularity_command
        ),

        "joint_limit_command": (
            joint_limit_command
        ),

        "secondary_command": (
            secondary_command
        ),

        "secondary_delta": (
            secondary_delta
        ),

        "total_delta": total_delta,

        "singularity_gradient": (
            singularity_gradient
        ),
    }


# =====================================================
# COMPLETE MULTI-OBJECTIVE IK SOLVER
# =====================================================

def solve_multi_objective_ik(
    target_position,
    q_initial,
    d6_initial,
    step_size=0.5,
    singularity_gain=0.005,
    joint_limit_gain=0.005,
    joint_limit_activation=0.80,
    safety_margin=0.05,
    joint_limit_max_multiplier=5.0,
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
        Activated Singularity Avoidance
        +
        Adaptive Joint-Limit Avoidance
        +
        Protected Joint Safety Margin
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

    if target_position.shape != (3,):
        raise ValueError(
            "target_position must have shape (3,)."
        )

    if q.shape != (5,):
        raise ValueError(
            "q_initial must contain five revolute joints."
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

    sigma_activation_history = []

    joint_limit_cost_history = []
    joint_limit_margin_history = []
    joint_limit_activation_history = []

    singularity_activity_history = []
    joint_limit_activity_history = []
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
            np.linalg.norm(error)
        )

        current_sigma_min = (
            translational_sigma_min(
                q,
                d6
            )
        )

        current_sigma_activation = (
            singularity_activation(
                current_sigma_min,
                sigma_threshold
            )
        )

        current_limit_cost = (
            joint_limit_cost(
                q,
                d6,
                joint_limit_activation
            )
        )

        current_limit_margin = (
            joint_limit_margin(
                q,
                d6
            )
        )

        current_limit_activation = (
            joint_limit_activation_level(
                q,
                d6,
                activation_threshold=(
                    joint_limit_activation
                )
            )
        )

        # ---------------------------------------------
        # SAVE STATE
        # ---------------------------------------------

        position_history.append(
            current_position.copy()
        )

        error_history.append(
            error_norm
        )

        sigma_min_history.append(
            current_sigma_min
        )

        sigma_activation_history.append(
            current_sigma_activation
        )

        joint_limit_cost_history.append(
            current_limit_cost
        )

        joint_limit_margin_history.append(
            current_limit_margin
        )

        joint_limit_activation_history.append(
            current_limit_activation
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

        if error_norm <= tolerance:
            converged = True
            break

        result = multi_objective_ik_step(
            q=q,
            d6=d6,
            target_position=target_position,

            step_size=step_size,

            singularity_gain=(
                singularity_gain
            ),

            joint_limit_gain=(
                joint_limit_gain
            ),

            joint_limit_activation=(
                joint_limit_activation
            ),

            safety_margin=(
                safety_margin
            ),

            joint_limit_max_multiplier=(
                joint_limit_max_multiplier
            ),

            sigma_threshold=(
                sigma_threshold
            ),

            lambda_min=lambda_min,
            lambda_max=lambda_max
        )

        q = result["q"]
        d6 = result["d6"]

        damping_history.append(
            result["damping"]
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

    final_sigma_min = (
        translational_sigma_min(
            q,
            d6
        )
    )

    final_limit_margin = (
        joint_limit_margin(
            q,
            d6
        )
    )

    final_limit_cost = (
        joint_limit_cost(
            q,
            d6,
            joint_limit_activation
        )
    )

    return {
        "q": q,
        "d6": d6,

        "target_position": target_position,
        "final_position": final_position,

        "final_error_vector": (
            final_error_vector
        ),

        "final_error": final_error,

        "final_sigma_min": (
            final_sigma_min
        ),

        "final_joint_limit_margin": (
            final_limit_margin
        ),

        "final_joint_limit_cost": (
            final_limit_cost
        ),

        "safety_margin": (
            safety_margin
        ),

        "converged": converged,
        "iterations": iteration + 1,

        "position_history": np.asarray(
            position_history
        ),

        "error_history": np.asarray(
            error_history
        ),

        "sigma_min_history": np.asarray(
            sigma_min_history
        ),

        "sigma_activation_history": np.asarray(
            sigma_activation_history
        ),

        "damping_history": np.asarray(
            damping_history
        ),

        "joint_limit_cost_history": np.asarray(
            joint_limit_cost_history
        ),

        "joint_limit_margin_history": np.asarray(
            joint_limit_margin_history
        ),

        "joint_limit_activation_history": np.asarray(
            joint_limit_activation_history
        ),

        "singularity_activity_history": np.asarray(
            singularity_activity_history
        ),

        "joint_limit_activity_history": np.asarray(
            joint_limit_activity_history
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
