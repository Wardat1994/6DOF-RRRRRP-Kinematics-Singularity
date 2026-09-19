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


# =====================================================
# JOINT-SPACE PARAMETERS
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
# NORMALIZED JOINT POSITION
# =====================================================

def normalized_joint_position(q, d6):
    """
    Normalize all six joint variables so that:

        center -> 0
        minimum limit -> -1
        maximum limit -> +1
    """

    q = np.asarray(
        q,
        dtype=float
    )

    if q.shape != (5,):
        raise ValueError(
            "q must contain five revolute joints."
        )

    state = np.concatenate([
        q,
        [float(d6)]
    ])

    normalized = (
        state - JOINT_CENTER
    ) / JOINT_HALF_RANGE

    return normalized


# =====================================================
# JOINT-LIMIT MARGIN
# =====================================================

def joint_limit_margin(q, d6):
    """
    Return the minimum normalized distance
    from any joint to its nearest limit.

    1.0 -> joint-space center
    0.0 -> at a hard joint limit
    """

    normalized = normalized_joint_position(
        q,
        d6
    )

    margins = (
        1.0
        - np.abs(normalized)
    )

    return float(
        np.min(margins)
    )


# =====================================================
# JOINT-LIMIT PENALTY
# =====================================================

def joint_limit_cost(
    q,
    d6,
    activation_threshold=0.80
):
    """
    Penalty becomes active only when a joint moves
    sufficiently close to one of its limits.

    activation_threshold = 0.80 means the avoidance
    task starts when a normalized joint coordinate
    exceeds 80% of its available range.
    """

    if not 0.0 <= activation_threshold < 1.0:
        raise ValueError(
            "activation_threshold must be in [0, 1)."
        )

    normalized = normalized_joint_position(
        q,
        d6
    )

    excess = np.maximum(
        np.abs(normalized)
        - activation_threshold,
        0.0
    )

    scaled_excess = (
        excess
        / (1.0 - activation_threshold)
    )

    return float(
        0.5
        * np.sum(
            scaled_excess ** 2
        )
    )


# =====================================================
# JOINT-LIMIT AVOIDANCE DIRECTION
# =====================================================

def joint_limit_avoidance_direction(
    q,
    d6,
    activation_threshold=0.80
):
    """
    Generate a preferred joint-space direction
    that moves active joints away from their limits
    and toward the center of their allowed range.

    The direction is based on the negative gradient
    of a normalized joint-limit penalty.
    """

    if not 0.0 <= activation_threshold < 1.0:
        raise ValueError(
            "activation_threshold must be in [0, 1)."
        )

    normalized = normalized_joint_position(
        q,
        d6
    )

    direction = np.zeros(6)

    denominator = (
        1.0
        - activation_threshold
    ) ** 2

    for i in range(6):

        abs_value = abs(
            normalized[i]
        )

        if abs_value > activation_threshold:

            excess = (
                abs_value
                - activation_threshold
            )

            # Gradient of the penalty in
            # normalized coordinates
            gradient_normalized = (
                np.sign(normalized[i])
                * excess
                / denominator
            )

            # Negative gradient:
            # direction away from joint limit
            direction[i] = (
                -JOINT_HALF_RANGE[i]
                * gradient_normalized
            )

    return direction


# =====================================================
# SINGLE JOINT-LIMIT-AVOIDANCE IK STEP
# =====================================================

def joint_limit_avoidance_step(
    q,
    d6,
    target_position,
    step_size=0.5,
    avoidance_gain=0.005,
    activation_threshold=0.80,
    sigma_threshold=0.10,
    lambda_min=1e-4,
    lambda_max=0.20,
    max_revolute_step=np.radians(5.0),
    max_prismatic_step=0.02
):
    """
    Perform one IK iteration using:

    1. Adaptive DLS for Cartesian position tracking.
    2. Null-space joint-limit avoidance.
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
    # END-EFFECTOR POSITION TRACKING
    # =================================================

    primary_delta = (
        step_size
        * J_dls
        @ error
    )

    # =================================================
    # SECONDARY TASK
    # JOINT-LIMIT AVOIDANCE
    # =================================================

    avoidance_direction = (
        joint_limit_avoidance_direction(
            q,
            d6,
            activation_threshold=activation_threshold
        )
    )

    direction_norm = float(
        np.linalg.norm(
            avoidance_direction
        )
    )

    if direction_norm > 1e-12:

        avoidance_direction = (
            avoidance_direction
            / direction_norm
        )

    else:

        avoidance_direction = (
            np.zeros(6)
        )

    # Damped null-space projector
    null_space_projector = (
        np.eye(6)
        - J_dls @ J
    )

    avoidance_delta = (
        avoidance_gain
        * null_space_projector
        @ avoidance_direction
    )

    # =================================================
    # TOTAL JOINT UPDATE
    # =================================================

    total_delta = (
        primary_delta
        + avoidance_delta
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

    q_new = (
        q
        + total_delta[:5]
    )

    d6_new = float(
        d6
        + total_delta[5]
    )

    # =================================================
    # HARD SAFETY LIMITS
    # =================================================

    q_new = np.clip(
        q_new,
        Q_MIN,
        Q_MAX
    )

    d6_new = float(
        np.clip(
            d6_new,
            D6_MIN,
            D6_MAX
        )
    )

    return {
        "q": q_new,
        "d6": d6_new,
        "error": error,
        "error_norm": error_norm,
        "sigma_min": sigma_min,
        "damping": damping,
        "primary_delta": primary_delta,
        "avoidance_delta": avoidance_delta,
        "total_delta": total_delta,
        "avoidance_direction": avoidance_direction,
        "joint_limit_cost": joint_limit_cost(
            q,
            d6,
            activation_threshold
        ),
        "joint_limit_margin": joint_limit_margin(
            q,
            d6
        ),
    }


# =====================================================
# COMPLETE IK SOLVER WITH JOINT-LIMIT AVOIDANCE
# =====================================================

def solve_position_ik_with_joint_limit_avoidance(
    target_position,
    q_initial,
    d6_initial,
    step_size=0.5,
    avoidance_gain=0.005,
    activation_threshold=0.80,
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
        Null-space joint-limit avoidance.
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

    if max_iterations < 1:
        raise ValueError(
            "max_iterations must be at least 1."
        )

    if tolerance <= 0.0:
        raise ValueError(
            "tolerance must be positive."
        )

    position_history = []
    error_history = []

    joint_limit_cost_history = []
    joint_limit_margin_history = []

    damping_history = []
    sigma_min_history = []
    avoidance_history = []

    q_history = []
    d6_history = []

    converged = False

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

        current_cost = (
            joint_limit_cost(
                q,
                d6,
                activation_threshold
            )
        )

        current_margin = (
            joint_limit_margin(
                q,
                d6
            )
        )

        J = translational_jacobian(
            q,
            d6
        )

        singular_values = np.linalg.svd(
            J,
            compute_uv=False
        )

        current_sigma_min = float(
            np.min(singular_values)
        )

        # Store history
        position_history.append(
            current_position.copy()
        )

        error_history.append(
            error_norm
        )

        joint_limit_cost_history.append(
            current_cost
        )

        joint_limit_margin_history.append(
            current_margin
        )

        sigma_min_history.append(
            current_sigma_min
        )

        q_history.append(
            q.copy()
        )

        d6_history.append(
            d6
        )

        # Convergence
        if error_norm <= tolerance:
            converged = True
            break

        step_result = (
            joint_limit_avoidance_step(
                q=q,
                d6=d6,
                target_position=target_position,
                step_size=step_size,
                avoidance_gain=avoidance_gain,
                activation_threshold=activation_threshold,
                sigma_threshold=sigma_threshold,
                lambda_min=lambda_min,
                lambda_max=lambda_max
            )
        )

        q = step_result["q"]
        d6 = step_result["d6"]

        damping_history.append(
            step_result["damping"]
        )

        avoidance_history.append(
            float(
                np.linalg.norm(
                    step_result[
                        "avoidance_delta"
                    ]
                )
            )
        )

    # =================================================
    # FINAL RESULTS
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

    return {
        "q": q,
        "d6": d6,
        "target_position": target_position,
        "final_position": final_position,
        "final_error_vector": final_error_vector,
        "final_error": final_error,
        "converged": converged,
        "iterations": iteration + 1,

        "position_history": np.asarray(
            position_history
        ),

        "error_history": np.asarray(
            error_history
        ),

        "joint_limit_cost_history": np.asarray(
            joint_limit_cost_history
        ),

        "joint_limit_margin_history": np.asarray(
            joint_limit_margin_history
        ),

        "sigma_min_history": np.asarray(
            sigma_min_history
        ),

        "damping_history": np.asarray(
            damping_history
        ),

        "avoidance_history": np.asarray(
            avoidance_history
        ),

        "q_history": np.asarray(
            q_history
        ),

        "d6_history": np.asarray(
            d6_history
        ),
    }
