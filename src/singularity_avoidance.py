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
# TRANSLATIONAL SINGULARITY METRIC
# =====================================================

def translational_sigma_min(q, d6):
    """
    Return the minimum singular value of the
    translational Jacobian Jv.
    """

    J = translational_jacobian(
        q,
        d6
    )

    singular_values = np.linalg.svd(
        J,
        compute_uv=False
    )

    return float(
        np.min(singular_values)
    )


# =====================================================
# NUMERICAL GRADIENT OF SIGMA_MIN
# =====================================================

def sigma_min_gradient(
    q,
    d6,
    angle_epsilon=1e-4,
    prismatic_epsilon=1e-4
):
    """
    Numerically estimate the gradient of sigma_min
    with respect to the five revolute joints and
    the prismatic displacement d6.

    Central finite differences are used.
    """

    q = np.asarray(
        q,
        dtype=float
    ).copy()

    gradient = np.zeros(6)

    # Revolute joints
    for i in range(5):

        q_plus = q.copy()
        q_minus = q.copy()

        q_plus[i] += angle_epsilon
        q_minus[i] -= angle_epsilon

        q_plus = np.clip(
            q_plus,
            Q_MIN,
            Q_MAX
        )

        q_minus = np.clip(
            q_minus,
            Q_MIN,
            Q_MAX
        )

        sigma_plus = translational_sigma_min(
            q_plus,
            d6
        )

        sigma_minus = translational_sigma_min(
            q_minus,
            d6
        )

        denominator = (
            q_plus[i]
            - q_minus[i]
        )

        if abs(denominator) > 1e-12:
            gradient[i] = (
                sigma_plus
                - sigma_minus
            ) / denominator

    # Prismatic joint
    d6_plus = float(
        np.clip(
            d6 + prismatic_epsilon,
            D6_MIN,
            D6_MAX
        )
    )

    d6_minus = float(
        np.clip(
            d6 - prismatic_epsilon,
            D6_MIN,
            D6_MAX
        )
    )

    sigma_plus = translational_sigma_min(
        q,
        d6_plus
    )

    sigma_minus = translational_sigma_min(
        q,
        d6_minus
    )

    denominator = (
        d6_plus
        - d6_minus
    )

    if abs(denominator) > 1e-12:
        gradient[5] = (
            sigma_plus
            - sigma_minus
        ) / denominator

    return gradient


# =====================================================
# SINGLE SINGULARITY-AVOIDANCE IK STEP
# =====================================================

def singularity_avoidance_step(
    q,
    d6,
    target_position,
    step_size=0.5,
    avoidance_gain=0.02,
    sigma_threshold=0.10,
    lambda_min=1e-4,
    lambda_max=0.20,
    max_revolute_step=np.radians(5.0),
    max_prismatic_step=0.02
):
    """
    Perform one inverse-kinematics step with:

    1. Adaptive DLS primary Cartesian task.
    2. Null-space singularity avoidance.

    The secondary task attempts to increase
    the minimum singular value sigma_min.
    """

    q = np.asarray(
        q,
        dtype=float
    ).copy()

    target_position = np.asarray(
        target_position,
        dtype=float
    )

    _, current_position, _ = get_end_effector_pose(
        q,
        d6
    )

    error = (
        target_position
        - current_position
    )

    error_norm = float(
        np.linalg.norm(error)
    )

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
    # Cartesian position tracking
    # =================================================

    primary_delta = (
        step_size
        * J_dls
        @ error
    )

    # =================================================
    # SECONDARY TASK
    # Increase sigma_min
    # =================================================

    gradient = sigma_min_gradient(
        q,
        d6
    )

    gradient_norm = float(
        np.linalg.norm(gradient)
    )

    if gradient_norm > 1e-12:
        gradient_direction = (
            gradient
            / gradient_norm
        )
    else:
        gradient_direction = np.zeros(6)

    # Damped null-space projection
    null_space_projector = (
        np.eye(6)
        - J_dls @ J
    )

    avoidance_delta = (
        avoidance_gain
        * null_space_projector
        @ gradient_direction
    )

    # =================================================
    # TOTAL JOINT UPDATE
    # =================================================

    total_delta = (
        primary_delta
        + avoidance_delta
    )

    # Limit incremental revolute motion
    total_delta[:5] = np.clip(
        total_delta[:5],
        -max_revolute_step,
        max_revolute_step
    )

    # Limit incremental prismatic motion
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

    # Hard joint limits
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
        "gradient": gradient,
    }


# =====================================================
# COMPLETE IK SOLVER WITH SINGULARITY AVOIDANCE
# =====================================================

def solve_position_ik_with_singularity_avoidance(
    target_position,
    q_initial,
    d6_initial,
    step_size=0.5,
    avoidance_gain=0.02,
    tolerance=1e-4,
    max_iterations=1000,
    sigma_threshold=0.10,
    lambda_min=1e-4,
    lambda_max=0.20
):
    """
    Solve Cartesian position IK using:

    Adaptive Damped Least Squares
    +
    Null-space singularity avoidance.
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

    if q.shape != (5,):
        raise ValueError(
            "q_initial must contain five revolute joints."
        )

    if target_position.shape != (3,):
        raise ValueError(
            "target_position must have shape (3,)."
        )

    position_history = []
    error_history = []
    sigma_min_history = []
    damping_history = []
    avoidance_history = []
    q_history = []
    d6_history = []

    converged = False

    for iteration in range(
        max_iterations
    ):

        _, current_position, _ = get_end_effector_pose(
            q,
            d6
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

        position_history.append(
            current_position.copy()
        )

        error_history.append(
            error_norm
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

        if error_norm <= tolerance:
            converged = True
            break

        step_result = singularity_avoidance_step(
            q=q,
            d6=d6,
            target_position=target_position,
            step_size=step_size,
            avoidance_gain=avoidance_gain,
            sigma_threshold=sigma_threshold,
            lambda_min=lambda_min,
            lambda_max=lambda_max
        )

        q = step_result["q"]
        d6 = step_result["d6"]

        damping_history.append(
            step_result["damping"]
        )

        avoidance_history.append(
            np.linalg.norm(
                step_result["avoidance_delta"]
            )
        )

    _, final_position, _ = get_end_effector_pose(
        q,
        d6
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
