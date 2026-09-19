import numpy as np

from src.kinematics import get_end_effector_pose
from src.jacobian import translational_jacobian

from src.inverse_kinematics import (
    Q_MIN,
    Q_MAX,
    D6_MIN,
    D6_MAX,
)


def calculate_adaptive_damping(
    J,
    sigma_threshold=0.10,
    lambda_min=1e-4,
    lambda_max=0.20
):
    """
    Calculate adaptive damping from the minimum
    singular value of the Jacobian.

    Far from singularity:
        lambda -> lambda_min

    Near singularity:
        lambda -> lambda_max
    """

    J = np.asarray(
        J,
        dtype=float
    )

    singular_values = np.linalg.svd(
        J,
        compute_uv=False
    )

    sigma_min = float(
        np.min(singular_values)
    )

    if sigma_min >= sigma_threshold:
        damping = lambda_min

    else:
        ratio = (
            sigma_min /
            sigma_threshold
        )

        damping = (
            lambda_min
            +
            (
                lambda_max
                - lambda_min
            )
            * (
                1.0
                - ratio ** 2
            )
        )

    return float(damping), sigma_min


def adaptive_dls_pseudoinverse(
    J,
    sigma_threshold=0.10,
    lambda_min=1e-4,
    lambda_max=0.20
):
    """
    Compute a DLS pseudoinverse with an
    automatically selected damping coefficient.
    """

    damping, sigma_min = (
        calculate_adaptive_damping(
            J,
            sigma_threshold=sigma_threshold,
            lambda_min=lambda_min,
            lambda_max=lambda_max
        )
    )

    task_dimension = J.shape[0]

    regularization = (
        damping ** 2
    ) * np.eye(task_dimension)

    J_dls = J.T @ np.linalg.solve(
        J @ J.T + regularization,
        np.eye(task_dimension)
    )

    return (
        J_dls,
        damping,
        sigma_min
    )


def solve_position_ik_adaptive(
    target_position,
    q_initial,
    d6_initial,
    step_size=0.5,
    tolerance=1e-4,
    max_iterations=1000,
    sigma_threshold=0.10,
    lambda_min=1e-4,
    lambda_max=0.20
):
    """
    Position inverse kinematics using adaptive
    Damped Least Squares.
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

    position_history = []
    error_history = []
    damping_history = []
    sigma_min_history = []
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

        position_history.append(
            current_position.copy()
        )

        error_history.append(
            error_norm
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

        damping_history.append(
            damping
        )

        sigma_min_history.append(
            sigma_min
        )

        delta_joint = (
            step_size
            * J_dls
            @ error
        )

        q = (
            q
            + delta_joint[:5]
        )

        d6 = float(
            d6
            + delta_joint[5]
        )

        q = np.clip(
            q,
            Q_MIN,
            Q_MAX
        )

        d6 = float(
            np.clip(
                d6,
                D6_MIN,
                D6_MAX
            )
        )

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
        "damping_history": np.asarray(
            damping_history
        ),
        "sigma_min_history": np.asarray(
            sigma_min_history
        ),
        "q_history": np.asarray(
            q_history
        ),
        "d6_history": np.asarray(
            d6_history
        ),
    }
