import numpy as np

from src.kinematics import get_end_effector_pose
from src.jacobian import translational_jacobian


# =====================================================
# DEFAULT JOINT LIMITS
# =====================================================

Q_MIN = np.radians([
    -180.0,
    -90.0,
    -90.0,
    -90.0,
    -90.0
])

Q_MAX = np.radians([
    180.0,
    90.0,
    90.0,
    90.0,
    90.0
])

D6_MIN = 0.03
D6_MAX = 0.48


# =====================================================
# DAMPED LEAST SQUARES PSEUDOINVERSE
# =====================================================

def damped_least_squares_pseudoinverse(
    J,
    damping=0.05
):
    """
    Compute the Damped Least Squares pseudoinverse.

    For a translational Jacobian J of shape (3, 6):

        J# = J.T @ inv(J @ J.T + lambda^2 I)

    Parameters
    ----------
    J : ndarray
        Jacobian matrix.

    damping : float
        Damping coefficient lambda.

    Returns
    -------
    ndarray
        DLS pseudoinverse.
    """

    J = np.asarray(
        J,
        dtype=float
    )

    if damping < 0.0:
        raise ValueError(
            "damping must be non-negative."
        )

    task_dimension = J.shape[0]

    regularization = (
        damping ** 2
    ) * np.eye(task_dimension)

    return J.T @ np.linalg.solve(
        J @ J.T + regularization,
        np.eye(task_dimension)
    )


# =====================================================
# SINGLE IK STEP
# =====================================================

def inverse_kinematics_step(
    q,
    d6,
    target_position,
    damping=0.05,
    step_size=0.5
):
    """
    Perform one DLS inverse-kinematics iteration.

    Returns
    -------
    q_new : ndarray
        Updated five revolute joint values.

    d6_new : float
        Updated prismatic displacement.

    error : ndarray
        Cartesian position error.

    error_norm : float
        Euclidean error magnitude.
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
            "q must contain five revolute joint values."
        )

    if target_position.shape != (3,):
        raise ValueError(
            "target_position must have shape (3,)."
        )

    if step_size <= 0.0:
        raise ValueError(
            "step_size must be positive."
        )

    # Current end-effector position
    _, current_position, _ = (
        get_end_effector_pose(
            q,
            d6
        )
    )

    # Cartesian position error
    error = (
        target_position
        - current_position
    )

    error_norm = float(
        np.linalg.norm(error)
    )

    # Translational Jacobian
    J = translational_jacobian(
        q,
        d6
    )

    # DLS pseudoinverse
    J_dls = (
        damped_least_squares_pseudoinverse(
            J,
            damping=damping
        )
    )

    # Joint-space correction
    delta_joint = (
        step_size
        * J_dls
        @ error
    )

    # First five values are revolute joints
    q_new = (
        q
        + delta_joint[:5]
    )

    # Sixth value belongs to prismatic joint
    d6_new = float(
        d6
        + delta_joint[5]
    )

    # Apply joint limits
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

    return (
        q_new,
        d6_new,
        error,
        error_norm
    )


# =====================================================
# COMPLETE POSITION IK SOLVER
# =====================================================

def solve_position_ik(
    target_position,
    q_initial,
    d6_initial,
    damping=0.05,
    step_size=0.5,
    tolerance=1e-4,
    max_iterations=1000
):
    """
    Solve Cartesian position inverse kinematics
    using iterative Damped Least Squares.

    Parameters
    ----------
    target_position : array-like, shape (3,)
        Desired Cartesian end-effector position.

    q_initial : array-like, shape (5,)
        Initial revolute joint configuration.

    d6_initial : float
        Initial prismatic joint displacement.

    damping : float
        DLS damping coefficient.

    step_size : float
        Scaling factor applied to each IK correction.

    tolerance : float
        Required Cartesian position accuracy.

    max_iterations : int
        Maximum number of IK iterations.

    Returns
    -------
    result : dict
        Contains final configuration, position,
        error, convergence state, and iteration history.
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

    position_history = []
    error_history = []
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

        # Store iteration data
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

        # Convergence check
        if error_norm <= tolerance:
            converged = True
            break

        q, d6, _, _ = (
            inverse_kinematics_step(
                q,
                d6,
                target_position,
                damping=damping,
                step_size=step_size
            )
        )

    # Final pose
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
        "final_position": final_position,
        "target_position": target_position,
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
        "q_history": np.asarray(
            q_history
        ),
        "d6_history": np.asarray(
            d6_history
        ),
    }
