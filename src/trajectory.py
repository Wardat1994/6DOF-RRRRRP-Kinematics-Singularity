import numpy as np

from src.kinematics import get_end_effector_pose


def generate_joint_trajectory(
    q_start,
    d6_start,
    q_end,
    d6_end,
    num_points=200
):
    """
    Generate a linear trajectory in joint space.

    Parameters
    ----------
    q_start : array-like
        Initial values of the five revolute joints [rad].

    d6_start : float
        Initial prismatic joint displacement [m].

    q_end : array-like
        Final values of the five revolute joints [rad].

    d6_end : float
        Final prismatic joint displacement [m].

    num_points : int
        Number of trajectory samples.

    Returns
    -------
    q_trajectory : ndarray, shape (num_points, 5)

    d6_trajectory : ndarray, shape (num_points,)
    """

    q_start = np.asarray(q_start, dtype=float)
    q_end = np.asarray(q_end, dtype=float)

    if q_start.shape != (5,):
        raise ValueError(
            "q_start must contain five revolute joint values."
        )

    if q_end.shape != (5,):
        raise ValueError(
            "q_end must contain five revolute joint values."
        )

    if num_points < 2:
        raise ValueError(
            "num_points must be at least 2."
        )

    q_trajectory = np.linspace(
        q_start,
        q_end,
        num_points
    )

    d6_trajectory = np.linspace(
        d6_start,
        d6_end,
        num_points
    )

    return q_trajectory, d6_trajectory


def calculate_end_effector_trajectory(
    q_trajectory,
    d6_trajectory
):
    """
    Calculate the Cartesian end-effector path produced
    by a joint-space trajectory.

    Returns
    -------
    positions : ndarray, shape (N, 3)
        Cartesian end-effector positions.
    """

    q_trajectory = np.asarray(
        q_trajectory,
        dtype=float
    )

    d6_trajectory = np.asarray(
        d6_trajectory,
        dtype=float
    )

    if q_trajectory.ndim != 2:
        raise ValueError(
            "q_trajectory must be a 2D array."
        )

    if q_trajectory.shape[1] != 5:
        raise ValueError(
            "q_trajectory must contain five revolute joints."
        )

    if len(q_trajectory) != len(d6_trajectory):
        raise ValueError(
            "q_trajectory and d6_trajectory must have equal lengths."
        )

    positions = np.zeros(
        (len(q_trajectory), 3),
        dtype=float
    )

    for i, (q, d6) in enumerate(
        zip(q_trajectory, d6_trajectory)
    ):
        _, position, _ = get_end_effector_pose(
            q,
            d6
        )

        positions[i] = position

    return positions


def calculate_trajectory_length(
    positions
):
    """
    Calculate the Cartesian length of an
    end-effector trajectory.
    """

    positions = np.asarray(
        positions,
        dtype=float
    )

    if positions.ndim != 2:
        raise ValueError(
            "positions must be a 2D array."
        )

    if positions.shape[1] != 3:
        raise ValueError(
            "positions must have shape (N, 3)."
        )

    if len(positions) < 2:
        return 0.0

    differences = np.diff(
        positions,
        axis=0
    )

    segment_lengths = np.linalg.norm(
        differences,
        axis=1
    )

    return float(
        np.sum(segment_lengths)
    )


def generate_end_effector_trajectory(
    q_start,
    d6_start,
    q_end,
    d6_end,
    num_points=200
):
    """
    Convenience function that generates a joint-space
    trajectory and calculates its Cartesian path.
    """

    q_trajectory, d6_trajectory = (
        generate_joint_trajectory(
            q_start,
            d6_start,
            q_end,
            d6_end,
            num_points
        )
    )

    positions = calculate_end_effector_trajectory(
        q_trajectory,
        d6_trajectory
    )

    return (
        q_trajectory,
        d6_trajectory,
        positions
    )
