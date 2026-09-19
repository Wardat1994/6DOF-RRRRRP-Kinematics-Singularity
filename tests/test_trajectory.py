import numpy as np

from src.trajectory import (
    generate_joint_trajectory,
    calculate_end_effector_trajectory,
    calculate_trajectory_length,
    generate_end_effector_trajectory,
)


Q_START = np.radians([
    -90.0,
    0.0,
    20.0,
    -10.0,
    0.0
])

Q_END = np.radians([
    -60.0,
    20.0,
    45.0,
    -25.0,
    30.0
])

D6_START = 0.10
D6_END = 0.30


def test_generate_joint_trajectory_shapes():

    q_traj, d6_traj = generate_joint_trajectory(
        Q_START,
        D6_START,
        Q_END,
        D6_END,
        num_points=100
    )

    assert q_traj.shape == (100, 5)
    assert d6_traj.shape == (100,)


def test_joint_trajectory_start_and_end():

    q_traj, d6_traj = generate_joint_trajectory(
        Q_START,
        D6_START,
        Q_END,
        D6_END,
        num_points=50
    )

    assert np.allclose(
        q_traj[0],
        Q_START
    )

    assert np.allclose(
        q_traj[-1],
        Q_END
    )

    assert np.isclose(
        d6_traj[0],
        D6_START
    )

    assert np.isclose(
        d6_traj[-1],
        D6_END
    )


def test_end_effector_trajectory_shape():

    q_traj, d6_traj = generate_joint_trajectory(
        Q_START,
        D6_START,
        Q_END,
        D6_END,
        num_points=80
    )

    positions = calculate_end_effector_trajectory(
        q_traj,
        d6_traj
    )

    assert positions.shape == (80, 3)


def test_trajectory_length_positive():

    q_traj, d6_traj = generate_joint_trajectory(
        Q_START,
        D6_START,
        Q_END,
        D6_END,
        num_points=100
    )

    positions = calculate_end_effector_trajectory(
        q_traj,
        d6_traj
    )

    length = calculate_trajectory_length(
        positions
    )

    assert length > 0.0


def test_stationary_trajectory_length_zero():

    q_traj, d6_traj = generate_joint_trajectory(
        Q_START,
        D6_START,
        Q_START,
        D6_START,
        num_points=20
    )

    positions = calculate_end_effector_trajectory(
        q_traj,
        d6_traj
    )

    length = calculate_trajectory_length(
        positions
    )

    assert np.isclose(
        length,
        0.0,
        atol=1e-12
    )


def test_complete_trajectory_function():

    q_traj, d6_traj, positions = (
        generate_end_effector_trajectory(
            Q_START,
            D6_START,
            Q_END,
            D6_END,
            num_points=120
        )
    )

    assert q_traj.shape == (120, 5)
    assert d6_traj.shape == (120,)
    assert positions.shape == (120, 3)
