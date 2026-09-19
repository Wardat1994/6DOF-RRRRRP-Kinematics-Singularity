import numpy as np

from src.kinematics import get_end_effector_pose
from src.jacobian import translational_jacobian

from src.inverse_kinematics import (
    damped_least_squares_pseudoinverse,
    inverse_kinematics_step,
    solve_position_ik,
)


Q_INITIAL = np.radians([
    -90.0,
    0.0,
    20.0,
    -10.0,
    0.0
])

D6_INITIAL = 0.15


Q_TARGET = np.radians([
    -70.0,
    10.0,
    30.0,
    -20.0,
    15.0
])

D6_TARGET = 0.22


_, TARGET_POSITION, _ = get_end_effector_pose(
    Q_TARGET,
    D6_TARGET
)


def test_dls_pseudoinverse_shape():

    J = translational_jacobian(
        Q_INITIAL,
        D6_INITIAL
    )

    J_dls = damped_least_squares_pseudoinverse(
        J,
        damping=0.05
    )

    assert J_dls.shape == (6, 3)


def test_inverse_kinematics_step_reduces_error():

    _, current_position, _ = get_end_effector_pose(
        Q_INITIAL,
        D6_INITIAL
    )

    initial_error = np.linalg.norm(
        TARGET_POSITION - current_position
    )

    q_new, d6_new, _, _ = inverse_kinematics_step(
        Q_INITIAL,
        D6_INITIAL,
        TARGET_POSITION,
        damping=0.05,
        step_size=0.5
    )

    _, new_position, _ = get_end_effector_pose(
        q_new,
        d6_new
    )

    new_error = np.linalg.norm(
        TARGET_POSITION - new_position
    )

    assert new_error < initial_error


def test_position_ik_converges():

    result = solve_position_ik(
        target_position=TARGET_POSITION,
        q_initial=Q_INITIAL,
        d6_initial=D6_INITIAL,
        damping=0.05,
        step_size=0.5,
        tolerance=1e-4,
        max_iterations=1000
    )

    assert result["converged"] is True
    assert result["final_error"] <= 1e-4


def test_final_position_matches_target():

    result = solve_position_ik(
        target_position=TARGET_POSITION,
        q_initial=Q_INITIAL,
        d6_initial=D6_INITIAL,
        damping=0.05,
        step_size=0.5,
        tolerance=1e-4,
        max_iterations=1000
    )

    assert np.allclose(
        result["final_position"],
        TARGET_POSITION,
        atol=1e-4
    )


def test_joint_limits_respected():

    result = solve_position_ik(
        target_position=TARGET_POSITION,
        q_initial=Q_INITIAL,
        d6_initial=D6_INITIAL
    )

    q = result["q"]
    d6 = result["d6"]

    assert np.all(
        q >= np.radians(
            [-180, -90, -90, -90, -90]
        )
    )

    assert np.all(
        q <= np.radians(
            [180, 90, 90, 90, 90]
        )
    )

    assert 0.03 <= d6 <= 0.48


def test_error_history_decreases():

    result = solve_position_ik(
        target_position=TARGET_POSITION,
        q_initial=Q_INITIAL,
        d6_initial=D6_INITIAL
    )

    history = result["error_history"]

    assert len(history) > 1
    assert history[-1] < history[0]
