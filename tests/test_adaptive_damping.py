import numpy as np

from src.kinematics import get_end_effector_pose
from src.jacobian import translational_jacobian

from src.adaptive_damping import (
    calculate_adaptive_damping,
    adaptive_dls_pseudoinverse,
    solve_position_ik_adaptive,
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


# =====================================================
# ADAPTIVE DAMPING TESTS
# =====================================================

def test_damping_is_small_far_from_singularity():

    J = np.hstack([
        np.eye(3),
        np.zeros((3, 3))
    ])

    damping, sigma_min = calculate_adaptive_damping(
        J,
        sigma_threshold=0.10,
        lambda_min=1e-4,
        lambda_max=0.20
    )

    assert np.isclose(
        sigma_min,
        1.0
    )

    assert np.isclose(
        damping,
        1e-4
    )


def test_damping_increases_near_singularity():

    J = np.zeros((3, 6))

    J[0, 0] = 1.0
    J[1, 1] = 0.5
    J[2, 2] = 0.01

    damping, sigma_min = calculate_adaptive_damping(
        J,
        sigma_threshold=0.10,
        lambda_min=1e-4,
        lambda_max=0.20
    )

    assert np.isclose(
        sigma_min,
        0.01
    )

    assert damping > 1e-4

    assert damping <= 0.20


# =====================================================
# PSEUDOINVERSE TEST
# =====================================================

def test_adaptive_dls_pseudoinverse_shape():

    J = translational_jacobian(
        Q_INITIAL,
        D6_INITIAL
    )

    J_dls, damping, sigma_min = (
        adaptive_dls_pseudoinverse(
            J
        )
    )

    assert J_dls.shape == (6, 3)

    assert damping >= 0.0

    assert sigma_min >= 0.0


# =====================================================
# COMPLETE ADAPTIVE IK TEST
# =====================================================

def test_adaptive_ik_converges():

    result = solve_position_ik_adaptive(
        target_position=TARGET_POSITION,
        q_initial=Q_INITIAL,
        d6_initial=D6_INITIAL,
        step_size=0.5,
        tolerance=1e-4,
        max_iterations=1000
    )

    assert result["converged"] is True

    assert result["final_error"] <= 1e-4


def test_adaptive_ik_final_position():

    result = solve_position_ik_adaptive(
        target_position=TARGET_POSITION,
        q_initial=Q_INITIAL,
        d6_initial=D6_INITIAL,
        step_size=0.5,
        tolerance=1e-4,
        max_iterations=1000
    )

    assert np.allclose(
        result["final_position"],
        TARGET_POSITION,
        atol=1e-4
    )


# =====================================================
# HISTORY TESTS
# =====================================================

def test_adaptive_histories_are_recorded():

    result = solve_position_ik_adaptive(
        target_position=TARGET_POSITION,
        q_initial=Q_INITIAL,
        d6_initial=D6_INITIAL
    )

    assert len(
        result["error_history"]
    ) > 1

    assert len(
        result["damping_history"]
    ) > 0

    assert len(
        result["sigma_min_history"]
    ) > 0

    assert (
        len(result["damping_history"])
        ==
        len(result["sigma_min_history"])
    )


def test_error_decreases_with_adaptive_damping():

    result = solve_position_ik_adaptive(
        target_position=TARGET_POSITION,
        q_initial=Q_INITIAL,
        d6_initial=D6_INITIAL
    )

    history = result["error_history"]

    assert history[-1] < history[0]
