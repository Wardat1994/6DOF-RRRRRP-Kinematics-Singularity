import numpy as np

from src.kinematics import get_end_effector_pose

from src.singularity_avoidance import (
    translational_sigma_min,
    sigma_min_gradient,
    singularity_avoidance_step,
    solve_position_ik_with_singularity_avoidance,
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
    -80.0,
    8.0,
    28.0,
    -16.0,
    10.0
])

D6_TARGET = 0.20


_, TARGET_POSITION, _ = get_end_effector_pose(
    Q_TARGET,
    D6_TARGET
)


# =====================================================
# SIGMA MIN TEST
# =====================================================

def test_translational_sigma_min_is_valid():

    sigma_min = translational_sigma_min(
        Q_INITIAL,
        D6_INITIAL
    )

    assert np.isfinite(sigma_min)
    assert sigma_min >= 0.0


# =====================================================
# GRADIENT TESTS
# =====================================================

def test_sigma_min_gradient_shape():

    gradient = sigma_min_gradient(
        Q_INITIAL,
        D6_INITIAL
    )

    assert gradient.shape == (6,)


def test_sigma_min_gradient_is_finite():

    gradient = sigma_min_gradient(
        Q_INITIAL,
        D6_INITIAL
    )

    assert np.all(
        np.isfinite(gradient)
    )


# =====================================================
# SINGLE AVOIDANCE STEP TEST
# =====================================================

def test_singularity_avoidance_step():

    result = singularity_avoidance_step(
        q=Q_INITIAL,
        d6=D6_INITIAL,
        target_position=TARGET_POSITION,
        step_size=0.5,
        avoidance_gain=0.005
    )

    assert result["q"].shape == (5,)

    assert np.isfinite(
        result["d6"]
    )

    assert result["error"].shape == (3,)

    assert result["gradient"].shape == (6,)

    assert result["primary_delta"].shape == (6,)

    assert result["avoidance_delta"].shape == (6,)

    assert result["total_delta"].shape == (6,)

    assert result["sigma_min"] >= 0.0

    assert result["damping"] >= 0.0


# =====================================================
# JOINT LIMIT TEST
# =====================================================

def test_singularity_avoidance_respects_joint_limits():

    result = singularity_avoidance_step(
        q=Q_INITIAL,
        d6=D6_INITIAL,
        target_position=TARGET_POSITION,
        avoidance_gain=0.005
    )

    q = result["q"]
    d6 = result["d6"]

    q_min = np.radians(
        [-180, -90, -90, -90, -90]
    )

    q_max = np.radians(
        [180, 90, 90, 90, 90]
    )

    assert np.all(
        q >= q_min
    )

    assert np.all(
        q <= q_max
    )

    assert 0.03 <= d6 <= 0.48


# =====================================================
# COMPLETE IK SOLVER TEST
# =====================================================

def test_singularity_avoidance_ik_converges():

    result = (
        solve_position_ik_with_singularity_avoidance(
            target_position=TARGET_POSITION,
            q_initial=Q_INITIAL,
            d6_initial=D6_INITIAL,
            step_size=0.5,
            avoidance_gain=0.005,
            tolerance=1e-4,
            max_iterations=1000
        )
    )

    assert result["converged"] is True

    assert result["final_error"] <= 1e-4


# =====================================================
# FINAL POSITION TEST
# =====================================================

def test_final_position_matches_target():

    result = (
        solve_position_ik_with_singularity_avoidance(
            target_position=TARGET_POSITION,
            q_initial=Q_INITIAL,
            d6_initial=D6_INITIAL,
            step_size=0.5,
            avoidance_gain=0.005,
            tolerance=1e-4,
            max_iterations=1000
        )
    )

    assert np.allclose(
        result["final_position"],
        TARGET_POSITION,
        atol=1e-4
    )


# =====================================================
# HISTORY TEST
# =====================================================

def test_singularity_avoidance_histories():

    result = (
        solve_position_ik_with_singularity_avoidance(
            target_position=TARGET_POSITION,
            q_initial=Q_INITIAL,
            d6_initial=D6_INITIAL,
            avoidance_gain=0.005
        )
    )

    assert len(
        result["error_history"]
    ) > 1

    assert len(
        result["sigma_min_history"]
    ) > 1

    assert len(
        result["damping_history"]
    ) > 0

    assert len(
        result["avoidance_history"]
    ) > 0

    assert (
        len(result["q_history"])
        ==
        len(result["d6_history"])
    )


# =====================================================
# ERROR REDUCTION TEST
# =====================================================

def test_error_decreases():

    result = (
        solve_position_ik_with_singularity_avoidance(
            target_position=TARGET_POSITION,
            q_initial=Q_INITIAL,
            d6_initial=D6_INITIAL,
            avoidance_gain=0.005
        )
    )

    history = result["error_history"]

    assert history[-1] < history[0]
