import numpy as np

from src.kinematics import get_end_effector_pose

from src.joint_limit_avoidance import (
    normalized_joint_position,
    joint_limit_margin,
    joint_limit_cost,
    joint_limit_avoidance_direction,
    joint_limit_avoidance_step,
    solve_position_ik_with_joint_limit_avoidance,
)


# =====================================================
# STANDARD CONFIGURATION
# =====================================================

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
# NORMALIZATION TEST
# =====================================================

def test_joint_center_normalizes_to_zero():

    q_center = np.zeros(5)

    d6_center = (
        0.03 + 0.48
    ) / 2.0

    normalized = normalized_joint_position(
        q_center,
        d6_center
    )

    assert normalized.shape == (6,)

    assert np.allclose(
        normalized,
        np.zeros(6)
    )


# =====================================================
# JOINT LIMIT MARGIN TESTS
# =====================================================

def test_joint_limit_margin_at_center():

    q_center = np.zeros(5)

    d6_center = (
        0.03 + 0.48
    ) / 2.0

    margin = joint_limit_margin(
        q_center,
        d6_center
    )

    assert np.isclose(
        margin,
        1.0
    )


def test_joint_limit_margin_at_limit():

    q = np.zeros(5)

    q[1] = np.radians(
        90.0
    )

    margin = joint_limit_margin(
        q,
        0.255
    )

    assert np.isclose(
        margin,
        0.0
    )


# =====================================================
# JOINT LIMIT COST TESTS
# =====================================================

def test_joint_limit_cost_is_zero_in_safe_region():

    q = np.zeros(5)

    cost = joint_limit_cost(
        q,
        0.255,
        activation_threshold=0.80
    )

    assert np.isclose(
        cost,
        0.0
    )


def test_joint_limit_cost_is_positive_near_limit():

    q = np.zeros(5)

    q[1] = np.radians(
        85.0
    )

    cost = joint_limit_cost(
        q,
        0.255,
        activation_threshold=0.80
    )

    assert cost > 0.0


# =====================================================
# AVOIDANCE DIRECTION TESTS
# =====================================================

def test_upper_limit_generates_negative_direction():

    q = np.zeros(5)

    q[1] = np.radians(
        85.0
    )

    direction = (
        joint_limit_avoidance_direction(
            q,
            0.255,
            activation_threshold=0.80
        )
    )

    # Joint 2 is close to its positive limit,
    # therefore the avoidance direction must
    # move it downward.
    assert direction[1] < 0.0


def test_lower_limit_generates_positive_direction():

    q = np.zeros(5)

    q[1] = np.radians(
        -85.0
    )

    direction = (
        joint_limit_avoidance_direction(
            q,
            0.255,
            activation_threshold=0.80
        )
    )

    # Joint 2 is close to its negative limit,
    # therefore the avoidance direction must
    # move it upward.
    assert direction[1] > 0.0


def test_prismatic_upper_limit_direction():

    q = np.zeros(5)

    direction = (
        joint_limit_avoidance_direction(
            q,
            0.47,
            activation_threshold=0.80
        )
    )

    # d6 is near its maximum limit,
    # therefore it must be pushed downward.
    assert direction[5] < 0.0


# =====================================================
# SINGLE IK STEP TEST
# =====================================================

def test_joint_limit_avoidance_step():

    result = joint_limit_avoidance_step(
        q=Q_INITIAL,
        d6=D6_INITIAL,
        target_position=TARGET_POSITION,
        step_size=0.5,
        avoidance_gain=0.005,
        activation_threshold=0.80
    )

    assert result["q"].shape == (5,)

    assert np.isfinite(
        result["d6"]
    )

    assert result["error"].shape == (3,)

    assert result["primary_delta"].shape == (6,)

    assert result["avoidance_delta"].shape == (6,)

    assert result["total_delta"].shape == (6,)

    assert result[
        "avoidance_direction"
    ].shape == (6,)

    assert np.isfinite(
        result["joint_limit_cost"]
    )

    assert np.isfinite(
        result["joint_limit_margin"]
    )


# =====================================================
# HARD LIMIT TEST
# =====================================================

def test_step_respects_hard_joint_limits():

    q_near_limits = np.radians([
        179.0,
        89.0,
        89.0,
        -89.0,
        89.0
    ])

    result = joint_limit_avoidance_step(
        q=q_near_limits,
        d6=0.475,
        target_position=TARGET_POSITION,
        avoidance_gain=0.005
    )

    q = result["q"]
    d6 = result["d6"]

    q_min = np.radians([
        -180,
        -90,
        -90,
        -90,
        -90
    ])

    q_max = np.radians([
        180,
        90,
        90,
        90,
        90
    ])

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

def test_joint_limit_avoidance_solver_converges():

    result = (
        solve_position_ik_with_joint_limit_avoidance(
            target_position=TARGET_POSITION,
            q_initial=Q_INITIAL,
            d6_initial=D6_INITIAL,
            step_size=0.5,
            avoidance_gain=0.005,
            activation_threshold=0.80,
            tolerance=1e-4,
            max_iterations=1000
        )
    )

    assert result["converged"] is True

    assert result[
        "final_error"
    ] <= 1e-4


# =====================================================
# FINAL POSITION TEST
# =====================================================

def test_final_position_matches_target():

    result = (
        solve_position_ik_with_joint_limit_avoidance(
            target_position=TARGET_POSITION,
            q_initial=Q_INITIAL,
            d6_initial=D6_INITIAL,
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

def test_joint_limit_histories_are_recorded():

    result = (
        solve_position_ik_with_joint_limit_avoidance(
            target_position=TARGET_POSITION,
            q_initial=Q_INITIAL,
            d6_initial=D6_INITIAL
        )
    )

    assert len(
        result["error_history"]
    ) > 1

    assert len(
        result["joint_limit_cost_history"]
    ) > 1

    assert len(
        result["joint_limit_margin_history"]
    ) > 1

    assert len(
        result["sigma_min_history"]
    ) > 1

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
        solve_position_ik_with_joint_limit_avoidance(
            target_position=TARGET_POSITION,
            q_initial=Q_INITIAL,
            d6_initial=D6_INITIAL
        )
    )

    history = result[
        "error_history"
    ]

    assert history[-1] < history[0]
