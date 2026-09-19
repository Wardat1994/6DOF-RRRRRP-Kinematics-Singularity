import numpy as np

from src.kinematics import get_end_effector_pose

from src.multi_objective_ik import (
    normalize_vector,
    multi_objective_ik_step,
    solve_multi_objective_ik,
)


# =====================================================
# STANDARD INITIAL CONFIGURATION
# =====================================================

Q_INITIAL = np.radians([
    -90.0,
    0.0,
    20.0,
    -10.0,
    0.0
])

D6_INITIAL = 0.15


# =====================================================
# REACHABLE TARGET
# =====================================================

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
# NORMALIZATION TESTS
# =====================================================

def test_normalize_vector():

    vector = np.array([
        3.0,
        4.0,
        0.0,
        0.0,
        0.0,
        0.0
    ])

    result = normalize_vector(
        vector
    )

    assert result.shape == (6,)

    assert np.isclose(
        np.linalg.norm(result),
        1.0
    )


def test_normalize_zero_vector():

    vector = np.zeros(6)

    result = normalize_vector(
        vector
    )

    assert np.allclose(
        result,
        np.zeros(6)
    )


# =====================================================
# SINGLE MULTI-OBJECTIVE STEP
# =====================================================

def test_multi_objective_step():

    result = multi_objective_ik_step(
        q=Q_INITIAL,
        d6=D6_INITIAL,
        target_position=TARGET_POSITION,
        step_size=0.5,
        singularity_gain=0.003,
        joint_limit_gain=0.003
    )

    assert result["q"].shape == (5,)

    assert np.isfinite(
        result["d6"]
    )

    assert result["error"].shape == (3,)

    assert result["primary_delta"].shape == (6,)

    assert result[
        "singularity_command"
    ].shape == (6,)

    assert result[
        "joint_limit_command"
    ].shape == (6,)

    assert result[
        "secondary_command"
    ].shape == (6,)

    assert result[
        "secondary_delta"
    ].shape == (6,)

    assert result[
        "total_delta"
    ].shape == (6,)

    assert result[
        "singularity_gradient"
    ].shape == (6,)

    assert np.isfinite(
        result["sigma_min"]
    )

    assert result[
        "sigma_min"
    ] >= 0.0

    assert np.isfinite(
        result["joint_limit_cost"]
    )

    assert np.isfinite(
        result["joint_limit_margin"]
    )


# =====================================================
# HARD JOINT LIMIT TEST
# =====================================================

def test_multi_objective_step_respects_limits():

    q_near_limits = np.radians([
        179.0,
        89.0,
        89.0,
        -89.0,
        89.0
    ])

    result = multi_objective_ik_step(
        q=q_near_limits,
        d6=0.475,
        target_position=TARGET_POSITION,
        singularity_gain=0.003,
        joint_limit_gain=0.005
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
# SECONDARY OBJECTIVES ACTIVE TEST
# =====================================================

def test_secondary_objectives_can_be_active():

    q_near_limits = np.radians([
        170.0,
        82.0,
        78.0,
        -82.0,
        80.0
    ])

    result = multi_objective_ik_step(
        q=q_near_limits,
        d6=0.46,
        target_position=TARGET_POSITION,
        singularity_gain=0.005,
        joint_limit_gain=0.005,
        joint_limit_activation=0.80
    )

    assert np.linalg.norm(
        result["singularity_command"]
    ) >= 0.0

    assert np.linalg.norm(
        result["joint_limit_command"]
    ) > 0.0

    assert np.all(
        np.isfinite(
            result["secondary_delta"]
        )
    )


# =====================================================
# COMPLETE SOLVER CONVERGENCE
# =====================================================

def test_multi_objective_solver_converges():

    result = solve_multi_objective_ik(
        target_position=TARGET_POSITION,
        q_initial=Q_INITIAL,
        d6_initial=D6_INITIAL,
        step_size=0.5,
        singularity_gain=0.003,
        joint_limit_gain=0.003,
        joint_limit_activation=0.80,
        tolerance=1e-4,
        max_iterations=1000
    )

    assert result[
        "converged"
    ] is True

    assert result[
        "final_error"
    ] <= 1e-4


# =====================================================
# FINAL CARTESIAN POSITION
# =====================================================

def test_multi_objective_final_position():

    result = solve_multi_objective_ik(
        target_position=TARGET_POSITION,
        q_initial=Q_INITIAL,
        d6_initial=D6_INITIAL,
        step_size=0.5,
        singularity_gain=0.003,
        joint_limit_gain=0.003,
        tolerance=1e-4,
        max_iterations=1000
    )

    assert np.allclose(
        result["final_position"],
        TARGET_POSITION,
        atol=1e-4
    )


# =====================================================
# FINAL METRICS
# =====================================================

def test_multi_objective_final_metrics_are_valid():

    result = solve_multi_objective_ik(
        target_position=TARGET_POSITION,
        q_initial=Q_INITIAL,
        d6_initial=D6_INITIAL,
        singularity_gain=0.003,
        joint_limit_gain=0.003
    )

    assert np.isfinite(
        result["final_sigma_min"]
    )

    assert result[
        "final_sigma_min"
    ] >= 0.0

    assert np.isfinite(
        result[
            "final_joint_limit_margin"
        ]
    )

    assert np.isfinite(
        result[
            "final_joint_limit_cost"
        ]
    )

    assert result[
        "final_joint_limit_cost"
    ] >= 0.0


# =====================================================
# HISTORY TESTS
# =====================================================

def test_multi_objective_histories():

    result = solve_multi_objective_ik(
        target_position=TARGET_POSITION,
        q_initial=Q_INITIAL,
        d6_initial=D6_INITIAL,
        singularity_gain=0.003,
        joint_limit_gain=0.003
    )

    number_of_states = len(
        result["error_history"]
    )

    assert number_of_states > 1

    assert len(
        result["position_history"]
    ) == number_of_states

    assert len(
        result["sigma_min_history"]
    ) == number_of_states

    assert len(
        result["joint_limit_cost_history"]
    ) == number_of_states

    assert len(
        result["joint_limit_margin_history"]
    ) == number_of_states

    assert len(
        result["q_history"]
    ) == number_of_states

    assert len(
        result["d6_history"]
    ) == number_of_states


# =====================================================
# ERROR REDUCTION
# =====================================================

def test_multi_objective_error_decreases():

    result = solve_multi_objective_ik(
        target_position=TARGET_POSITION,
        q_initial=Q_INITIAL,
        d6_initial=D6_INITIAL,
        singularity_gain=0.003,
        joint_limit_gain=0.003
    )

    history = result[
        "error_history"
    ]

    assert history[-1] < history[0]
