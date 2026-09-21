import numpy as np
import pytest

from src.kinematics import (
    get_end_effector_pose,
)

from src.obstacle_aware_ik import (
    obstacle_aware_ik_step,
    solve_obstacle_aware_ik,
)


# =====================================================
# STANDARD ROBOT CONFIGURATION
# =====================================================

Q_INITIAL = np.radians([
    -90.0,
    0.0,
    30.0,
    -15.0,
    0.0
])

D6_INITIAL = 0.20


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


_, TARGET_POSITION, _ = (
    get_end_effector_pose(
        Q_TARGET,
        D6_TARGET
    )
)


# =====================================================
# ACTIVE OBSTACLE
# =====================================================
#
# Positioned 0.35 m from Link 3 centerline.
#
# obstacle radius = 0.15 m
# link radius     = 0.05 m
#
# Expected initial surface clearance:
#
# 0.35 - 0.15 - 0.05 = 0.15 m
#
# This lies inside the 0.40 m influence region
# but outside the 0.10 m safety region.
# =====================================================

NEAR_OBSTACLE_CENTER = np.array([
    0.35,
    -1.34641016,
    0.70
])

OBSTACLE_RADIUS = 0.15

LINK_RADIUS = 0.05


# =====================================================
# FAR OBSTACLE
# =====================================================

FAR_OBSTACLE_CENTER = np.array([
    20.0,
    20.0,
    20.0
])


# =====================================================
# SINGLE STEP
# =====================================================

def test_obstacle_aware_step_outputs():

    result = obstacle_aware_ik_step(
        q=Q_INITIAL,
        d6=D6_INITIAL,
        target_position=TARGET_POSITION,
        obstacle_center=(
            NEAR_OBSTACLE_CENTER
        ),
        obstacle_radius=(
            OBSTACLE_RADIUS
        ),
        link_radius=LINK_RADIUS
    )

    assert result["q"].shape == (5,)

    assert np.isfinite(
        result["d6"]
    )

    assert result[
        "error"
    ].shape == (3,)

    assert result[
        "primary_delta"
    ].shape == (6,)

    assert result[
        "singularity_command"
    ].shape == (6,)

    assert result[
        "joint_limit_command"
    ].shape == (6,)

    assert result[
        "obstacle_command"
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
        "obstacle_gradient"
    ].shape == (6,)

    assert np.all(
        np.isfinite(
            result[
                "obstacle_gradient"
            ]
        )
    )


# =====================================================
# OBSTACLE ACTIVATION
# =====================================================

def test_near_obstacle_is_active():

    result = obstacle_aware_ik_step(
        q=Q_INITIAL,
        d6=D6_INITIAL,
        target_position=TARGET_POSITION,
        obstacle_center=(
            NEAR_OBSTACLE_CENTER
        ),
        obstacle_radius=(
            OBSTACLE_RADIUS
        ),
        link_radius=LINK_RADIUS,
        obstacle_safety_distance=0.10,
        obstacle_influence_distance=0.40
    )

    assert (
        result[
            "obstacle_activation"
        ]
        > 0.0
    )

    assert (
        result[
            "obstacle_activation"
        ]
        <= 1.0
    )

    assert (
        result[
            "obstacle_clearance"
        ]
        > 0.0
    )

    assert result[
        "obstacle_collision"
    ] is False


# =====================================================
# EXPECTED INITIAL CLEARANCE
# =====================================================

def test_near_obstacle_initial_clearance():

    result = obstacle_aware_ik_step(
        q=Q_INITIAL,
        d6=D6_INITIAL,
        target_position=TARGET_POSITION,
        obstacle_center=(
            NEAR_OBSTACLE_CENTER
        ),
        obstacle_radius=(
            OBSTACLE_RADIUS
        ),
        link_radius=LINK_RADIUS
    )

    assert np.isclose(
        result[
            "obstacle_clearance"
        ],
        0.15,
        atol=1e-6
    )

    assert result[
        "closest_obstacle_link"
    ] == 3


# =====================================================
# OBSTACLE COMMAND ACTIVE
# =====================================================

def test_near_obstacle_generates_avoidance_command():

    result = obstacle_aware_ik_step(
        q=Q_INITIAL,
        d6=D6_INITIAL,
        target_position=TARGET_POSITION,
        obstacle_center=(
            NEAR_OBSTACLE_CENTER
        ),
        obstacle_radius=(
            OBSTACLE_RADIUS
        ),
        link_radius=LINK_RADIUS,
        obstacle_gain=0.01
    )

    activity = np.linalg.norm(
        result[
            "obstacle_command"
        ]
    )

    assert activity > 0.0

    assert np.all(
        np.isfinite(
            result[
                "obstacle_command"
            ]
        )
    )


# =====================================================
# FAR OBSTACLE
# =====================================================

def test_far_obstacle_is_inactive():

    result = obstacle_aware_ik_step(
        q=Q_INITIAL,
        d6=D6_INITIAL,
        target_position=TARGET_POSITION,
        obstacle_center=(
            FAR_OBSTACLE_CENTER
        ),
        obstacle_radius=(
            OBSTACLE_RADIUS
        ),
        link_radius=LINK_RADIUS
    )

    assert np.isclose(
        result[
            "obstacle_activation"
        ],
        0.0
    )

    assert np.allclose(
        result[
            "obstacle_command"
        ],
        np.zeros(6)
    )

    assert np.allclose(
        result[
            "obstacle_gradient"
        ],
        np.zeros(6)
    )


# =====================================================
# PROTECTED JOINT LIMITS
# =====================================================

def test_obstacle_aware_step_respects_joint_limits():

    q_near_limits = np.radians([
        179.0,
        89.0,
        89.0,
        -89.0,
        89.0
    ])

    result = obstacle_aware_ik_step(
        q=q_near_limits,
        d6=0.475,
        target_position=TARGET_POSITION,
        obstacle_center=(
            FAR_OBSTACLE_CENTER
        ),
        obstacle_radius=(
            OBSTACLE_RADIUS
        )
    )

    q = result["q"]

    d6 = result["d6"]

    q_min = np.radians([
        -180.0,
        -90.0,
        -90.0,
        -90.0,
        -90.0
    ])

    q_max = np.radians([
        180.0,
        90.0,
        90.0,
        90.0,
        90.0
    ])

    assert np.all(
        q >= q_min
    )

    assert np.all(
        q <= q_max
    )

    assert (
        0.03
        <= d6
        <= 0.48
    )


# =====================================================
# COMPLETE SOLVER WITH FAR OBSTACLE
# =====================================================

def test_obstacle_aware_solver_converges_with_far_obstacle():

    result = solve_obstacle_aware_ik(
        target_position=TARGET_POSITION,
        q_initial=Q_INITIAL,
        d6_initial=D6_INITIAL,

        obstacle_center=(
            FAR_OBSTACLE_CENTER
        ),

        obstacle_radius=(
            OBSTACLE_RADIUS
        ),

        link_radius=LINK_RADIUS,

        tolerance=1e-4,
        max_iterations=1000
    )

    assert result[
        "converged"
    ] is True

    assert (
        result[
            "final_error"
        ]
        <= 1e-4
    )

    assert result[
        "collision"
    ] is False


# =====================================================
# SOLVER HISTORIES
# =====================================================

def test_obstacle_aware_histories():

    result = solve_obstacle_aware_ik(
        target_position=TARGET_POSITION,
        q_initial=Q_INITIAL,
        d6_initial=D6_INITIAL,

        obstacle_center=(
            FAR_OBSTACLE_CENTER
        ),

        obstacle_radius=(
            OBSTACLE_RADIUS
        ),

        max_iterations=1000
    )

    number_of_states = len(
        result[
            "error_history"
        ]
    )

    assert number_of_states > 1

    assert len(
        result[
            "position_history"
        ]
    ) == number_of_states

    assert len(
        result[
            "sigma_min_history"
        ]
    ) == number_of_states

    assert len(
        result[
            "obstacle_clearance_history"
        ]
    ) == number_of_states

    assert len(
        result[
            "obstacle_activation_history"
        ]
    ) == number_of_states

    assert len(
        result[
            "closest_obstacle_link_history"
        ]
    ) == number_of_states

    assert len(
        result[
            "q_history"
        ]
    ) == number_of_states

    assert len(
        result[
            "d6_history"
        ]
    ) == number_of_states


# =====================================================
# CLEARANCE HISTORY VALIDITY
# =====================================================

def test_obstacle_clearance_history_is_finite():

    result = solve_obstacle_aware_ik(
        target_position=TARGET_POSITION,
        q_initial=Q_INITIAL,
        d6_initial=D6_INITIAL,

        obstacle_center=(
            FAR_OBSTACLE_CENTER
        ),

        obstacle_radius=(
            OBSTACLE_RADIUS
        )
    )

    history = result[
        "obstacle_clearance_history"
    ]

    assert np.all(
        np.isfinite(
            history
        )
    )

    assert np.all(
        history > 0.0
    )


# =====================================================
# INPUT VALIDATION
# =====================================================

def test_negative_obstacle_gain_rejected():

    with pytest.raises(
        ValueError
    ):

        obstacle_aware_ik_step(
            q=Q_INITIAL,
            d6=D6_INITIAL,
            target_position=(
                TARGET_POSITION
            ),
            obstacle_center=(
                NEAR_OBSTACLE_CENTER
            ),
            obstacle_radius=(
                OBSTACLE_RADIUS
            ),
            obstacle_gain=-0.01
        )


def test_invalid_obstacle_distance_range_rejected():

    with pytest.raises(
        ValueError
    ):

        obstacle_aware_ik_step(
            q=Q_INITIAL,
            d6=D6_INITIAL,
            target_position=(
                TARGET_POSITION
            ),
            obstacle_center=(
                NEAR_OBSTACLE_CENTER
            ),
            obstacle_radius=(
                OBSTACLE_RADIUS
            ),
            obstacle_safety_distance=0.40,
            obstacle_influence_distance=0.20
        )
