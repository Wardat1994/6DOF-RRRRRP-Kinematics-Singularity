import numpy as np
import pytest

from src.kinematics import (
    get_end_effector_pose,
)

from src.multi_obstacle_aware_ik import (
    multi_obstacle_aware_ik_step,
    solve_multi_obstacle_aware_ik,
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
# REACHABLE CARTESIAN TARGET
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
# COMMON PARAMETERS
# =====================================================

LINK_RADIUS = 0.05

SAFETY_DISTANCE = 0.10

INFLUENCE_DISTANCE = 0.40


# =====================================================
# TWO ACTIVE OBSTACLES + ONE FAR OBSTACLE
# =====================================================

ACTIVE_OBSTACLES = [
    {
        "name": "Near Link 3",
        "center": [
            0.35,
            -1.34641016,
            0.70
        ],
        "radius": 0.15,
    },
    {
        "name": "Near Link 4",
        "center": [
            0.33,
            -1.98259807,
            0.97764570
        ],
        "radius": 0.10,
    },
    {
        "name": "Far Obstacle",
        "center": [
            20.0,
            20.0,
            20.0
        ],
        "radius": 0.20,
    },
]


# =====================================================
# FAR OBSTACLES
# =====================================================

FAR_OBSTACLES = [
    {
        "name": "Far 1",
        "center": [
            20.0,
            20.0,
            20.0
        ],
        "radius": 0.20,
    },
    {
        "name": "Far 2",
        "center": [
            -20.0,
            20.0,
            15.0
        ],
        "radius": 0.20,
    },
]


# =====================================================
# SINGLE STEP OUTPUTS
# =====================================================

def test_multi_obstacle_step_outputs():

    result = (
        multi_obstacle_aware_ik_step(
            q=Q_INITIAL,
            d6=D6_INITIAL,
            target_position=TARGET_POSITION,
            obstacles=ACTIVE_OBSTACLES,
            link_radius=LINK_RADIUS
        )
    )

    assert result[
        "q"
    ].shape == (5,)

    assert np.isfinite(
        result[
            "d6"
        ]
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
        "combined_obstacle_direction"
    ].shape == (6,)

    assert np.all(
        np.isfinite(
            result[
                "total_delta"
            ]
        )
    )


# =====================================================
# MULTIPLE ACTIVE OBSTACLES
# =====================================================

def test_multiple_obstacles_are_active():

    result = (
        multi_obstacle_aware_ik_step(
            q=Q_INITIAL,
            d6=D6_INITIAL,
            target_position=TARGET_POSITION,
            obstacles=ACTIVE_OBSTACLES,
            link_radius=LINK_RADIUS,
            obstacle_safety_distance=(
                SAFETY_DISTANCE
            ),
            obstacle_influence_distance=(
                INFLUENCE_DISTANCE
            )
        )
    )

    assert result[
        "active_obstacle_count"
    ] == 2

    assert result[
        "critical_obstacle_name"
    ] == "Near Link 3"

    assert result[
        "critical_obstacle_number"
    ] == 1

    assert result[
        "critical_link_number"
    ] == 3

    assert np.isclose(
        result[
            "minimum_obstacle_clearance"
        ],
        0.15,
        atol=1e-6
    )


# =====================================================
# SINGLE STEP SAFETY
# =====================================================

def test_multi_obstacle_step_is_sampled_safe():

    result = (
        multi_obstacle_aware_ik_step(
            q=Q_INITIAL,
            d6=D6_INITIAL,
            target_position=TARGET_POSITION,
            obstacles=ACTIVE_OBSTACLES,
            link_radius=LINK_RADIUS,

            obstacle_gain=0.03,

            obstacle_safety_distance=(
                SAFETY_DISTANCE
            ),

            obstacle_influence_distance=(
                INFLUENCE_DISTANCE
            ),

            use_safety_filter=True,

            safety_filter_samples=20
        )
    )

    assert (
        result[
            "minimum_path_clearance"
        ]
        >= (
            SAFETY_DISTANCE
            - 1e-9
        )
    )

    assert result[
        "obstacle_collision"
    ] is False

    assert result[
        "path_critical_obstacle_number"
    ] is not None

    assert result[
        "path_critical_link_number"
    ] is not None


# =====================================================
# JOINT LIMIT PROTECTION
# =====================================================

def test_multi_obstacle_step_respects_joint_limits():

    q_near_limits = np.radians([
        179.0,
        89.0,
        89.0,
        -89.0,
        89.0
    ])

    result = (
        multi_obstacle_aware_ik_step(
            q=q_near_limits,
            d6=0.475,
            target_position=TARGET_POSITION,
            obstacles=FAR_OBSTACLES,
            link_radius=LINK_RADIUS
        )
    )

    q = result[
        "q"
    ]

    d6 = result[
        "d6"
    ]

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
# COMPLETE SOLVER WITH FAR OBSTACLES
# =====================================================

def test_solver_converges_with_far_obstacles():

    result = (
        solve_multi_obstacle_aware_ik(
            target_position=TARGET_POSITION,

            q_initial=Q_INITIAL,

            d6_initial=D6_INITIAL,

            obstacles=FAR_OBSTACLES,

            link_radius=LINK_RADIUS,

            tolerance=1e-4,

            max_iterations=1000
        )
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

    assert np.allclose(
        result[
            "final_position"
        ],
        TARGET_POSITION,
        atol=1e-4
    )


# =====================================================
# SOLVER HISTORIES
# =====================================================

def test_multi_obstacle_solver_histories():

    result = (
        solve_multi_obstacle_aware_ik(
            target_position=TARGET_POSITION,
            q_initial=Q_INITIAL,
            d6_initial=D6_INITIAL,
            obstacles=FAR_OBSTACLES,
            link_radius=LINK_RADIUS,
            tolerance=1e-4,
            max_iterations=1000
        )
    )

    number_of_states = len(
        result[
            "error_history"
        ]
    )

    number_of_steps = len(
        result[
            "minimum_path_clearance_history"
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
            "minimum_obstacle_clearance_history"
        ]
    ) == number_of_states

    assert len(
        result[
            "critical_obstacle_number_history"
        ]
    ) == number_of_states

    assert len(
        result[
            "critical_obstacle_name_history"
        ]
    ) == number_of_states

    assert len(
        result[
            "critical_link_number_history"
        ]
    ) == number_of_states

    assert len(
        result[
            "active_obstacle_count_history"
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

    assert (
        number_of_steps
        <= number_of_states
    )

    assert (
        number_of_steps
        >= number_of_states - 1
    )

    assert len(
        result[
            "safety_filter_corrected_history"
        ]
    ) == number_of_steps

    assert len(
        result[
            "safety_filter_accepted_history"
        ]
    ) == number_of_steps


# =====================================================
# TRAJECTORY SAFETY
# =====================================================

def test_solver_reports_safe_trajectory():

    result = (
        solve_multi_obstacle_aware_ik(
            target_position=TARGET_POSITION,

            q_initial=Q_INITIAL,

            d6_initial=D6_INITIAL,

            obstacles=FAR_OBSTACLES,

            link_radius=LINK_RADIUS,

            obstacle_safety_distance=(
                SAFETY_DISTANCE
            ),

            use_safety_filter=True,

            tolerance=1e-4,

            max_iterations=1000
        )
    )

    assert result[
        "trajectory_safety_satisfied"
    ] is True

    assert (
        result[
            "minimum_trajectory_clearance"
        ]
        >= SAFETY_DISTANCE
    )

    assert (
        result[
            "final_obstacle_clearance"
        ]
        >= SAFETY_DISTANCE
    )

    assert result[
        "collision"
    ] is False


# =====================================================
# INITIAL UNSAFE CONFIGURATION
# =====================================================

def test_initial_unsafe_configuration_rejected():

    unsafe_obstacles = [
        {
            "name": "Base Collision",
            "center": [
                0.0,
                0.0,
                0.0
            ],
            "radius": 0.10,
        },
        {
            "name": "Far",
            "center": [
                20.0,
                20.0,
                20.0
            ],
            "radius": 0.10,
        },
    ]

    with pytest.raises(
        ValueError
    ):

        solve_multi_obstacle_aware_ik(
            target_position=TARGET_POSITION,

            q_initial=Q_INITIAL,

            d6_initial=D6_INITIAL,

            obstacles=unsafe_obstacles,

            link_radius=LINK_RADIUS,

            obstacle_safety_distance=(
                SAFETY_DISTANCE
            ),

            use_safety_filter=True
        )


# =====================================================
# INVALID OBSTACLE GAIN
# =====================================================

def test_negative_obstacle_gain_rejected():

    with pytest.raises(
        ValueError
    ):

        multi_obstacle_aware_ik_step(
            q=Q_INITIAL,
            d6=D6_INITIAL,

            target_position=(
                TARGET_POSITION
            ),

            obstacles=(
                ACTIVE_OBSTACLES
            ),

            obstacle_gain=-0.01
        )


# =====================================================
# INVALID SAFETY / INFLUENCE RANGE
# =====================================================

def test_invalid_obstacle_distance_range_rejected():

    with pytest.raises(
        ValueError
    ):

        multi_obstacle_aware_ik_step(
            q=Q_INITIAL,
            d6=D6_INITIAL,

            target_position=(
                TARGET_POSITION
            ),

            obstacles=(
                ACTIVE_OBSTACLES
            ),

            obstacle_safety_distance=0.40,

            obstacle_influence_distance=0.20
        )


# =====================================================
# EMPTY OBSTACLE COLLECTION
# =====================================================

def test_empty_obstacle_list_rejected():

    with pytest.raises(
        ValueError
    ):

        multi_obstacle_aware_ik_step(
            q=Q_INITIAL,
            d6=D6_INITIAL,

            target_position=(
                TARGET_POSITION
            ),

            obstacles=[]
        )
