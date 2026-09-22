import numpy as np
import pytest

from src.multi_obstacle_avoidance import (
    normalize_obstacles,
    multi_obstacle_clearances,
    minimum_multi_obstacle_clearance,
    robot_in_collision_with_obstacles,
    multi_obstacle_states,
    combined_obstacle_avoidance_direction,
    multi_obstacle_avoidance_state,
)


# =====================================================
# STANDARD ROBOT CONFIGURATION
# =====================================================

Q_TEST = np.radians([
    -90.0,
    0.0,
    30.0,
    -15.0,
    0.0
])

D6_TEST = 0.20

LINK_RADIUS = 0.05


# =====================================================
# TEST OBSTACLES
# =====================================================
#
# Obstacle 1:
# Near Link 3.
#
# Expected clearance:
# approximately 0.15 m.
#
# Obstacle 2:
# Near Link 4.
#
# Expected clearance:
# approximately 0.18 m.
#
# Obstacle 3:
# Far from the robot.
# =====================================================

OBSTACLES = [
    {
        "name": "Near Link 3",
        "center": np.array([
            0.35,
            -1.34641016,
            0.70
        ]),
        "radius": 0.15,
    },
    {
        "name": "Near Link 4",
        "center": np.array([
            0.33,
            -1.98259807,
            0.97764570
        ]),
        "radius": 0.10,
    },
    {
        "name": "Far Obstacle",
        "center": np.array([
            20.0,
            20.0,
            20.0
        ]),
        "radius": 0.20,
    },
]


# =====================================================
# NORMALIZATION
# =====================================================

def test_normalize_obstacles():

    normalized = normalize_obstacles(
        OBSTACLES
    )

    assert len(normalized) == 3

    assert normalized[0][
        "number"
    ] == 1

    assert normalized[1][
        "number"
    ] == 2

    assert normalized[2][
        "number"
    ] == 3

    assert normalized[0][
        "name"
    ] == "Near Link 3"

    assert normalized[0][
        "center"
    ].shape == (3,)

    assert np.isclose(
        normalized[0][
            "radius"
        ],
        0.15
    )


# =====================================================
# DEFAULT OBSTACLE NAME
# =====================================================

def test_default_obstacle_name():

    obstacles = [
        {
            "center": [
                1.0,
                2.0,
                3.0
            ],
            "radius": 0.1,
        }
    ]

    normalized = (
        normalize_obstacles(
            obstacles
        )
    )

    assert normalized[0][
        "name"
    ] == "Obstacle 1"


# =====================================================
# CLEARANCE TO EVERY OBSTACLE
# =====================================================

def test_multi_obstacle_clearances():

    results = (
        multi_obstacle_clearances(
            q=Q_TEST,
            d6=D6_TEST,
            obstacles=OBSTACLES,
            link_radius=(
                LINK_RADIUS
            )
        )
    )

    assert len(results) == 3

    for result in results:

        assert np.isfinite(
            result[
                "clearance"
            ]
        )

        assert result[
            "closest_point"
        ].shape == (3,)

        assert (
            1
            <= result[
                "link_number"
            ]
            <= 6
        )


# =====================================================
# KNOWN CLEARANCES
# =====================================================

def test_known_near_obstacle_clearances():

    results = (
        multi_obstacle_clearances(
            q=Q_TEST,
            d6=D6_TEST,
            obstacles=OBSTACLES,
            link_radius=(
                LINK_RADIUS
            )
        )
    )

    assert np.isclose(
        results[0][
            "clearance"
        ],
        0.15,
        atol=1e-6
    )

    assert results[0][
        "link_number"
    ] == 3

    assert np.isclose(
        results[1][
            "clearance"
        ],
        0.18,
        atol=1e-6
    )

    assert results[1][
        "link_number"
    ] == 4


# =====================================================
# GLOBAL CRITICAL OBSTACLE
# =====================================================

def test_minimum_multi_obstacle_clearance():

    result = (
        minimum_multi_obstacle_clearance(
            q=Q_TEST,
            d6=D6_TEST,
            obstacles=OBSTACLES,
            link_radius=(
                LINK_RADIUS
            )
        )
    )

    assert result[
        "obstacle_number"
    ] == 1

    assert result[
        "obstacle_name"
    ] == "Near Link 3"

    assert result[
        "link_number"
    ] == 3

    assert np.isclose(
        result[
            "clearance"
        ],
        0.15,
        atol=1e-6
    )


# =====================================================
# NO COLLISION
# =====================================================

def test_robot_not_in_collision():

    collision = (
        robot_in_collision_with_obstacles(
            q=Q_TEST,
            d6=D6_TEST,
            obstacles=OBSTACLES,
            link_radius=(
                LINK_RADIUS
            )
        )
    )

    assert collision is False


# =====================================================
# COLLISION WITH ONE OF MANY OBSTACLES
# =====================================================

def test_robot_collision_with_one_obstacle():

    obstacles = (
        OBSTACLES
        + [
            {
                "name": "Base Collision",
                "center": [
                    0.0,
                    0.0,
                    0.0
                ],
                "radius": 0.10,
            }
        ]
    )

    collision = (
        robot_in_collision_with_obstacles(
            q=Q_TEST,
            d6=D6_TEST,
            obstacles=obstacles,
            link_radius=(
                LINK_RADIUS
            )
        )
    )

    assert collision is True

    critical = (
        minimum_multi_obstacle_clearance(
            q=Q_TEST,
            d6=D6_TEST,
            obstacles=obstacles,
            link_radius=(
                LINK_RADIUS
            )
        )
    )

    assert critical[
        "obstacle_name"
    ] == "Base Collision"

    assert (
        critical[
            "clearance"
        ]
        <= 0.0
    )


# =====================================================
# PER-OBSTACLE STATES
# =====================================================

def test_multi_obstacle_states():

    states = (
        multi_obstacle_states(
            q=Q_TEST,
            d6=D6_TEST,
            obstacles=OBSTACLES,
            link_radius=(
                LINK_RADIUS
            ),
            safety_distance=0.10,
            influence_distance=0.40
        )
    )

    assert len(states) == 3

    for state in states:

        assert (
            0.0
            <= state[
                "activation"
            ]
            <= 1.0
        )

        assert state[
            "gradient"
        ].shape == (6,)

        assert state[
            "direction"
        ].shape == (6,)

        assert np.all(
            np.isfinite(
                state[
                    "gradient"
                ]
            )
        )

        assert np.all(
            np.isfinite(
                state[
                    "direction"
                ]
            )
        )


# =====================================================
# ACTIVE OBSTACLE COUNT
# =====================================================

def test_active_obstacle_count():

    result = (
        combined_obstacle_avoidance_direction(
            q=Q_TEST,
            d6=D6_TEST,
            obstacles=OBSTACLES,
            link_radius=(
                LINK_RADIUS
            ),
            safety_distance=0.10,
            influence_distance=0.40
        )
    )

    # First two obstacles are inside the
    # influence region.
    #
    # The third is far away.
    assert result[
        "active_obstacle_count"
    ] == 2


# =====================================================
# COMBINED AVOIDANCE DIRECTION
# =====================================================

def test_combined_avoidance_direction():

    result = (
        combined_obstacle_avoidance_direction(
            q=Q_TEST,
            d6=D6_TEST,
            obstacles=OBSTACLES,
            link_radius=(
                LINK_RADIUS
            ),
            safety_distance=0.10,
            influence_distance=0.40
        )
    )

    direction = result[
        "direction"
    ]

    raw_direction = result[
        "raw_direction"
    ]

    assert direction.shape == (6,)

    assert raw_direction.shape == (6,)

    assert np.all(
        np.isfinite(
            direction
        )
    )

    assert np.all(
        np.isfinite(
            raw_direction
        )
    )

    assert (
        np.linalg.norm(
            raw_direction
        )
        > 0.0
    )

    assert np.isclose(
        np.linalg.norm(
            direction
        ),
        1.0
    )


# =====================================================
# ALL OBSTACLES FAR AWAY
# =====================================================

def test_all_far_obstacles_produce_zero_direction():

    far_obstacles = [
        {
            "center": [
                20.0,
                20.0,
                20.0
            ],
            "radius": 0.10,
        },
        {
            "center": [
                -20.0,
                20.0,
                20.0
            ],
            "radius": 0.10,
        },
    ]

    result = (
        combined_obstacle_avoidance_direction(
            q=Q_TEST,
            d6=D6_TEST,
            obstacles=far_obstacles,
            link_radius=(
                LINK_RADIUS
            )
        )
    )

    assert result[
        "active_obstacle_count"
    ] == 0

    assert np.allclose(
        result[
            "raw_direction"
        ],
        np.zeros(6)
    )

    assert np.allclose(
        result[
            "direction"
        ],
        np.zeros(6)
    )


# =====================================================
# COMPLETE MULTI-OBSTACLE STATE
# =====================================================

def test_complete_multi_obstacle_state():

    result = (
        multi_obstacle_avoidance_state(
            q=Q_TEST,
            d6=D6_TEST,
            obstacles=OBSTACLES,
            link_radius=(
                LINK_RADIUS
            ),
            safety_distance=0.10,
            influence_distance=0.40
        )
    )

    assert result[
        "critical_obstacle_number"
    ] == 1

    assert result[
        "critical_obstacle_name"
    ] == "Near Link 3"

    assert result[
        "critical_link_number"
    ] == 3

    assert np.isclose(
        result[
            "minimum_clearance"
        ],
        0.15,
        atol=1e-6
    )

    assert result[
        "active_obstacle_count"
    ] == 2

    assert result[
        "combined_direction"
    ].shape == (6,)

    assert result[
        "collision"
    ] is False


# =====================================================
# EMPTY OBSTACLE LIST
# =====================================================

def test_empty_obstacle_list_rejected():

    with pytest.raises(
        ValueError
    ):

        normalize_obstacles([])


# =====================================================
# INVALID CENTER
# =====================================================

def test_invalid_obstacle_center_rejected():

    with pytest.raises(
        ValueError
    ):

        normalize_obstacles([
            {
                "center": [
                    1.0,
                    2.0
                ],
                "radius": 0.1,
            }
        ])


# =====================================================
# NEGATIVE RADIUS
# =====================================================

def test_negative_radius_rejected():

    with pytest.raises(
        ValueError
    ):

        normalize_obstacles([
            {
                "center": [
                    1.0,
                    2.0,
                    3.0
                ],
                "radius": -0.1,
            }
        ])
