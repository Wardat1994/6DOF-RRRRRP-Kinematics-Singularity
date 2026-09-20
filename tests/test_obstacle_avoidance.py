import numpy as np

from src.obstacle_avoidance import (
    point_to_sphere_clearance,
    closest_point_on_segment,
    link_to_sphere_clearance,
    robot_to_sphere_clearances,
    minimum_obstacle_clearance,
    robot_in_collision_with_sphere,
    obstacle_activation,
    obstacle_clearance_gradient,
    obstacle_avoidance_direction,
    obstacle_avoidance_state,
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


# =====================================================
# POINT TO SPHERE CLEARANCE
# =====================================================

def test_point_to_sphere_clearance():

    point = np.array([
        2.0,
        0.0,
        0.0
    ])

    center = np.array([
        0.0,
        0.0,
        0.0
    ])

    clearance = (
        point_to_sphere_clearance(
            point=point,
            obstacle_center=center,
            obstacle_radius=0.5
        )
    )

    assert np.isclose(
        clearance,
        1.5
    )


def test_point_to_sphere_clearance_with_robot_radius():

    point = np.array([
        2.0,
        0.0,
        0.0
    ])

    center = np.array([
        0.0,
        0.0,
        0.0
    ])

    clearance = (
        point_to_sphere_clearance(
            point=point,
            obstacle_center=center,
            obstacle_radius=0.5,
            point_radius=0.2
        )
    )

    assert np.isclose(
        clearance,
        1.3
    )


# =====================================================
# CLOSEST POINT ON SEGMENT
# =====================================================

def test_closest_point_on_segment():

    point = np.array([
        1.0,
        1.0,
        0.0
    ])

    segment_start = np.array([
        0.0,
        0.0,
        0.0
    ])

    segment_end = np.array([
        2.0,
        0.0,
        0.0
    ])

    closest_point, t = (
        closest_point_on_segment(
            point,
            segment_start,
            segment_end
        )
    )

    assert np.allclose(
        closest_point,
        np.array([
            1.0,
            0.0,
            0.0
        ])
    )

    assert np.isclose(
        t,
        0.5
    )


# =====================================================
# ZERO-LENGTH SEGMENT
# =====================================================

def test_zero_length_segment():

    point = np.array([
        5.0,
        5.0,
        5.0
    ])

    segment_point = np.array([
        1.0,
        2.0,
        3.0
    ])

    closest_point, t = (
        closest_point_on_segment(
            point,
            segment_point,
            segment_point
        )
    )

    assert np.allclose(
        closest_point,
        segment_point
    )

    assert np.isclose(
        t,
        0.0
    )


# =====================================================
# LINK TO SPHERE CLEARANCE
# =====================================================

def test_link_to_sphere_clearance():

    result = (
        link_to_sphere_clearance(
            link_start=np.array([
                0.0,
                0.0,
                0.0
            ]),
            link_end=np.array([
                2.0,
                0.0,
                0.0
            ]),
            obstacle_center=np.array([
                1.0,
                1.0,
                0.0
            ]),
            obstacle_radius=0.25,
            link_radius=0.10
        )
    )

    assert np.isclose(
        result[
            "center_distance"
        ],
        1.0
    )

    assert np.isclose(
        result[
            "clearance"
        ],
        0.65
    )

    assert np.allclose(
        result[
            "closest_point"
        ],
        np.array([
            1.0,
            0.0,
            0.0
        ])
    )

    assert np.isclose(
        result[
            "segment_parameter"
        ],
        0.5
    )


# =====================================================
# COMPLETE ROBOT CLEARANCES
# =====================================================

def test_robot_to_sphere_clearances():

    obstacle_center = np.array([
        10.0,
        10.0,
        10.0
    ])

    results = (
        robot_to_sphere_clearances(
            q=Q_TEST,
            d6=D6_TEST,
            obstacle_center=(
                obstacle_center
            ),
            obstacle_radius=0.20,
            link_radius=0.05
        )
    )

    # RRRRRP robot has six link segments
    assert len(results) == 6

    for index, result in enumerate(
        results
    ):

        assert result[
            "link_index"
        ] == index

        assert result[
            "link_number"
        ] == index + 1

        assert np.isfinite(
            result[
                "clearance"
            ]
        )

        assert np.isfinite(
            result[
                "center_distance"
            ]
        )

        assert result[
            "closest_point"
        ].shape == (3,)

        assert (
            0.0
            <= result[
                "segment_parameter"
            ]
            <= 1.0
        )


# =====================================================
# MINIMUM CLEARANCE
# =====================================================

def test_minimum_obstacle_clearance():

    obstacle_center = np.array([
        1.5,
        0.5,
        0.8
    ])

    all_results = (
        robot_to_sphere_clearances(
            q=Q_TEST,
            d6=D6_TEST,
            obstacle_center=(
                obstacle_center
            ),
            obstacle_radius=0.20
        )
    )

    minimum_result = (
        minimum_obstacle_clearance(
            q=Q_TEST,
            d6=D6_TEST,
            obstacle_center=(
                obstacle_center
            ),
            obstacle_radius=0.20
        )
    )

    expected_minimum = min(
        result["clearance"]
        for result in all_results
    )

    assert np.isclose(
        minimum_result[
            "clearance"
        ],
        expected_minimum
    )

    assert (
        1
        <= minimum_result[
            "link_number"
        ]
        <= 6
    )


# =====================================================
# COLLISION DETECTION
# =====================================================

def test_robot_collision_detection():

    # Obstacle centered at the robot base.
    # The first robot link begins at this point,
    # therefore collision must be detected.
    obstacle_center = np.array([
        0.0,
        0.0,
        0.0
    ])

    collision = (
        robot_in_collision_with_sphere(
            q=Q_TEST,
            d6=D6_TEST,
            obstacle_center=(
                obstacle_center
            ),
            obstacle_radius=0.10
        )
    )

    assert collision is True


def test_robot_no_collision_when_obstacle_far_away():

    obstacle_center = np.array([
        20.0,
        20.0,
        20.0
    ])

    collision = (
        robot_in_collision_with_sphere(
            q=Q_TEST,
            d6=D6_TEST,
            obstacle_center=(
                obstacle_center
            ),
            obstacle_radius=0.10
        )
    )

    assert collision is False


# =====================================================
# OBSTACLE ACTIVATION
# =====================================================

def test_obstacle_activation():

    safety_distance = 0.10
    influence_distance = 0.40

    # Far from obstacle
    assert np.isclose(
        obstacle_activation(
            clearance=0.50,
            safety_distance=(
                safety_distance
            ),
            influence_distance=(
                influence_distance
            )
        ),
        0.0
    )

    # Exactly at influence boundary
    assert np.isclose(
        obstacle_activation(
            clearance=0.40,
            safety_distance=(
                safety_distance
            ),
            influence_distance=(
                influence_distance
            )
        ),
        0.0
    )

    # Halfway between safety and influence
    assert np.isclose(
        obstacle_activation(
            clearance=0.25,
            safety_distance=(
                safety_distance
            ),
            influence_distance=(
                influence_distance
            )
        ),
        0.5
    )

    # At safety boundary
    assert np.isclose(
        obstacle_activation(
            clearance=0.10,
            safety_distance=(
                safety_distance
            ),
            influence_distance=(
                influence_distance
            )
        ),
        1.0
    )

    # Inside safety region
    assert np.isclose(
        obstacle_activation(
            clearance=0.0,
            safety_distance=(
                safety_distance
            ),
            influence_distance=(
                influence_distance
            )
        ),
        1.0
    )


# =====================================================
# CLEARANCE GRADIENT
# =====================================================

def test_obstacle_clearance_gradient():

    obstacle_center = np.array([
        1.5,
        0.5,
        0.8
    ])

    gradient = (
        obstacle_clearance_gradient(
            q=Q_TEST,
            d6=D6_TEST,
            obstacle_center=(
                obstacle_center
            ),
            obstacle_radius=0.15,
            link_radius=0.03
        )
    )

    assert gradient.shape == (6,)

    assert np.all(
        np.isfinite(
            gradient
        )
    )


# =====================================================
# OBSTACLE AVOIDANCE DIRECTION
# =====================================================

def test_obstacle_avoidance_direction():

    obstacle_center = np.array([
        1.5,
        0.5,
        0.8
    ])

    direction = (
        obstacle_avoidance_direction(
            q=Q_TEST,
            d6=D6_TEST,
            obstacle_center=(
                obstacle_center
            ),
            obstacle_radius=0.15,
            link_radius=0.03
        )
    )

    assert direction.shape == (6,)

    assert np.all(
        np.isfinite(
            direction
        )
    )

    norm = np.linalg.norm(
        direction
    )

    # Direction is either normalized or zero
    assert (
        np.isclose(
            norm,
            1.0
        )
        or
        np.isclose(
            norm,
            0.0
        )
    )


# =====================================================
# COMPLETE OBSTACLE STATE
# =====================================================

def test_obstacle_avoidance_state():

    obstacle_center = np.array([
        1.5,
        0.5,
        0.8
    ])

    result = (
        obstacle_avoidance_state(
            q=Q_TEST,
            d6=D6_TEST,
            obstacle_center=(
                obstacle_center
            ),
            obstacle_radius=0.15,
            link_radius=0.03,
            safety_distance=0.10,
            influence_distance=0.40
        )
    )

    assert np.isfinite(
        result[
            "clearance"
        ]
    )

    assert (
        1
        <= result[
            "link_number"
        ]
        <= 6
    )

    assert result[
        "closest_point"
    ].shape == (3,)

    assert (
        0.0
        <= result[
            "activation"
        ]
        <= 1.0
    )

    assert result[
        "gradient"
    ].shape == (6,)

    assert result[
        "direction"
    ].shape == (6,)

    assert np.all(
        np.isfinite(
            result[
                "gradient"
            ]
        )
    )

    assert np.all(
        np.isfinite(
            result[
                "direction"
            ]
        )
    )

    assert isinstance(
        result[
            "collision"
        ],
        bool
    )


# =====================================================
# INPUT VALIDATION
# =====================================================

def test_negative_obstacle_radius_rejected():

    try:

        point_to_sphere_clearance(
            point=np.zeros(3),
            obstacle_center=np.ones(3),
            obstacle_radius=-0.10
        )

        assert False

    except ValueError:

        assert True


def test_invalid_activation_distances_rejected():

    try:

        obstacle_activation(
            clearance=0.20,
            safety_distance=0.40,
            influence_distance=0.20
        )

        assert False

    except ValueError:

        assert True
