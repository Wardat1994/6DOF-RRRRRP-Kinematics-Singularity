import numpy as np
import pytest

from src.multi_obstacle_safety_filter import (
    state_multi_obstacle_clearance,
    linearized_multi_obstacle_safety_filter,
    minimum_multi_obstacle_clearance_along_step,
    enforce_multi_obstacle_safe_step,
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

SAFETY_DISTANCE = 0.10

SAFETY_BUFFER = 0.005


# =====================================================
# MULTIPLE OBSTACLES
# =====================================================

OBSTACLES = [
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
        "name": "Far Obstacle 1",
        "center": [
            20.0,
            20.0,
            20.0
        ],
        "radius": 0.20,
    },
    {
        "name": "Far Obstacle 2",
        "center": [
            -20.0,
            20.0,
            15.0
        ],
        "radius": 0.25,
    },
]


# =====================================================
# GLOBAL CLEARANCE STATE
# =====================================================

def test_state_multi_obstacle_clearance():

    result = (
        state_multi_obstacle_clearance(
            q=Q_TEST,
            d6=D6_TEST,
            obstacles=OBSTACLES,
            link_radius=LINK_RADIUS
        )
    )

    assert np.isclose(
        result["clearance"],
        0.15,
        atol=1e-6
    )

    assert result[
        "obstacle_name"
    ] == "Near Link 3"

    assert result[
        "link_number"
    ] == 3


# =====================================================
# SAFE ZERO STEP
# =====================================================

def test_linearized_filter_preserves_safe_zero_step():

    delta = np.zeros(
        6,
        dtype=float
    )

    result = (
        linearized_multi_obstacle_safety_filter(
            delta=delta,
            q=Q_TEST,
            d6=D6_TEST,
            obstacles=OBSTACLES,
            link_radius=LINK_RADIUS,
            safety_distance=SAFETY_DISTANCE,
            safety_buffer=SAFETY_BUFFER
        )
    )

    assert result[
        "corrected"
    ] is False

    assert np.allclose(
        result["delta"],
        delta
    )

    assert (
        result[
            "minimum_predicted_clearance"
        ]
        >= SAFETY_DISTANCE
    )


# =====================================================
# UNSAFE NOMINAL STEP
# =====================================================

def test_linearized_filter_corrects_unsafe_step():

    delta = np.zeros(
        6,
        dtype=float
    )

    delta[0] = 0.08

    result = (
        linearized_multi_obstacle_safety_filter(
            delta=delta,
            q=Q_TEST,
            d6=D6_TEST,
            obstacles=OBSTACLES,
            link_radius=LINK_RADIUS,
            safety_distance=SAFETY_DISTANCE,
            safety_buffer=SAFETY_BUFFER,
            projection_passes=5
        )
    )

    assert result[
        "corrected"
    ] is True

    assert not np.allclose(
        result["delta"],
        delta
    )

    assert len(
        result["constraints"]
    ) == 3

    assert len(
        result["predictions"]
    ) == 3

    assert (
        result[
            "minimum_predicted_clearance"
        ]
        >= (
            SAFETY_DISTANCE
            + SAFETY_BUFFER
            - 1e-9
        )
    )


# =====================================================
# UNSAFE RAW MOTION DETECTION
# =====================================================

def test_raw_step_detects_unsafe_motion():

    delta = np.zeros(
        6,
        dtype=float
    )

    delta[0] = 0.08

    result = (
        minimum_multi_obstacle_clearance_along_step(
            q=Q_TEST,
            d6=D6_TEST,
            delta=delta,
            obstacles=OBSTACLES,
            link_radius=LINK_RADIUS,
            samples=20
        )
    )

    assert np.isfinite(
        result[
            "minimum_clearance"
        ]
    )

    assert (
        result[
            "minimum_clearance"
        ]
        < SAFETY_DISTANCE
    )

    assert result[
        "critical_obstacle_name"
    ] == "Near Link 3"

    assert (
        0.0
        <= result[
            "minimum_fraction"
        ]
        <= 1.0
    )


# =====================================================
# ENFORCED SAFE STEP
# =====================================================

def test_enforce_multi_obstacle_safe_step():

    delta = np.zeros(
        6,
        dtype=float
    )

    delta[0] = 0.08

    result = (
        enforce_multi_obstacle_safe_step(
            q=Q_TEST,
            d6=D6_TEST,
            delta=delta,
            obstacles=OBSTACLES,
            link_radius=LINK_RADIUS,
            safety_distance=SAFETY_DISTANCE,
            safety_buffer=SAFETY_BUFFER,
            samples=20,
            projection_passes=5,
            backtracking_factor=0.5,
            max_backtracking_steps=12
        )
    )

    assert result[
        "accepted"
    ] is True

    assert result[
        "reason"
    ] == "safe_step"

    assert result[
        "delta"
    ].shape == (6,)

    assert np.all(
        np.isfinite(
            result["delta"]
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

    assert (
        0.0
        < result[
            "scale"
        ]
        <= 1.0
    )


# =====================================================
# DIRECT RECHECK OF ACCEPTED MOTION
# =====================================================

def test_accepted_step_is_safe_when_rechecked():

    nominal_delta = np.zeros(
        6,
        dtype=float
    )

    nominal_delta[0] = 0.08

    filtered = (
        enforce_multi_obstacle_safe_step(
            q=Q_TEST,
            d6=D6_TEST,
            delta=nominal_delta,
            obstacles=OBSTACLES,
            link_radius=LINK_RADIUS,
            safety_distance=SAFETY_DISTANCE,
            safety_buffer=SAFETY_BUFFER,
            samples=20
        )
    )

    check = (
        minimum_multi_obstacle_clearance_along_step(
            q=Q_TEST,
            d6=D6_TEST,
            delta=filtered[
                "delta"
            ],
            obstacles=OBSTACLES,
            link_radius=LINK_RADIUS,
            samples=30
        )
    )

    assert filtered[
        "accepted"
    ] is True

    assert (
        check[
            "minimum_clearance"
        ]
        >= (
            SAFETY_DISTANCE
            - 1e-9
        )
    )


# =====================================================
# CURRENT STATE ALREADY UNSAFE
# =====================================================

def test_current_unsafe_state_is_rejected():

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
            "name": "Far Obstacle",
            "center": [
                20.0,
                20.0,
                20.0
            ],
            "radius": 0.10,
        },
    ]

    result = (
        enforce_multi_obstacle_safe_step(
            q=Q_TEST,
            d6=D6_TEST,
            delta=np.zeros(6),
            obstacles=unsafe_obstacles,
            link_radius=LINK_RADIUS,
            safety_distance=SAFETY_DISTANCE
        )
    )

    assert result[
        "accepted"
    ] is False

    assert result[
        "reason"
    ] == "current_state_unsafe"

    assert result[
        "scale"
    ] == 0.0

    assert np.allclose(
        result["delta"],
        np.zeros(6)
    )

    assert (
        result[
            "minimum_path_clearance"
        ]
        < SAFETY_DISTANCE
    )


# =====================================================
# ALL OBSTACLES FAR AWAY
# =====================================================

def test_far_obstacles_do_not_modify_zero_step():

    far_obstacles = [
        {
            "name": "Far 1",
            "center": [
                20.0,
                20.0,
                20.0
            ],
            "radius": 0.10,
        },
        {
            "name": "Far 2",
            "center": [
                -20.0,
                -20.0,
                20.0
            ],
            "radius": 0.10,
        },
    ]

    delta = np.zeros(
        6,
        dtype=float
    )

    result = (
        linearized_multi_obstacle_safety_filter(
            delta=delta,
            q=Q_TEST,
            d6=D6_TEST,
            obstacles=far_obstacles,
            link_radius=LINK_RADIUS,
            safety_distance=SAFETY_DISTANCE,
            safety_buffer=SAFETY_BUFFER
        )
    )

    assert result[
        "corrected"
    ] is False

    assert np.allclose(
        result["delta"],
        delta
    )


# =====================================================
# INVALID DELTA SHAPE
# =====================================================

def test_invalid_delta_shape_rejected():

    with pytest.raises(
        ValueError
    ):

        linearized_multi_obstacle_safety_filter(
            delta=np.zeros(5),
            q=Q_TEST,
            d6=D6_TEST,
            obstacles=OBSTACLES
        )


# =====================================================
# INVALID NUMBER OF SAMPLES
# =====================================================

def test_invalid_samples_rejected():

    with pytest.raises(
        ValueError
    ):

        minimum_multi_obstacle_clearance_along_step(
            q=Q_TEST,
            d6=D6_TEST,
            delta=np.zeros(6),
            obstacles=OBSTACLES,
            samples=0
        )


# =====================================================
# INVALID PROJECTION PASSES
# =====================================================

def test_invalid_projection_passes_rejected():

    with pytest.raises(
        ValueError
    ):

        linearized_multi_obstacle_safety_filter(
            delta=np.zeros(6),
            q=Q_TEST,
            d6=D6_TEST,
            obstacles=OBSTACLES,
            projection_passes=0
        )


# =====================================================
# INVALID BACKTRACKING FACTOR
# =====================================================

def test_invalid_backtracking_factor_rejected():

    with pytest.raises(
        ValueError
    ):

        enforce_multi_obstacle_safe_step(
            q=Q_TEST,
            d6=D6_TEST,
            delta=np.zeros(6),
            obstacles=OBSTACLES,
            backtracking_factor=1.0
        )
