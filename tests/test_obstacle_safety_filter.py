import numpy as np
import pytest

from src.obstacle_safety_filter import (
    generalized_state,
    state_obstacle_clearance,
    linearized_clearance_safety_filter,
    minimum_clearance_along_step,
    enforce_safe_step,
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
# CONTROLLED OBSTACLE
# =====================================================
#
# For this robot configuration:
#
# centerline distance to Link 3 ≈ 0.35 m
#
# obstacle radius = 0.15 m
# link radius     = 0.05 m
#
# therefore:
#
# clearance ≈ 0.35 - 0.15 - 0.05
#           ≈ 0.15 m
#
# This configuration starts safely above:
#
# safety_distance = 0.10 m
# =====================================================

OBSTACLE_CENTER = np.array([
    0.35,
    -1.34641016,
    0.70
])

OBSTACLE_RADIUS = 0.15

LINK_RADIUS = 0.05

SAFETY_DISTANCE = 0.10

SAFETY_BUFFER = 0.005


# =====================================================
# GENERALIZED STATE
# =====================================================

def test_generalized_state():

    state = generalized_state(
        Q_TEST,
        D6_TEST
    )

    assert state.shape == (6,)

    assert np.allclose(
        state[:5],
        Q_TEST
    )

    assert np.isclose(
        state[5],
        D6_TEST
    )


# =====================================================
# INITIAL CLEARANCE
# =====================================================

def test_initial_clearance_is_expected():

    clearance = (
        state_obstacle_clearance(
            q=Q_TEST,
            d6=D6_TEST,
            obstacle_center=(
                OBSTACLE_CENTER
            ),
            obstacle_radius=(
                OBSTACLE_RADIUS
            ),
            link_radius=(
                LINK_RADIUS
            )
        )
    )

    assert np.isclose(
        clearance,
        0.15,
        atol=1e-6
    )

    assert (
        clearance
        > SAFETY_DISTANCE
    )


# =====================================================
# SAFE COMMAND SHOULD REMAIN UNCHANGED
# =====================================================

def test_linearized_filter_preserves_safe_command():

    delta = np.zeros(
        6,
        dtype=float
    )

    result = (
        linearized_clearance_safety_filter(
            delta=delta,
            q=Q_TEST,
            d6=D6_TEST,
            obstacle_center=(
                OBSTACLE_CENTER
            ),
            obstacle_radius=(
                OBSTACLE_RADIUS
            ),
            link_radius=(
                LINK_RADIUS
            ),
            safety_distance=(
                SAFETY_DISTANCE
            ),
            safety_buffer=(
                SAFETY_BUFFER
            )
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
            "predicted_clearance"
        ]
        >= SAFETY_DISTANCE
    )


# =====================================================
# UNSAFE NOMINAL COMMAND
# =====================================================

def test_linearized_filter_corrects_unsafe_command():

    # Positive theta1 motion moves the robot
    # toward this obstacle in the selected geometry.
    delta = np.zeros(
        6,
        dtype=float
    )

    delta[0] = 0.08

    result = (
        linearized_clearance_safety_filter(
            delta=delta,
            q=Q_TEST,
            d6=D6_TEST,
            obstacle_center=(
                OBSTACLE_CENTER
            ),
            obstacle_radius=(
                OBSTACLE_RADIUS
            ),
            link_radius=(
                LINK_RADIUS
            ),
            safety_distance=(
                SAFETY_DISTANCE
            ),
            safety_buffer=(
                SAFETY_BUFFER
            )
        )
    )

    assert result[
        "corrected"
    ] is True

    assert not np.allclose(
        result["delta"],
        delta
    )

    assert result[
        "gradient"
    ].shape == (6,)

    assert np.all(
        np.isfinite(
            result[
                "gradient"
            ]
        )
    )

    assert (
        result[
            "predicted_clearance"
        ]
        >= (
            SAFETY_DISTANCE
            + SAFETY_BUFFER
            - 1e-10
        )
    )


# =====================================================
# UNSAFE RAW STEP DETECTION
# =====================================================

def test_minimum_clearance_along_step_detects_unsafe_motion():

    delta = np.zeros(
        6,
        dtype=float
    )

    delta[0] = 0.08

    result = (
        minimum_clearance_along_step(
            q=Q_TEST,
            d6=D6_TEST,
            delta=delta,
            obstacle_center=(
                OBSTACLE_CENTER
            ),
            obstacle_radius=(
                OBSTACLE_RADIUS
            ),
            link_radius=(
                LINK_RADIUS
            ),
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

    assert (
        0.0
        <= result[
            "minimum_fraction"
        ]
        <= 1.0
    )


# =====================================================
# SAFETY FILTERED STEP
# =====================================================

def test_enforce_safe_step_produces_safe_motion():

    delta = np.zeros(
        6,
        dtype=float
    )

    delta[0] = 0.08

    result = (
        enforce_safe_step(
            q=Q_TEST,
            d6=D6_TEST,
            delta=delta,
            obstacle_center=(
                OBSTACLE_CENTER
            ),
            obstacle_radius=(
                OBSTACLE_RADIUS
            ),
            link_radius=(
                LINK_RADIUS
            ),
            safety_distance=(
                SAFETY_DISTANCE
            ),
            safety_buffer=(
                SAFETY_BUFFER
            ),
            samples=20,
            backtracking_factor=0.5,
            max_backtracking_steps=12
        )
    )

    assert result[
        "accepted"
    ] is True

    assert result[
        "delta"
    ].shape == (6,)

    assert np.all(
        np.isfinite(
            result[
                "delta"
            ]
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

    assert (
        result[
            "backtracking_steps"
        ]
        >= 0
    )


# =====================================================
# VERIFY ACCEPTED STEP DIRECTLY
# =====================================================

def test_accepted_step_remains_safe_when_rechecked():

    nominal_delta = np.zeros(
        6,
        dtype=float
    )

    nominal_delta[0] = 0.08

    filtered = (
        enforce_safe_step(
            q=Q_TEST,
            d6=D6_TEST,
            delta=nominal_delta,
            obstacle_center=(
                OBSTACLE_CENTER
            ),
            obstacle_radius=(
                OBSTACLE_RADIUS
            ),
            link_radius=(
                LINK_RADIUS
            ),
            safety_distance=(
                SAFETY_DISTANCE
            ),
            safety_buffer=(
                SAFETY_BUFFER
            ),
            samples=20
        )
    )

    check = (
        minimum_clearance_along_step(
            q=Q_TEST,
            d6=D6_TEST,
            delta=filtered[
                "delta"
            ],
            obstacle_center=(
                OBSTACLE_CENTER
            ),
            obstacle_radius=(
                OBSTACLE_RADIUS
            ),
            link_radius=(
                LINK_RADIUS
            ),
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
# INVALID DELTA
# =====================================================

def test_invalid_delta_shape_rejected():

    with pytest.raises(
        ValueError
    ):

        linearized_clearance_safety_filter(
            delta=np.zeros(5),
            q=Q_TEST,
            d6=D6_TEST,
            obstacle_center=(
                OBSTACLE_CENTER
            ),
            obstacle_radius=(
                OBSTACLE_RADIUS
            )
        )


# =====================================================
# INVALID NUMBER OF SAMPLES
# =====================================================

def test_invalid_samples_rejected():

    with pytest.raises(
        ValueError
    ):

        minimum_clearance_along_step(
            q=Q_TEST,
            d6=D6_TEST,
            delta=np.zeros(6),
            obstacle_center=(
                OBSTACLE_CENTER
            ),
            obstacle_radius=(
                OBSTACLE_RADIUS
            ),
            samples=0
        )


# =====================================================
# INVALID BACKTRACKING FACTOR
# =====================================================

def test_invalid_backtracking_factor_rejected():

    with pytest.raises(
        ValueError
    ):

        enforce_safe_step(
            q=Q_TEST,
            d6=D6_TEST,
            delta=np.zeros(6),
            obstacle_center=(
                OBSTACLE_CENTER
            ),
            obstacle_radius=(
                OBSTACLE_RADIUS
            ),
            backtracking_factor=1.0
        )
