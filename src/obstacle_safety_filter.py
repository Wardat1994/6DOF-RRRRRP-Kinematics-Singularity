import numpy as np

from src.obstacle_avoidance import (
    minimum_obstacle_clearance,
    obstacle_clearance_gradient,
)


# =====================================================
# GENERALIZED STATE
# =====================================================

def generalized_state(q, d6):
    """
    Combine the five revolute coordinates and the
    prismatic coordinate into one 6D state.
    """

    q = np.asarray(
        q,
        dtype=float
    )

    if q.shape != (5,):
        raise ValueError(
            "q must contain five revolute joints."
        )

    return np.concatenate([
        q,
        [float(d6)]
    ])


# =====================================================
# STATE CLEARANCE
# =====================================================

def state_obstacle_clearance(
    q,
    d6,
    obstacle_center,
    obstacle_radius,
    link_radius=0.05
):
    """
    Return minimum robot-obstacle surface clearance.
    """

    result = (
        minimum_obstacle_clearance(
            q=q,
            d6=d6,
            obstacle_center=obstacle_center,
            obstacle_radius=obstacle_radius,
            link_radius=link_radius
        )
    )

    return float(
        result["clearance"]
    )


# =====================================================
# LINEARIZED CLEARANCE SAFETY FILTER
# =====================================================

def linearized_clearance_safety_filter(
    delta,
    q,
    d6,
    obstacle_center,
    obstacle_radius,
    link_radius=0.05,
    safety_distance=0.10,
    safety_buffer=0.005
):
    """
    Project a nominal generalized-coordinate update
    onto a local clearance safety half-space.

    Local model:

        d(q + delta)
        ≈
        d(q) + grad(d)^T delta

    The filtered command attempts to satisfy:

        d(q + delta)
        >=
        safety_distance + safety_buffer

    whenever a correction is required.

    Notes
    -----
    This is a local first-order safety filter.
    The nonlinear geometry is checked separately
    using backtracking before the step is accepted.
    """

    delta = np.asarray(
        delta,
        dtype=float
    ).copy()

    if delta.shape != (6,):
        raise ValueError(
            "delta must have shape (6,)."
        )

    if safety_distance < 0.0:
        raise ValueError(
            "safety_distance must be non-negative."
        )

    if safety_buffer < 0.0:
        raise ValueError(
            "safety_buffer must be non-negative."
        )

    current_clearance = (
        state_obstacle_clearance(
            q=q,
            d6=d6,
            obstacle_center=obstacle_center,
            obstacle_radius=obstacle_radius,
            link_radius=link_radius
        )
    )

    gradient = (
        obstacle_clearance_gradient(
            q=q,
            d6=d6,
            obstacle_center=obstacle_center,
            obstacle_radius=obstacle_radius,
            link_radius=link_radius
        )
    )

    gradient_norm_squared = float(
        gradient @ gradient
    )

    # If the gradient is numerically zero,
    # no local correction direction is available.
    if gradient_norm_squared <= 1e-16:

        return {
            "delta": delta,
            "clearance": current_clearance,
            "gradient": gradient,
            "corrected": False,
            "predicted_clearance": (
                current_clearance
            ),
        }

    target_clearance = (
        safety_distance
        + safety_buffer
    )

    predicted_change = float(
        gradient @ delta
    )

    predicted_clearance = (
        current_clearance
        + predicted_change
    )

    # The nominal step is already locally safe.
    if predicted_clearance >= target_clearance:

        return {
            "delta": delta,
            "clearance": current_clearance,
            "gradient": gradient,
            "corrected": False,
            "predicted_clearance": (
                predicted_clearance
            ),
        }

    # Minimum correction along the clearance gradient.
    required_change = (
        target_clearance
        - predicted_clearance
    )

    correction = (
        required_change
        / gradient_norm_squared
    ) * gradient

    filtered_delta = (
        delta
        + correction
    )

    filtered_prediction = (
        current_clearance
        + float(
            gradient
            @ filtered_delta
        )
    )

    return {
        "delta": filtered_delta,
        "clearance": current_clearance,
        "gradient": gradient,
        "corrected": True,
        "predicted_clearance": (
            filtered_prediction
        ),
    }


# =====================================================
# INTERPOLATED STEP CLEARANCE
# =====================================================

def minimum_clearance_along_step(
    q,
    d6,
    delta,
    obstacle_center,
    obstacle_radius,
    link_radius=0.05,
    samples=10
):
    """
    Sample the generalized motion between the current
    state and the proposed next state.

    This detects a possible collision between two
    discrete IK iterations.
    """

    delta = np.asarray(
        delta,
        dtype=float
    )

    if delta.shape != (6,):
        raise ValueError(
            "delta must have shape (6,)."
        )

    if samples < 1:
        raise ValueError(
            "samples must be at least 1."
        )

    minimum_clearance = np.inf

    minimum_fraction = 0.0

    for index in range(
        samples + 1
    ):

        fraction = (
            index
            / samples
        )

        q_sample = (
            q
            + fraction
            * delta[:5]
        )

        d6_sample = (
            float(d6)
            + fraction
            * delta[5]
        )

        clearance = (
            state_obstacle_clearance(
                q=q_sample,
                d6=d6_sample,
                obstacle_center=(
                    obstacle_center
                ),
                obstacle_radius=(
                    obstacle_radius
                ),
                link_radius=(
                    link_radius
                )
            )
        )

        if clearance < minimum_clearance:

            minimum_clearance = (
                clearance
            )

            minimum_fraction = (
                fraction
            )

    return {
        "minimum_clearance": float(
            minimum_clearance
        ),
        "minimum_fraction": float(
            minimum_fraction
        ),
    }


# =====================================================
# NONLINEAR SAFE STEP
# =====================================================

def enforce_safe_step(
    q,
    d6,
    delta,
    obstacle_center,
    obstacle_radius,
    link_radius=0.05,
    safety_distance=0.10,
    safety_buffer=0.005,
    samples=10,
    backtracking_factor=0.5,
    max_backtracking_steps=12
):
    """
    Apply:

        1. Linearized clearance correction.
        2. Nonlinear sampled clearance validation.
        3. Backtracking if the proposed motion is unsafe.

    The function returns the accepted generalized
    coordinate update.
    """

    if not 0.0 < backtracking_factor < 1.0:
        raise ValueError(
            "backtracking_factor must be in (0, 1)."
        )

    if max_backtracking_steps < 0:
        raise ValueError(
            "max_backtracking_steps must be non-negative."
        )

    linear_result = (
        linearized_clearance_safety_filter(
            delta=delta,
            q=q,
            d6=d6,
            obstacle_center=(
                obstacle_center
            ),
            obstacle_radius=(
                obstacle_radius
            ),
            link_radius=(
                link_radius
            ),
            safety_distance=(
                safety_distance
            ),
            safety_buffer=(
                safety_buffer
            )
        )
    )

    filtered_delta = (
        linear_result[
            "delta"
        ].copy()
    )

    required_clearance = (
        safety_distance
    )

    scale = 1.0

    for backtracking_step in range(
        max_backtracking_steps + 1
    ):

        candidate_delta = (
            scale
            * filtered_delta
        )

        path_result = (
            minimum_clearance_along_step(
                q=q,
                d6=d6,
                delta=candidate_delta,
                obstacle_center=(
                    obstacle_center
                ),
                obstacle_radius=(
                    obstacle_radius
                ),
                link_radius=(
                    link_radius
                ),
                samples=samples
            )
        )

        if (
            path_result[
                "minimum_clearance"
            ]
            >= required_clearance
        ):

            return {
                "delta": candidate_delta,
                "accepted": True,
                "scale": float(scale),
                "backtracking_steps": (
                    backtracking_step
                ),
                "minimum_path_clearance": (
                    path_result[
                        "minimum_clearance"
                    ]
                ),
                "minimum_path_fraction": (
                    path_result[
                        "minimum_fraction"
                    ]
                ),
                "linear_filter_corrected": (
                    linear_result[
                        "corrected"
                    ]
                ),
                "predicted_clearance": (
                    linear_result[
                        "predicted_clearance"
                    ]
                ),
            }

        scale *= (
            backtracking_factor
        )

    # No safe forward step was found.
    return {
        "delta": np.zeros(
            6,
            dtype=float
        ),
        "accepted": False,
        "scale": 0.0,
        "backtracking_steps": (
            max_backtracking_steps
        ),
        "minimum_path_clearance": (
            state_obstacle_clearance(
                q=q,
                d6=d6,
                obstacle_center=(
                    obstacle_center
                ),
                obstacle_radius=(
                    obstacle_radius
                ),
                link_radius=(
                    link_radius
                )
            )
        ),
        "minimum_path_fraction": 0.0,
        "linear_filter_corrected": (
            linear_result[
                "corrected"
            ]
        ),
        "predicted_clearance": (
            linear_result[
                "predicted_clearance"
            ]
        ),
    }
