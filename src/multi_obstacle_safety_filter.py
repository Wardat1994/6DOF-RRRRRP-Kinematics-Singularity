import numpy as np

from src.obstacle_avoidance import (
    minimum_obstacle_clearance,
    obstacle_clearance_gradient,
)

from src.multi_obstacle_avoidance import (
    normalize_obstacles,
    minimum_multi_obstacle_clearance,
)


# =====================================================
# MINIMUM MULTI-OBSTACLE CLEARANCE
# =====================================================

def state_multi_obstacle_clearance(
    q,
    d6,
    obstacles,
    link_radius=0.05
):
    """
    Return the globally minimum robot-obstacle
    clearance across all spherical obstacles.
    """

    result = (
        minimum_multi_obstacle_clearance(
            q=q,
            d6=d6,
            obstacles=obstacles,
            link_radius=link_radius
        )
    )

    return {
        "clearance": float(
            result["clearance"]
        ),

        "obstacle_index": (
            result[
                "obstacle_index"
            ]
        ),

        "obstacle_number": (
            result[
                "obstacle_number"
            ]
        ),

        "obstacle_name": (
            result[
                "obstacle_name"
            ]
        ),

        "link_index": (
            result[
                "link_index"
            ]
        ),

        "link_number": (
            result[
                "link_number"
            ]
        ),
    }


# =====================================================
# LINEARIZED MULTI-OBSTACLE SAFETY FILTER
# =====================================================

def linearized_multi_obstacle_safety_filter(
    delta,
    q,
    d6,
    obstacles,
    link_radius=0.05,
    safety_distance=0.10,
    safety_buffer=0.005,
    projection_passes=5
):
    """
    Apply local linearized clearance constraints for
    every obstacle.

    Each obstacle approximately enforces:

        clearance
        +
        gradient.T @ delta
        >=
        safety_distance + safety_buffer

    Sequential projection is repeated for several
    passes because correcting one obstacle can affect
    another obstacle constraint.
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

    if projection_passes < 1:
        raise ValueError(
            "projection_passes must be at least 1."
        )

    obstacles = (
        normalize_obstacles(
            obstacles
        )
    )

    filtered_delta = (
        delta.copy()
    )

    target_clearance = (
        safety_distance
        + safety_buffer
    )

    corrected = False

    constraint_states = []

    # =================================================
    # BUILD LOCAL CONSTRAINTS
    # =================================================

    for obstacle in obstacles:

        result = (
            minimum_obstacle_clearance(
                q=q,
                d6=d6,

                obstacle_center=(
                    obstacle[
                        "center"
                    ]
                ),

                obstacle_radius=(
                    obstacle[
                        "radius"
                    ]
                ),

                link_radius=(
                    link_radius
                )
            )
        )

        clearance = float(
            result[
                "clearance"
            ]
        )

        gradient = (
            obstacle_clearance_gradient(
                q=q,
                d6=d6,

                obstacle_center=(
                    obstacle[
                        "center"
                    ]
                ),

                obstacle_radius=(
                    obstacle[
                        "radius"
                    ]
                ),

                link_radius=(
                    link_radius
                )
            )
        )

        constraint_states.append({
            "obstacle_index": (
                obstacle["index"]
            ),

            "obstacle_number": (
                obstacle["number"]
            ),

            "obstacle_name": (
                obstacle["name"]
            ),

            "clearance": (
                clearance
            ),

            "gradient": (
                gradient
            ),
        })

    # =================================================
    # SEQUENTIAL CONSTRAINT PROJECTION
    # =================================================

    for _ in range(
        projection_passes
    ):

        pass_corrected = False

        for state in constraint_states:

            gradient = (
                state[
                    "gradient"
                ]
            )

            gradient_norm_squared = float(
                gradient
                @ gradient
            )

            if (
                gradient_norm_squared
                <= 1e-16
            ):
                continue

            predicted_clearance = (
                state[
                    "clearance"
                ]
                + float(
                    gradient
                    @ filtered_delta
                )
            )

            if (
                predicted_clearance
                >= target_clearance
            ):
                continue

            required_change = (
                target_clearance
                - predicted_clearance
            )

            correction = (
                required_change
                / gradient_norm_squared
            ) * gradient

            filtered_delta += (
                correction
            )

            corrected = True
            pass_corrected = True

        if not pass_corrected:
            break

    # =================================================
    # FINAL LOCAL PREDICTIONS
    # =================================================

    predictions = []

    for state in constraint_states:

        predicted_clearance = (
            state[
                "clearance"
            ]
            + float(
                state[
                    "gradient"
                ]
                @ filtered_delta
            )
        )

        predictions.append({
            "obstacle_index": (
                state[
                    "obstacle_index"
                ]
            ),

            "obstacle_number": (
                state[
                    "obstacle_number"
                ]
            ),

            "obstacle_name": (
                state[
                    "obstacle_name"
                ]
            ),

            "current_clearance": (
                state[
                    "clearance"
                ]
            ),

            "predicted_clearance": (
                float(
                    predicted_clearance
                )
            ),
        })

    minimum_prediction = min(
        item[
            "predicted_clearance"
        ]
        for item in predictions
    )

    return {
        "delta": filtered_delta,

        "corrected": bool(
            corrected
        ),

        "constraints": (
            constraint_states
        ),

        "predictions": (
            predictions
        ),

        "minimum_predicted_clearance": (
            float(
                minimum_prediction
            )
        ),
    }


# =====================================================
# CLEARANCE ALONG A MOTION STEP
# =====================================================

def minimum_multi_obstacle_clearance_along_step(
    q,
    d6,
    delta,
    obstacles,
    link_radius=0.05,
    samples=20
):
    """
    Sample a generalized-coordinate motion and return
    the smallest clearance encountered across:

        all interpolation samples
        ×
        all obstacles
        ×
        all robot links
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

    obstacles = (
        normalize_obstacles(
            obstacles
        )
    )

    minimum_clearance = np.inf

    minimum_fraction = 0.0

    critical_obstacle_number = None
    critical_obstacle_name = None
    critical_link_number = None

    for index in range(
        samples + 1
    ):

        fraction = (
            index / samples
        )

        q_sample = (
            np.asarray(
                q,
                dtype=float
            )
            + fraction
            * delta[:5]
        )

        d6_sample = (
            float(d6)
            + fraction
            * delta[5]
        )

        result = (
            minimum_multi_obstacle_clearance(
                q=q_sample,
                d6=d6_sample,
                obstacles=obstacles,
                link_radius=link_radius
            )
        )

        clearance = float(
            result[
                "clearance"
            ]
        )

        if clearance < minimum_clearance:

            minimum_clearance = (
                clearance
            )

            minimum_fraction = (
                fraction
            )

            critical_obstacle_number = (
                result[
                    "obstacle_number"
                ]
            )

            critical_obstacle_name = (
                result[
                    "obstacle_name"
                ]
            )

            critical_link_number = (
                result[
                    "link_number"
                ]
            )

    return {
        "minimum_clearance": float(
            minimum_clearance
        ),

        "minimum_fraction": float(
            minimum_fraction
        ),

        "critical_obstacle_number": (
            critical_obstacle_number
        ),

        "critical_obstacle_name": (
            critical_obstacle_name
        ),

        "critical_link_number": (
            critical_link_number
        ),
    }


# =====================================================
# NONLINEAR MULTI-OBSTACLE SAFE STEP
# =====================================================

def enforce_multi_obstacle_safe_step(
    q,
    d6,
    delta,
    obstacles,
    link_radius=0.05,
    safety_distance=0.10,
    safety_buffer=0.005,
    samples=20,
    projection_passes=5,
    backtracking_factor=0.5,
    max_backtracking_steps=12
):
    """
    Apply:

        1. Multi-obstacle linearized projection.
        2. Nonlinear sampled clearance validation.
        3. Backtracking if any obstacle becomes unsafe.

    A motion is accepted only when every sampled
    robot configuration maintains the requested
    minimum clearance from every obstacle.
    """

    if not (
        0.0
        < backtracking_factor
        < 1.0
    ):
        raise ValueError(
            "backtracking_factor must be in (0, 1)."
        )

    if max_backtracking_steps < 0:
        raise ValueError(
            "max_backtracking_steps must be non-negative."
        )

    obstacles = (
        normalize_obstacles(
            obstacles
        )
    )

    current_state = (
        state_multi_obstacle_clearance(
            q=q,
            d6=d6,
            obstacles=obstacles,
            link_radius=link_radius
        )
    )

    # A safety-preserving filter assumes that the
    # current configuration already satisfies the
    # requested safety distance.
    if (
        current_state[
            "clearance"
        ]
        < safety_distance
    ):

        return {
            "delta": np.zeros(
                6,
                dtype=float
            ),

            "accepted": False,

            "reason": (
                "current_state_unsafe"
            ),

            "scale": 0.0,

            "backtracking_steps": 0,

            "minimum_path_clearance": (
                current_state[
                    "clearance"
                ]
            ),

            "minimum_path_fraction": 0.0,

            "critical_obstacle_number": (
                current_state[
                    "obstacle_number"
                ]
            ),

            "critical_obstacle_name": (
                current_state[
                    "obstacle_name"
                ]
            ),

            "critical_link_number": (
                current_state[
                    "link_number"
                ]
            ),

            "linear_filter_corrected": False,
        }

    linear_result = (
        linearized_multi_obstacle_safety_filter(
            delta=delta,
            q=q,
            d6=d6,
            obstacles=obstacles,
            link_radius=link_radius,
            safety_distance=(
                safety_distance
            ),
            safety_buffer=(
                safety_buffer
            ),
            projection_passes=(
                projection_passes
            )
        )
    )

    filtered_delta = (
        linear_result[
            "delta"
        ].copy()
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
            minimum_multi_obstacle_clearance_along_step(
                q=q,
                d6=d6,
                delta=candidate_delta,
                obstacles=obstacles,
                link_radius=link_radius,
                samples=samples
            )
        )

        if (
            path_result[
                "minimum_clearance"
            ]
            >= safety_distance
        ):

            return {
                "delta": candidate_delta,

                "accepted": True,

                "reason": (
                    "safe_step"
                ),

                "scale": float(
                    scale
                ),

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

                "critical_obstacle_number": (
                    path_result[
                        "critical_obstacle_number"
                    ]
                ),

                "critical_obstacle_name": (
                    path_result[
                        "critical_obstacle_name"
                    ]
                ),

                "critical_link_number": (
                    path_result[
                        "critical_link_number"
                    ]
                ),

                "linear_filter_corrected": bool(
                    linear_result[
                        "corrected"
                    ]
                ),

                "minimum_predicted_clearance": (
                    linear_result[
                        "minimum_predicted_clearance"
                    ]
                ),
            }

        scale *= (
            backtracking_factor
        )

    # =================================================
    # NO SAFE MOTION FOUND
    # =================================================

    return {
        "delta": np.zeros(
            6,
            dtype=float
        ),

        "accepted": False,

        "reason": (
            "no_safe_step_found"
        ),

        "scale": 0.0,

        "backtracking_steps": (
            max_backtracking_steps
        ),

        "minimum_path_clearance": (
            current_state[
                "clearance"
            ]
        ),

        "minimum_path_fraction": 0.0,

        "critical_obstacle_number": (
            current_state[
                "obstacle_number"
            ]
        ),

        "critical_obstacle_name": (
            current_state[
                "obstacle_name"
            ]
        ),

        "critical_link_number": (
            current_state[
                "link_number"
            ]
        ),

        "linear_filter_corrected": bool(
            linear_result[
                "corrected"
            ]
        ),

        "minimum_predicted_clearance": (
            linear_result[
                "minimum_predicted_clearance"
            ]
        ),
    }
