import numpy as np

from src.obstacle_avoidance import (
    minimum_obstacle_clearance,
    obstacle_activation,
    obstacle_clearance_gradient,
)


# =====================================================
# OBSTACLE VALIDATION
# =====================================================

def normalize_obstacles(obstacles):
    """
    Validate and normalize a collection of spherical
    obstacles.

    Each obstacle must be a dictionary containing:

        {
            "center": [x, y, z],
            "radius": radius
        }

    An optional name may also be supplied:

        {
            "name": "Obstacle A",
            "center": [x, y, z],
            "radius": radius
        }

    Returns
    -------
    list of dict
        Validated obstacle definitions.
    """

    if not isinstance(
        obstacles,
        (list, tuple)
    ):
        raise ValueError(
            "obstacles must be a list or tuple."
        )

    if len(obstacles) == 0:
        raise ValueError(
            "at least one obstacle is required."
        )

    normalized = []

    for index, obstacle in enumerate(
        obstacles
    ):

        if not isinstance(
            obstacle,
            dict
        ):
            raise ValueError(
                "each obstacle must be a dictionary."
            )

        if "center" not in obstacle:
            raise ValueError(
                "each obstacle must define 'center'."
            )

        if "radius" not in obstacle:
            raise ValueError(
                "each obstacle must define 'radius'."
            )

        center = np.asarray(
            obstacle["center"],
            dtype=float
        )

        if center.shape != (3,):
            raise ValueError(
                "each obstacle center must have "
                "shape (3,)."
            )

        if not np.all(
            np.isfinite(center)
        ):
            raise ValueError(
                "obstacle centers must contain "
                "finite values."
            )

        radius = float(
            obstacle["radius"]
        )

        if not np.isfinite(
            radius
        ):
            raise ValueError(
                "obstacle radii must be finite."
            )

        if radius < 0.0:
            raise ValueError(
                "obstacle radii must be "
                "non-negative."
            )

        name = obstacle.get(
            "name",
            f"Obstacle {index + 1}"
        )

        name = str(
            name
        )

        normalized.append({
            "index": index,
            "number": index + 1,
            "name": name,
            "center": center.copy(),
            "radius": radius,
        })

    return normalized


# =====================================================
# CLEARANCE TO EVERY OBSTACLE
# =====================================================

def multi_obstacle_clearances(
    q,
    d6,
    obstacles,
    link_radius=0.0
):
    """
    Calculate the minimum robot clearance to every
    spherical obstacle independently.

    Returns
    -------
    list of dict
        One minimum-clearance result for each obstacle.
    """

    obstacles = normalize_obstacles(
        obstacles
    )

    results = []

    for obstacle in obstacles:

        clearance_result = (
            minimum_obstacle_clearance(
                q=q,
                d6=d6,

                obstacle_center=(
                    obstacle["center"]
                ),

                obstacle_radius=(
                    obstacle["radius"]
                ),

                link_radius=(
                    link_radius
                )
            )
        )

        result = {
            key: (
                value.copy()
                if isinstance(
                    value,
                    np.ndarray
                )
                else value
            )
            for key, value
            in clearance_result.items()
        }

        result[
            "obstacle_index"
        ] = obstacle["index"]

        result[
            "obstacle_number"
        ] = obstacle["number"]

        result[
            "obstacle_name"
        ] = obstacle["name"]

        result[
            "obstacle_center"
        ] = obstacle[
            "center"
        ].copy()

        result[
            "obstacle_radius"
        ] = obstacle["radius"]

        results.append(
            result
        )

    return results


# =====================================================
# GLOBAL MINIMUM CLEARANCE
# =====================================================

def minimum_multi_obstacle_clearance(
    q,
    d6,
    obstacles,
    link_radius=0.0
):
    """
    Return the globally smallest clearance across
    all robot links and all obstacles.

    The returned obstacle is the currently most
    critical obstacle.
    """

    results = (
        multi_obstacle_clearances(
            q=q,
            d6=d6,
            obstacles=obstacles,
            link_radius=link_radius
        )
    )

    minimum_result = min(
        results,
        key=lambda item: (
            item["clearance"]
        )
    )

    return {
        key: (
            value.copy()
            if isinstance(
                value,
                np.ndarray
            )
            else value
        )
        for key, value
        in minimum_result.items()
    }


# =====================================================
# MULTI-OBSTACLE COLLISION CHECK
# =====================================================

def robot_in_collision_with_obstacles(
    q,
    d6,
    obstacles,
    link_radius=0.0
):
    """
    Return True if the robot touches or penetrates
    any obstacle.
    """

    result = (
        minimum_multi_obstacle_clearance(
            q=q,
            d6=d6,
            obstacles=obstacles,
            link_radius=link_radius
        )
    )

    return bool(
        result["clearance"]
        <= 0.0
    )


# =====================================================
# PER-OBSTACLE STATES
# =====================================================

def multi_obstacle_states(
    q,
    d6,
    obstacles,
    link_radius=0.0,
    safety_distance=0.10,
    influence_distance=0.40,
    epsilon_revolute=1e-5,
    epsilon_prismatic=1e-5
):
    """
    Calculate avoidance information for every
    obstacle.

    Expensive numerical gradients are calculated
    only for obstacles inside the influence region.
    """

    obstacles = normalize_obstacles(
        obstacles
    )

    clearance_results = (
        multi_obstacle_clearances(
            q=q,
            d6=d6,
            obstacles=obstacles,
            link_radius=link_radius
        )
    )

    states = []

    for obstacle, clearance_result in zip(
        obstacles,
        clearance_results
    ):

        clearance = float(
            clearance_result[
                "clearance"
            ]
        )

        activation = (
            obstacle_activation(
                clearance=clearance,
                safety_distance=(
                    safety_distance
                ),
                influence_distance=(
                    influence_distance
                )
            )
        )

        # Calculate gradient only when required.
        if activation > 0.0:

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
                    ),

                    epsilon_revolute=(
                        epsilon_revolute
                    ),

                    epsilon_prismatic=(
                        epsilon_prismatic
                    )
                )
            )

            gradient_norm = float(
                np.linalg.norm(
                    gradient
                )
            )

            if gradient_norm <= 1e-12:

                direction = np.zeros(
                    6,
                    dtype=float
                )

            else:

                direction = (
                    gradient
                    / gradient_norm
                )

        else:

            gradient = np.zeros(
                6,
                dtype=float
            )

            direction = np.zeros(
                6,
                dtype=float
            )

        states.append({
            "obstacle_index": (
                obstacle["index"]
            ),

            "obstacle_number": (
                obstacle["number"]
            ),

            "obstacle_name": (
                obstacle["name"]
            ),

            "obstacle_center": (
                obstacle[
                    "center"
                ].copy()
            ),

            "obstacle_radius": (
                obstacle["radius"]
            ),

            "clearance": (
                clearance
            ),

            "link_index": (
                clearance_result[
                    "link_index"
                ]
            ),

            "link_number": (
                clearance_result[
                    "link_number"
                ]
            ),

            "closest_point": (
                clearance_result[
                    "closest_point"
                ].copy()
            ),

            "center_distance": (
                clearance_result[
                    "center_distance"
                ]
            ),

            "activation": (
                activation
            ),

            "gradient": (
                gradient
            ),

            "direction": (
                direction
            ),

            "collision": bool(
                clearance <= 0.0
            ),
        })

    return states


# =====================================================
# COMBINED MULTI-OBSTACLE AVOIDANCE DIRECTION
# =====================================================

def combined_obstacle_avoidance_direction(
    q,
    d6,
    obstacles,
    link_radius=0.0,
    safety_distance=0.10,
    influence_distance=0.40,
    epsilon_revolute=1e-5,
    epsilon_prismatic=1e-5
):
    """
    Combine avoidance directions from all currently
    active obstacles.

    Each obstacle direction is weighted by its
    activation level.

    Obstacles outside the influence region therefore
    contribute zero.

    Returns
    -------
    dict
        direction
        raw_direction
        states
        active_obstacle_count
    """

    states = (
        multi_obstacle_states(
            q=q,
            d6=d6,
            obstacles=obstacles,
            link_radius=link_radius,
            safety_distance=(
                safety_distance
            ),
            influence_distance=(
                influence_distance
            ),
            epsilon_revolute=(
                epsilon_revolute
            ),
            epsilon_prismatic=(
                epsilon_prismatic
            )
        )
    )

    raw_direction = np.zeros(
        6,
        dtype=float
    )

    active_count = 0

    for state in states:

        activation = float(
            state["activation"]
        )

        if activation <= 0.0:
            continue

        active_count += 1

        raw_direction += (
            activation
            * state["direction"]
        )

    norm = float(
        np.linalg.norm(
            raw_direction
        )
    )

    if norm <= 1e-12:

        direction = np.zeros(
            6,
            dtype=float
        )

    else:

        direction = (
            raw_direction
            / norm
        )

    return {
        "direction": direction,
        "raw_direction": (
            raw_direction
        ),
        "states": states,
        "active_obstacle_count": (
            active_count
        ),
    }


# =====================================================
# COMPLETE MULTI-OBSTACLE STATE
# =====================================================

def multi_obstacle_avoidance_state(
    q,
    d6,
    obstacles,
    link_radius=0.0,
    safety_distance=0.10,
    influence_distance=0.40,
    epsilon_revolute=1e-5,
    epsilon_prismatic=1e-5
):
    """
    Return the complete state of the robot relative
    to a collection of spherical obstacles.

    The function identifies:

    - clearance to every obstacle,
    - the globally critical obstacle,
    - active obstacles,
    - collision state,
    - combined avoidance direction.
    """

    combined = (
        combined_obstacle_avoidance_direction(
            q=q,
            d6=d6,
            obstacles=obstacles,
            link_radius=link_radius,
            safety_distance=(
                safety_distance
            ),
            influence_distance=(
                influence_distance
            ),
            epsilon_revolute=(
                epsilon_revolute
            ),
            epsilon_prismatic=(
                epsilon_prismatic
            )
        )
    )

    states = combined[
        "states"
    ]

    critical_state = min(
        states,
        key=lambda item: (
            item["clearance"]
        )
    )

    collision = any(
        state["collision"]
        for state in states
    )

    minimum_clearance = float(
        critical_state[
            "clearance"
        ]
    )

    return {
        "states": states,

        "minimum_clearance": (
            minimum_clearance
        ),

        "critical_obstacle_index": (
            critical_state[
                "obstacle_index"
            ]
        ),

        "critical_obstacle_number": (
            critical_state[
                "obstacle_number"
            ]
        ),

        "critical_obstacle_name": (
            critical_state[
                "obstacle_name"
            ]
        ),

        "critical_link_index": (
            critical_state[
                "link_index"
            ]
        ),

        "critical_link_number": (
            critical_state[
                "link_number"
            ]
        ),

        "critical_closest_point": (
            critical_state[
                "closest_point"
            ].copy()
        ),

        "critical_activation": (
            critical_state[
                "activation"
            ]
        ),

        "combined_direction": (
            combined[
                "direction"
            ]
        ),

        "raw_combined_direction": (
            combined[
                "raw_direction"
            ]
        ),

        "active_obstacle_count": (
            combined[
                "active_obstacle_count"
            ]
        ),

        "collision": bool(
            collision
        ),
    }
