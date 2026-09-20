import numpy as np

from src.kinematics import get_joint_positions


# =====================================================
# VECTOR VALIDATION
# =====================================================

def _as_vector3(value, name):
    """
    Convert a value to a finite 3D NumPy vector.
    """

    vector = np.asarray(
        value,
        dtype=float
    )

    if vector.shape != (3,):
        raise ValueError(
            f"{name} must have shape (3,)."
        )

    if not np.all(
        np.isfinite(vector)
    ):
        raise ValueError(
            f"{name} must contain finite values."
        )

    return vector


# =====================================================
# LINK RADIUS HANDLING
# =====================================================

def _link_radii_array(
    link_radius,
    number_of_links=6
):
    """
    Convert a scalar or array-like link radius
    into one radius for each robot link.
    """

    radii = np.asarray(
        link_radius,
        dtype=float
    )

    if radii.ndim == 0:
        radius = float(radii)

        if radius < 0.0:
            raise ValueError(
                "link_radius must be non-negative."
            )

        return np.full(
            number_of_links,
            radius,
            dtype=float
        )

    if radii.shape != (
        number_of_links,
    ):
        raise ValueError(
            "link_radius must be a scalar or "
            f"contain {number_of_links} values."
        )

    if np.any(
        radii < 0.0
    ):
        raise ValueError(
            "link radii must be non-negative."
        )

    if not np.all(
        np.isfinite(radii)
    ):
        raise ValueError(
            "link radii must be finite."
        )

    return radii.copy()


# =====================================================
# POINT TO SPHERE CLEARANCE
# =====================================================

def point_to_sphere_clearance(
    point,
    obstacle_center,
    obstacle_radius,
    point_radius=0.0
):
    """
    Calculate surface-to-surface clearance between
    a point/spherical robot element and a spherical
    obstacle.

    Positive:
        separated.

    Zero:
        touching.

    Negative:
        penetration / collision.
    """

    point = _as_vector3(
        point,
        "point"
    )

    obstacle_center = _as_vector3(
        obstacle_center,
        "obstacle_center"
    )

    obstacle_radius = float(
        obstacle_radius
    )

    point_radius = float(
        point_radius
    )

    if obstacle_radius < 0.0:
        raise ValueError(
            "obstacle_radius must be non-negative."
        )

    if point_radius < 0.0:
        raise ValueError(
            "point_radius must be non-negative."
        )

    center_distance = float(
        np.linalg.norm(
            point
            - obstacle_center
        )
    )

    clearance = (
        center_distance
        - obstacle_radius
        - point_radius
    )

    return float(clearance)


# =====================================================
# CLOSEST POINT ON LINE SEGMENT
# =====================================================

def closest_point_on_segment(
    point,
    segment_start,
    segment_end
):
    """
    Return the closest point on a finite line segment
    to a specified Cartesian point.

    Also returns the clamped segment parameter t:

        t = 0 -> segment_start
        t = 1 -> segment_end
    """

    point = _as_vector3(
        point,
        "point"
    )

    segment_start = _as_vector3(
        segment_start,
        "segment_start"
    )

    segment_end = _as_vector3(
        segment_end,
        "segment_end"
    )

    segment_vector = (
        segment_end
        - segment_start
    )

    length_squared = float(
        segment_vector
        @ segment_vector
    )

    # Some robot frames may have coincident origins.
    # Treat a zero-length segment as a point.
    if length_squared <= 1e-16:
        return (
            segment_start.copy(),
            0.0
        )

    t = float(
        (
            point
            - segment_start
        )
        @ segment_vector
        / length_squared
    )

    t_clamped = float(
        np.clip(
            t,
            0.0,
            1.0
        )
    )

    closest_point = (
        segment_start
        + t_clamped
        * segment_vector
    )

    return (
        closest_point,
        t_clamped
    )


# =====================================================
# LINK TO SPHERE CLEARANCE
# =====================================================

def link_to_sphere_clearance(
    link_start,
    link_end,
    obstacle_center,
    obstacle_radius,
    link_radius=0.0
):
    """
    Calculate minimum surface clearance between
    a robot link segment and a spherical obstacle.

    The robot link can optionally be modeled with
    a non-zero radius, producing a capsule-like
    collision approximation.

    Returns
    -------
    dict
        clearance
        center_distance
        closest_point
        segment_parameter
    """

    link_start = _as_vector3(
        link_start,
        "link_start"
    )

    link_end = _as_vector3(
        link_end,
        "link_end"
    )

    obstacle_center = _as_vector3(
        obstacle_center,
        "obstacle_center"
    )

    obstacle_radius = float(
        obstacle_radius
    )

    link_radius = float(
        link_radius
    )

    if obstacle_radius < 0.0:
        raise ValueError(
            "obstacle_radius must be non-negative."
        )

    if link_radius < 0.0:
        raise ValueError(
            "link_radius must be non-negative."
        )

    (
        closest_point,
        segment_parameter
    ) = closest_point_on_segment(
        obstacle_center,
        link_start,
        link_end
    )

    center_distance = float(
        np.linalg.norm(
            closest_point
            - obstacle_center
        )
    )

    clearance = (
        center_distance
        - obstacle_radius
        - link_radius
    )

    return {
        "clearance": float(clearance),
        "center_distance": center_distance,
        "closest_point": closest_point,
        "segment_parameter": (
            segment_parameter
        ),
    }


# =====================================================
# ROBOT TO SPHERE CLEARANCES
# =====================================================

def robot_to_sphere_clearances(
    q,
    d6,
    obstacle_center,
    obstacle_radius,
    link_radius=0.0
):
    """
    Calculate clearance from every robot link
    to one spherical obstacle.

    The six robot links are represented by the
    seven Cartesian points returned by:

        get_joint_positions()

    Returns
    -------
    list of dict
        One result for each robot link.
    """

    q = np.asarray(
        q,
        dtype=float
    )

    if q.shape != (5,):
        raise ValueError(
            "q must contain five revolute joints."
        )

    obstacle_center = _as_vector3(
        obstacle_center,
        "obstacle_center"
    )

    obstacle_radius = float(
        obstacle_radius
    )

    if obstacle_radius < 0.0:
        raise ValueError(
            "obstacle_radius must be non-negative."
        )

    joint_positions = (
        get_joint_positions(
            q,
            d6
        )
    )

    number_of_links = (
        joint_positions.shape[0]
        - 1
    )

    link_radii = (
        _link_radii_array(
            link_radius,
            number_of_links
        )
    )

    results = []

    for index in range(
        number_of_links
    ):

        result = (
            link_to_sphere_clearance(
                link_start=(
                    joint_positions[index]
                ),
                link_end=(
                    joint_positions[index + 1]
                ),
                obstacle_center=(
                    obstacle_center
                ),
                obstacle_radius=(
                    obstacle_radius
                ),
                link_radius=(
                    link_radii[index]
                )
            )
        )

        result["link_index"] = (
            index
        )

        # Human-readable numbering:
        # Link 1 ... Link 6
        result["link_number"] = (
            index + 1
        )

        results.append(
            result
        )

    return results


# =====================================================
# MINIMUM ROBOT-OBSTACLE CLEARANCE
# =====================================================

def minimum_obstacle_clearance(
    q,
    d6,
    obstacle_center,
    obstacle_radius,
    link_radius=0.0
):
    """
    Return the minimum clearance between the
    complete robot chain and one spherical obstacle.

    Returns
    -------
    dict
        clearance
        link_index
        link_number
        closest_point
        center_distance
        segment_parameter
    """

    clearances = (
        robot_to_sphere_clearances(
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
    )

    minimum_result = min(
        clearances,
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
# COLLISION CHECK
# =====================================================

def robot_in_collision_with_sphere(
    q,
    d6,
    obstacle_center,
    obstacle_radius,
    link_radius=0.0
):
    """
    Return True when any robot link intersects
    or touches the spherical obstacle.
    """

    result = (
        minimum_obstacle_clearance(
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
    )

    return bool(
        result["clearance"]
        <= 0.0
    )


# =====================================================
# OBSTACLE ACTIVATION
# =====================================================

def obstacle_activation(
    clearance,
    safety_distance=0.10,
    influence_distance=0.40
):
    """
    Continuous obstacle-avoidance activation.

    clearance >= influence_distance:
        activation = 0

    clearance <= safety_distance:
        activation = 1

    Between the two:
        activation changes linearly from 0 to 1.
    """

    clearance = float(
        clearance
    )

    safety_distance = float(
        safety_distance
    )

    influence_distance = float(
        influence_distance
    )

    if safety_distance < 0.0:
        raise ValueError(
            "safety_distance must be non-negative."
        )

    if (
        influence_distance
        <= safety_distance
    ):
        raise ValueError(
            "influence_distance must be greater "
            "than safety_distance."
        )

    activation = (
        influence_distance
        - clearance
    ) / (
        influence_distance
        - safety_distance
    )

    return float(
        np.clip(
            activation,
            0.0,
            1.0
        )
    )


# =====================================================
# CLEARANCE GRADIENT
# =====================================================

def obstacle_clearance_gradient(
    q,
    d6,
    obstacle_center,
    obstacle_radius,
    link_radius=0.0,
    epsilon_revolute=1e-5,
    epsilon_prismatic=1e-5
):
    """
    Estimate the generalized gradient of minimum
    robot-obstacle clearance using central
    finite differences.

    The gradient has six elements:

        [
            d(clearance)/d(theta1),
            ...
            d(clearance)/d(theta5),
            d(clearance)/d(d6)
        ]

    Notes
    -----
    The minimum-distance function can be non-smooth
    when the identity of the closest robot link
    changes. This numerical gradient is intended
    for the first obstacle-avoidance research stage.
    """

    q = np.asarray(
        q,
        dtype=float
    ).copy()

    if q.shape != (5,):
        raise ValueError(
            "q must contain five revolute joints."
        )

    epsilon_revolute = float(
        epsilon_revolute
    )

    epsilon_prismatic = float(
        epsilon_prismatic
    )

    if epsilon_revolute <= 0.0:
        raise ValueError(
            "epsilon_revolute must be positive."
        )

    if epsilon_prismatic <= 0.0:
        raise ValueError(
            "epsilon_prismatic must be positive."
        )

    gradient = np.zeros(
        6,
        dtype=float
    )

    # Revolute coordinates
    for index in range(5):

        q_plus = q.copy()
        q_minus = q.copy()

        q_plus[index] += (
            epsilon_revolute
        )

        q_minus[index] -= (
            epsilon_revolute
        )

        clearance_plus = (
            minimum_obstacle_clearance(
                q=q_plus,
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
            )["clearance"]
        )

        clearance_minus = (
            minimum_obstacle_clearance(
                q=q_minus,
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
            )["clearance"]
        )

        gradient[index] = (
            clearance_plus
            - clearance_minus
        ) / (
            2.0
            * epsilon_revolute
        )

    # Prismatic coordinate
    clearance_plus = (
        minimum_obstacle_clearance(
            q=q,
            d6=(
                d6
                + epsilon_prismatic
            ),
            obstacle_center=(
                obstacle_center
            ),
            obstacle_radius=(
                obstacle_radius
            ),
            link_radius=(
                link_radius
            )
        )["clearance"]
    )

    clearance_minus = (
        minimum_obstacle_clearance(
            q=q,
            d6=(
                d6
                - epsilon_prismatic
            ),
            obstacle_center=(
                obstacle_center
            ),
            obstacle_radius=(
                obstacle_radius
            ),
            link_radius=(
                link_radius
            )
        )["clearance"]
    )

    gradient[5] = (
        clearance_plus
        - clearance_minus
    ) / (
        2.0
        * epsilon_prismatic
    )

    return gradient


# =====================================================
# NORMALIZED OBSTACLE AVOIDANCE DIRECTION
# =====================================================

def obstacle_avoidance_direction(
    q,
    d6,
    obstacle_center,
    obstacle_radius,
    link_radius=0.0,
    epsilon_revolute=1e-5,
    epsilon_prismatic=1e-5
):
    """
    Return a normalized generalized direction
    that locally increases robot-obstacle clearance.
    """

    gradient = (
        obstacle_clearance_gradient(
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
            epsilon_revolute=(
                epsilon_revolute
            ),
            epsilon_prismatic=(
                epsilon_prismatic
            )
        )
    )

    norm = float(
        np.linalg.norm(
            gradient
        )
    )

    if norm <= 1e-12:
        return np.zeros(
            6,
            dtype=float
        )

    return (
        gradient / norm
    )


# =====================================================
# OBSTACLE AVOIDANCE STATE
# =====================================================

def obstacle_avoidance_state(
    q,
    d6,
    obstacle_center,
    obstacle_radius,
    link_radius=0.0,
    safety_distance=0.10,
    influence_distance=0.40,
    epsilon_revolute=1e-5,
    epsilon_prismatic=1e-5
):
    """
    Calculate the main obstacle-avoidance quantities
    required by a future multi-objective controller.

    Returns
    -------
    dict
        clearance
        closest link
        activation
        gradient
        normalized avoidance direction
        collision state
    """

    minimum_result = (
        minimum_obstacle_clearance(
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
    )

    clearance = float(
        minimum_result[
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

    gradient = (
        obstacle_clearance_gradient(
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

    return {
        "clearance": clearance,
        "link_index": (
            minimum_result[
                "link_index"
            ]
        ),
        "link_number": (
            minimum_result[
                "link_number"
            ]
        ),
        "closest_point": (
            minimum_result[
                "closest_point"
            ]
        ),
        "center_distance": (
            minimum_result[
                "center_distance"
            ]
        ),
        "activation": activation,
        "gradient": gradient,
        "direction": direction,
        "collision": bool(
            clearance <= 0.0
        ),
    }
