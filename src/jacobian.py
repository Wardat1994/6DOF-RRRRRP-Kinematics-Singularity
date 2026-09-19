import numpy as np

from src.kinematics import forward_kinematics


# =====================================================
# FRAME DATA
# =====================================================

def get_frame_origins_and_axes(q, d6):
    """
    Extract frame origins and z-axes required for
    Jacobian construction.

    Returns
    -------
    origins : list of ndarray
        o0 ... o6

    z_axes : list of ndarray
        z0 ... z5
    """

    T = forward_kinematics(q, d6)

    # Base frame
    o0 = np.array([0.0, 0.0, 0.0])
    z0 = np.array([0.0, 0.0, 1.0])

    # Frame origins
    o1 = T["T01"][:3, 3]
    o2 = T["T02"][:3, 3]
    o3 = T["T03"][:3, 3]
    o4 = T["T04"][:3, 3]
    o5 = T["T05"][:3, 3]
    o6 = T["T06"][:3, 3]

    # Frame z-axes expressed in base coordinates
    z1 = T["T01"][:3, 2]
    z2 = T["T02"][:3, 2]
    z3 = T["T03"][:3, 2]
    z4 = T["T04"][:3, 2]
    z5 = T["T05"][:3, 2]

    origins = [
        o0,
        o1,
        o2,
        o3,
        o4,
        o5,
        o6
    ]

    z_axes = [
        z0,
        z1,
        z2,
        z3,
        z4,
        z5
    ]

    return origins, z_axes


# =====================================================
# TRANSLATIONAL JACOBIAN
# =====================================================

def translational_jacobian(q, d6):
    """
    Calculate the 3x6 translational Jacobian Jv.

    Joints 1-5 are revolute.
    Joint 6 is prismatic.
    """

    origins, z_axes = get_frame_origins_and_axes(
        q,
        d6
    )

    o6 = origins[6]

    Jv = np.zeros(
        (3, 6),
        dtype=float
    )

    # Revolute joints J1 ... J5
    for i in range(5):

        Jv[:, i] = np.cross(
            z_axes[i],
            o6 - origins[i]
        )

    # Prismatic joint J6
    Jv[:, 5] = z_axes[5]

    return Jv


# =====================================================
# ANGULAR JACOBIAN
# =====================================================

def angular_jacobian(q, d6):
    """
    Calculate the 3x6 angular Jacobian Jw.
    """

    _, z_axes = get_frame_origins_and_axes(
        q,
        d6
    )

    Jw = np.zeros(
        (3, 6),
        dtype=float
    )

    # Revolute joints J1 ... J5
    for i in range(5):
        Jw[:, i] = z_axes[i]

    # Joint 6 is prismatic
    Jw[:, 5] = np.zeros(3)

    return Jw


# =====================================================
# FULL GEOMETRIC JACOBIAN
# =====================================================

def geometric_jacobian(q, d6):
    """
    Calculate the complete 6x6 geometric Jacobian:

            [ Jv ]
        J = [    ]
            [ Jw ]
    """

    Jv = translational_jacobian(
        q,
        d6
    )

    Jw = angular_jacobian(
        q,
        d6
    )

    return np.vstack([
        Jv,
        Jw
    ])


# =====================================================
# BASIC JACOBIAN INFORMATION
# =====================================================

def jacobian_info(q, d6):
    """
    Calculate basic numerical information for the
    complete geometric Jacobian.
    """

    J = geometric_jacobian(
        q,
        d6
    )

    singular_values = np.linalg.svd(
        J,
        compute_uv=False
    )

    rank = np.linalg.matrix_rank(J)

    determinant = np.linalg.det(J)

    sigma_min = np.min(
        singular_values
    )

    sigma_max = np.max(
        singular_values
    )

    if sigma_min > 1e-12:
        condition_number = (
            sigma_max / sigma_min
        )
    else:
        condition_number = np.inf

    return {
        "jacobian": J,
        "rank": rank,
        "determinant": determinant,
        "singular_values": singular_values,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "condition_number": condition_number
    }


# =====================================================
# DEMO
# =====================================================

if __name__ == "__main__":

    q = np.radians([
        -90,
        0,
        30,
        -15,
        0
    ])

    d6 = 0.20

    Jv = translational_jacobian(
        q,
        d6
    )

    J = geometric_jacobian(
        q,
        d6
    )

    info = jacobian_info(
        q,
        d6
    )

    np.set_printoptions(
        precision=5,
        suppress=True
    )

    print("Translational Jacobian Jv:")
    print(Jv)

    print("\nFull Geometric Jacobian J:")
    print(J)

    print(
        "\nRank:",
        info["rank"]
    )

    print(
        "Determinant:",
        info["determinant"]
    )

    print(
        "Singular Values:",
        info["singular_values"]
    )

    print(
        "Minimum Singular Value:",
        info["sigma_min"]
    )

    print(
        "Condition Number:",
        info["condition_number"]
    )
