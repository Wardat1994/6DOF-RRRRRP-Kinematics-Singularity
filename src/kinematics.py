import numpy as np


# =====================================================
# ROBOT GEOMETRY
# =====================================================

# Link dimensions [m]
H = 0.5
a2 = 1.0
a3 = 0.8
a4 = 0.6


# =====================================================
# DENAVIT-HARTENBERG TRANSFORMATION
# =====================================================

def dh_matrix(a, alpha, d, theta):
    """
    Standard Denavit-Hartenberg homogeneous transformation matrix.
    """

    return np.array([
        [
            np.cos(theta),
            -np.sin(theta) * np.cos(alpha),
            np.sin(theta) * np.sin(alpha),
            a * np.cos(theta)
        ],
        [
            np.sin(theta),
            np.cos(theta) * np.cos(alpha),
            -np.cos(theta) * np.sin(alpha),
            a * np.sin(theta)
        ],
        [
            0.0,
            np.sin(alpha),
            np.cos(alpha),
            d
        ],
        [
            0.0,
            0.0,
            0.0,
            1.0
        ]
    ], dtype=float)


# =====================================================
# FORWARD KINEMATICS
# =====================================================

def forward_kinematics(q, d6):
    """
    Forward kinematics of the 6-DOF RRRRRP manipulator.

    Parameters
    ----------
    q : array-like
        Five revolute joint angles:
        [theta1, theta2, theta3, theta4, theta5] [rad]

    d6 : float
        Prismatic displacement of Joint 6 [m]

    Returns
    -------
    dict
        Homogeneous transformations T01 ... T06
    """

    q = np.asarray(q, dtype=float)

    if q.size != 5:
        raise ValueError(
            "q must contain five revolute joint angles."
        )

    # Individual DH transformations
    T01 = dh_matrix(
        0.0,
        np.pi / 2,
        H,
        q[0]
    )

    T12 = dh_matrix(
        a2,
        0.0,
        0.0,
        q[1]
    )

    T23 = dh_matrix(
        a3,
        0.0,
        0.0,
        q[2]
    )

    T34 = dh_matrix(
        a4,
        0.0,
        0.0,
        q[3]
    )

    T45 = dh_matrix(
        0.0,
        -np.pi / 2,
        0.0,
        q[4]
    )

    T56 = dh_matrix(
        0.0,
        0.0,
        d6,
        0.0
    )

    # Base-to-frame transformations
    T02 = T01 @ T12
    T03 = T02 @ T23
    T04 = T03 @ T34
    T05 = T04 @ T45
    T06 = T05 @ T56

    return {
        "T01": T01,
        "T02": T02,
        "T03": T03,
        "T04": T04,
        "T05": T05,
        "T06": T06
    }


# =====================================================
# JOINT POSITIONS
# =====================================================

def get_joint_positions(q, d6):
    """
    Return Cartesian positions of the base, joints,
    and end-effector.
    """

    T = forward_kinematics(q, d6)

    p0 = np.array([0.0, 0.0, 0.0])

    p1 = T["T01"][:3, 3]
    p2 = T["T02"][:3, 3]
    p3 = T["T03"][:3, 3]
    p4 = T["T04"][:3, 3]
    p5 = T["T05"][:3, 3]
    p6 = T["T06"][:3, 3]

    return np.vstack([
        p0,
        p1,
        p2,
        p3,
        p4,
        p5,
        p6
    ])


# =====================================================
# END-EFFECTOR POSE
# =====================================================

def get_end_effector_pose(q, d6):
    """
    Calculate the complete end-effector pose.

    Returns
    -------
    T06 : 4x4 transformation matrix
    position : [x, y, z]
    rotation : 3x3 orientation matrix
    """

    T06 = forward_kinematics(q, d6)["T06"]

    position = T06[:3, 3]

    rotation = T06[:3, :3]

    return T06, position, rotation


# =====================================================
# ROTATION MATRIX VALIDATION
# =====================================================

def validate_rotation_matrix(R, tolerance=1e-9):
    """
    Verify:

        R.T @ R = I

    and:

        det(R) = 1
    """

    orthogonal = np.allclose(
        R.T @ R,
        np.eye(3),
        atol=tolerance
    )

    determinant = np.isclose(
        np.linalg.det(R),
        1.0,
        atol=tolerance
    )

    return orthogonal and determinant


# =====================================================
# EXAMPLE
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

    T06, position, rotation = get_end_effector_pose(
        q,
        d6
    )

    print("T06:")
    print(T06)

    print("\nEnd-Effector Position [m]:")
    print(position)

    print("\nEnd-Effector Rotation Matrix:")
    print(rotation)

    print(
        "\nValid Rotation Matrix:",
        validate_rotation_matrix(rotation)
    )
