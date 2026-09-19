import numpy as np

from src.kinematics import (
    forward_kinematics,
    get_joint_positions,
    get_end_effector_pose,
    validate_rotation_matrix,
)


def test_forward_kinematics():

    q = np.radians([
        -90,
        0,
        30,
        -15,
        0
    ])

    d6 = 0.20

    transforms = forward_kinematics(q, d6)

    # Check that all transformations exist
    assert "T01" in transforms
    assert "T02" in transforms
    assert "T03" in transforms
    assert "T04" in transforms
    assert "T05" in transforms
    assert "T06" in transforms

    # Every homogeneous transformation must be 4x4
    for T in transforms.values():
        assert T.shape == (4, 4)


def test_joint_positions():

    q = np.radians([
        -90,
        0,
        30,
        -15,
        0
    ])

    d6 = 0.20

    points = get_joint_positions(q, d6)

    # Base + 6 joints / end-effector
    assert points.shape == (7, 3)

    # Base position must be the origin
    assert np.allclose(
        points[0],
        np.array([0.0, 0.0, 0.0])
    )


def test_end_effector_pose():

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

    assert T06.shape == (4, 4)

    assert position.shape == (3,)

    assert rotation.shape == (3, 3)

    # Homogeneous transformation final row
    assert np.allclose(
        T06[3, :],
        [0.0, 0.0, 0.0, 1.0]
    )

    # Rotation matrix validation
    assert validate_rotation_matrix(rotation)

    # R^T R = I
    assert np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        atol=1e-9
    )

    # det(R) = 1
    assert np.isclose(
        np.linalg.det(rotation),
        1.0,
        atol=1e-9
    )
