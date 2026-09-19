import numpy as np

from src.kinematics import get_end_effector_pose
from src.jacobian import (
    translational_jacobian,
    angular_jacobian,
    geometric_jacobian,
    jacobian_info,
)


# =====================================================
# TEST CONFIGURATION
# =====================================================

Q_TEST = np.radians([
    -90.0,
    10.0,
    30.0,
    -15.0,
    20.0
])

D6_TEST = 0.20


# =====================================================
# DIMENSION TESTS
# =====================================================

def test_translational_jacobian_shape():

    Jv = translational_jacobian(
        Q_TEST,
        D6_TEST
    )

    assert Jv.shape == (3, 6)


def test_angular_jacobian_shape():

    Jw = angular_jacobian(
        Q_TEST,
        D6_TEST
    )

    assert Jw.shape == (3, 6)


def test_geometric_jacobian_shape():

    J = geometric_jacobian(
        Q_TEST,
        D6_TEST
    )

    assert J.shape == (6, 6)


# =====================================================
# PRISMATIC JOINT TEST
# =====================================================

def test_prismatic_joint_has_no_angular_velocity():

    Jw = angular_jacobian(
        Q_TEST,
        D6_TEST
    )

    assert np.allclose(
        Jw[:, 5],
        np.zeros(3)
    )


# =====================================================
# FINITE-DIFFERENCE VALIDATION
# =====================================================

def test_translational_jacobian_with_finite_difference():
    """
    Validate Jv against numerical differentiation
    of the forward kinematics.
    """

    Jv = translational_jacobian(
        Q_TEST,
        D6_TEST
    )

    epsilon = 1e-7

    J_numeric = np.zeros(
        (3, 6)
    )

    # Revolute joints
    for i in range(5):

        q_plus = Q_TEST.copy()
        q_minus = Q_TEST.copy()

        q_plus[i] += epsilon
        q_minus[i] -= epsilon

        _, p_plus, _ = get_end_effector_pose(
            q_plus,
            D6_TEST
        )

        _, p_minus, _ = get_end_effector_pose(
            q_minus,
            D6_TEST
        )

        J_numeric[:, i] = (
            p_plus - p_minus
        ) / (2.0 * epsilon)

    # Prismatic joint
    _, p_plus, _ = get_end_effector_pose(
        Q_TEST,
        D6_TEST + epsilon
    )

    _, p_minus, _ = get_end_effector_pose(
        Q_TEST,
        D6_TEST - epsilon
    )

    J_numeric[:, 5] = (
        p_plus - p_minus
    ) / (2.0 * epsilon)

    assert np.allclose(
        Jv,
        J_numeric,
        atol=1e-6
    )


# =====================================================
# SVD / JACOBIAN INFORMATION TEST
# =====================================================

def test_jacobian_information():

    info = jacobian_info(
        Q_TEST,
        D6_TEST
    )

    assert info["jacobian"].shape == (6, 6)

    assert len(
        info["singular_values"]
    ) == 6

    assert (
        0
        <= info["rank"]
        <= 6
    )

    assert info["sigma_min"] >= 0.0

    assert (
        info["sigma_max"]
        >= info["sigma_min"]
    )
