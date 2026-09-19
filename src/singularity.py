import numpy as np

from src.jacobian import (
    translational_jacobian,
    geometric_jacobian,
)


# =====================================================
# FULL GEOMETRIC JACOBIAN ANALYSIS
# =====================================================

def geometric_singularity_metrics(
    q,
    d6,
    tolerance=1e-9
):
    """
    Analyze the complete 6x6 geometric Jacobian.

    Returns:
        rank
        determinant
        singular values
        minimum singular value
        maximum singular value
        condition number
        singular flag
    """

    J = geometric_jacobian(q, d6)

    singular_values = np.linalg.svd(
        J,
        compute_uv=False
    )

    sigma_max = float(
        np.max(singular_values)
    )

    sigma_min = float(
        np.min(singular_values)
    )

    rank = int(
        np.linalg.matrix_rank(
            J,
            tol=tolerance
        )
    )

    determinant = float(
        np.linalg.det(J)
    )

    if sigma_min <= tolerance:
        condition_number = np.inf
    else:
        condition_number = (
            sigma_max / sigma_min
        )

    is_singular = (
        rank < 6
        or sigma_min <= tolerance
    )

    return {
        "jacobian": J,
        "rank": rank,
        "determinant": determinant,
        "singular_values": singular_values,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "condition_number": condition_number,
        "is_singular": is_singular,
    }


# =====================================================
# TRANSLATIONAL SINGULARITY ANALYSIS
# =====================================================

def translational_singularity_metrics(
    q,
    d6,
    tolerance=1e-9
):
    """
    Analyze the 3x6 translational Jacobian.

    This is particularly relevant to the existing
    Cartesian position-control implementation.
    """

    Jv = translational_jacobian(
        q,
        d6
    )

    singular_values = np.linalg.svd(
        Jv,
        compute_uv=False
    )

    sigma_max = float(
        np.max(singular_values)
    )

    sigma_min = float(
        np.min(singular_values)
    )

    rank = int(
        np.linalg.matrix_rank(
            Jv,
            tol=tolerance
        )
    )

    if sigma_min <= tolerance:
        condition_number = np.inf
    else:
        condition_number = (
            sigma_max / sigma_min
        )

    # Yoshikawa translational manipulability
    manipulability = float(
        np.prod(singular_values)
    )

    is_singular = (
        rank < 3
        or sigma_min <= tolerance
    )

    return {
        "jacobian": Jv,
        "rank": rank,
        "singular_values": singular_values,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "condition_number": condition_number,
        "manipulability": manipulability,
        "is_singular": is_singular,
    }


# =====================================================
# CONFIGURATION CLASSIFICATION
# =====================================================

def classify_configuration(
    q,
    d6,
    singular_threshold=1e-4,
    warning_threshold=5e-2
):
    """
    Classify a robot configuration according to the
    minimum singular value of the geometric Jacobian.

    Returns:
        "singular"
        "near-singular"
        "regular"

    Thresholds are numerical design parameters and
    can later be tuned experimentally.
    """

    metrics = geometric_singularity_metrics(
        q,
        d6
    )

    sigma_min = metrics["sigma_min"]

    if sigma_min <= singular_threshold:
        status = "singular"

    elif sigma_min <= warning_threshold:
        status = "near-singular"

    else:
        status = "regular"

    return status, metrics


# =====================================================
# DEMO
# =====================================================

if __name__ == "__main__":

    q = np.radians([
        -90.0,
        10.0,
        30.0,
        -15.0,
        20.0
    ])

    d6 = 0.20

    status, full_metrics = classify_configuration(
        q,
        d6
    )

    position_metrics = translational_singularity_metrics(
        q,
        d6
    )

    np.set_printoptions(
        precision=6,
        suppress=True
    )

    print("=" * 60)
    print("6-DOF RRRRRP SINGULARITY ANALYSIS")
    print("=" * 60)

    print("\nConfiguration status:")
    print(status)

    print("\nFull geometric Jacobian:")
    print(full_metrics["jacobian"])

    print("\nFull Jacobian rank:")
    print(full_metrics["rank"])

    print("\nFull Jacobian determinant:")
    print(full_metrics["determinant"])

    print("\nFull Jacobian singular values:")
    print(full_metrics["singular_values"])

    print("\nMinimum singular value:")
    print(full_metrics["sigma_min"])

    print("\nCondition number:")
    print(full_metrics["condition_number"])

    print("\nTranslational manipulability:")
    print(position_metrics["manipulability"])
