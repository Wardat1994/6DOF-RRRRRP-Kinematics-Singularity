import numpy as np

from src.singularity import (
    geometric_singularity_metrics,
    translational_singularity_metrics,
    classify_configuration,
)


Q_TEST = np.radians([
    -90.0,
    10.0,
    30.0,
    -15.0,
    20.0
])

D6_TEST = 0.20


def test_geometric_singularity_metrics_structure():

    metrics = geometric_singularity_metrics(
        Q_TEST,
        D6_TEST
    )

    assert metrics["jacobian"].shape == (6, 6)

    assert len(
        metrics["singular_values"]
    ) == 6

    assert 0 <= metrics["rank"] <= 6

    assert metrics["sigma_min"] >= 0.0

    assert (
        metrics["sigma_max"]
        >= metrics["sigma_min"]
    )

    assert isinstance(
        metrics["is_singular"],
        (bool, np.bool_)
    )


def test_translational_singularity_metrics_structure():

    metrics = translational_singularity_metrics(
        Q_TEST,
        D6_TEST
    )

    assert metrics["jacobian"].shape == (3, 6)

    assert len(
        metrics["singular_values"]
    ) == 3

    assert 0 <= metrics["rank"] <= 3

    assert metrics["sigma_min"] >= 0.0

    assert metrics["manipulability"] >= 0.0

    assert isinstance(
        metrics["is_singular"],
        (bool, np.bool_)
    )


def test_manipulability_matches_singular_values():

    metrics = translational_singularity_metrics(
        Q_TEST,
        D6_TEST
    )

    expected = np.prod(
        metrics["singular_values"]
    )

    assert np.isclose(
        metrics["manipulability"],
        expected,
        atol=1e-10
    )


def test_configuration_classification():

    status, metrics = classify_configuration(
        Q_TEST,
        D6_TEST
    )

    assert status in {
        "regular",
        "near-singular",
        "singular"
    }

    assert metrics["jacobian"].shape == (6, 6)
