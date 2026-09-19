import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.singularity import (
    geometric_singularity_metrics,
    translational_singularity_metrics,
)


# =====================================================
# TRAJECTORY THROUGH JOINT SPACE
# =====================================================

num_samples = 300

theta3_values = np.radians(
    np.linspace(-90.0, 90.0, num_samples)
)

sigma_min_values = []
condition_values = []
manipulability_values = []


for theta3 in theta3_values:

    q = np.radians([
        -90.0,
        10.0,
        0.0,
        -15.0,
        20.0
    ])

    q[2] = theta3

    d6 = 0.20

    full_metrics = geometric_singularity_metrics(
        q,
        d6
    )

    translational_metrics = (
        translational_singularity_metrics(
            q,
            d6
        )
    )

    sigma_min_values.append(
        full_metrics["sigma_min"]
    )

    condition_values.append(
        full_metrics["condition_number"]
    )

    manipulability_values.append(
        translational_metrics["manipulability"]
    )


sigma_min_values = np.array(
    sigma_min_values
)

condition_values = np.array(
    condition_values
)

manipulability_values = np.array(
    manipulability_values
)


# =====================================================
# RESULTS DIRECTORY
# =====================================================

results_dir = PROJECT_ROOT / "results"

results_dir.mkdir(
    exist_ok=True
)


# =====================================================
# PLOT 1 — MINIMUM SINGULAR VALUE
# =====================================================

fig1, ax1 = plt.subplots(
    figsize=(8, 5)
)

ax1.plot(
    np.degrees(theta3_values),
    sigma_min_values,
    linewidth=2
)

ax1.set_xlabel(
    "Joint 3 angle θ3 [deg]"
)

ax1.set_ylabel(
    "Minimum Singular Value σ_min"
)

ax1.set_title(
    "Minimum Singular Value vs Joint 3 Angle"
)

ax1.grid(True)

fig1.tight_layout()

fig1.savefig(
    results_dir / "sigma_min_vs_theta3.png",
    dpi=300,
    bbox_inches="tight"
)


# =====================================================
# PLOT 2 — CONDITION NUMBER
# =====================================================

fig2, ax2 = plt.subplots(
    figsize=(8, 5)
)

finite_mask = np.isfinite(
    condition_values
)

ax2.plot(
    np.degrees(theta3_values)[finite_mask],
    condition_values[finite_mask],
    linewidth=2
)

ax2.set_xlabel(
    "Joint 3 angle θ3 [deg]"
)

ax2.set_ylabel(
    "Jacobian Condition Number"
)

ax2.set_title(
    "Jacobian Condition Number vs Joint 3 Angle"
)

ax2.grid(True)

fig2.tight_layout()

fig2.savefig(
    results_dir / "condition_number_vs_theta3.png",
    dpi=300,
    bbox_inches="tight"
)


# =====================================================
# PLOT 3 — TRANSLATIONAL MANIPULABILITY
# =====================================================

fig3, ax3 = plt.subplots(
    figsize=(8, 5)
)

ax3.plot(
    np.degrees(theta3_values),
    manipulability_values,
    linewidth=2
)

ax3.set_xlabel(
    "Joint 3 angle θ3 [deg]"
)

ax3.set_ylabel(
    "Translational Manipulability"
)

ax3.set_title(
    "Translational Manipulability vs Joint 3 Angle"
)

ax3.grid(True)

fig3.tight_layout()

fig3.savefig(
    results_dir / "manipulability_vs_theta3.png",
    dpi=300,
    bbox_inches="tight"
)


# =====================================================
# SUMMARY
# =====================================================

minimum_index = int(
    np.argmin(
        sigma_min_values
    )
)

critical_theta3 = np.degrees(
    theta3_values[minimum_index]
)

critical_sigma = sigma_min_values[
    minimum_index
]


print("=" * 60)
print("SINGULARITY ANALYSIS RESULTS")
print("=" * 60)

print(
    f"Minimum sigma_min = "
    f"{critical_sigma:.6e}"
)

print(
    f"Critical theta3 = "
    f"{critical_theta3:.2f} deg"
)

print("\nSaved figures:")

print(
    "results/sigma_min_vs_theta3.png"
)

print(
    "results/condition_number_vs_theta3.png"
)

print(
    "results/manipulability_vs_theta3.png"
)

plt.show()
