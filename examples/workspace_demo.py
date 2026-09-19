import sys
from pathlib import Path

import matplotlib.pyplot as plt


# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.workspace import (
    generate_workspace,
    workspace_statistics,
)


# =====================================================
# GENERATE WORKSPACE
# =====================================================

workspace_points, joint_samples = generate_workspace(
    number_of_samples=20000,
    seed=42
)


# =====================================================
# WORKSPACE STATISTICS
# =====================================================

stats = workspace_statistics(
    workspace_points
)

print("=" * 60)
print("6-DOF RRRRRP ROBOT WORKSPACE")
print("=" * 60)

print(
    f"\nGenerated points: "
    f"{len(workspace_points)}"
)

print("\nWorkspace Statistics:")

for key, value in stats.items():
    print(
        f"{key}: {value:.4f}"
    )


# =====================================================
# 3D WORKSPACE PLOT
# =====================================================

fig = plt.figure(
    figsize=(9, 8)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)

ax.scatter(
    workspace_points[:, 0],
    workspace_points[:, 1],
    workspace_points[:, 2],
    s=2,
    alpha=0.25
)

ax.set_title(
    "Reachable Workspace of the 6-DOF RRRRRP Manipulator"
)

ax.set_xlabel(
    "X [m]"
)

ax.set_ylabel(
    "Y [m]"
)

ax.set_zlabel(
    "Z [m]"
)

ax.grid(True)

plt.tight_layout()


# =====================================================
# SAVE RESULT
# =====================================================

results_dir = PROJECT_ROOT / "results"

results_dir.mkdir(
    exist_ok=True
)

output_file = (
    results_dir
    / "workspace.png"
)

fig.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

print(
    f"\nWorkspace figure saved to: "
    f"{output_file}"
)

plt.show()
