import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from src.trajectory import (
    generate_end_effector_trajectory,
    calculate_trajectory_length,
)


# =====================================================
# START AND END CONFIGURATIONS
# =====================================================

q_start = np.radians([
    -90.0,
    0.0,
    20.0,
    -10.0,
    0.0
])

q_end = np.radians([
    -60.0,
    20.0,
    45.0,
    -25.0,
    30.0
])

d6_start = 0.10
d6_end = 0.30

num_points = 200


# =====================================================
# GENERATE TRAJECTORY
# =====================================================

q_trajectory, d6_trajectory, positions = (
    generate_end_effector_trajectory(
        q_start,
        d6_start,
        q_end,
        d6_end,
        num_points=num_points
    )
)


trajectory_length = (
    calculate_trajectory_length(
        positions
    )
)


# =====================================================
# RESULTS DIRECTORY
# =====================================================

results_dir = PROJECT_ROOT / "results"

results_dir.mkdir(
    exist_ok=True
)


# =====================================================
# 3D TRAJECTORY PLOT
# =====================================================

fig = plt.figure(
    figsize=(8, 7)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)


ax.plot(
    positions[:, 0],
    positions[:, 1],
    positions[:, 2],
    linewidth=2,
    label="End-Effector Trajectory"
)


# Start point
ax.scatter(
    positions[0, 0],
    positions[0, 1],
    positions[0, 2],
    s=80,
    marker="o",
    label="Start"
)


# End point
ax.scatter(
    positions[-1, 0],
    positions[-1, 1],
    positions[-1, 2],
    s=100,
    marker="X",
    label="End"
)


ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")

ax.set_title(
    "3D End-Effector Trajectory"
)

ax.grid(True)

ax.legend()


# =====================================================
# BALANCED AXES
# =====================================================

x_min = np.min(positions[:, 0])
x_max = np.max(positions[:, 0])

y_min = np.min(positions[:, 1])
y_max = np.max(positions[:, 1])

z_min = np.min(positions[:, 2])
z_max = np.max(positions[:, 2])


x_mid = (x_min + x_max) / 2.0
y_mid = (y_min + y_max) / 2.0
z_mid = (z_min + z_max) / 2.0


max_range = max(
    x_max - x_min,
    y_max - y_min,
    z_max - z_min
)


if max_range == 0:
    max_range = 1.0


half_range = max_range / 2.0


ax.set_xlim(
    x_mid - half_range,
    x_mid + half_range
)

ax.set_ylim(
    y_mid - half_range,
    y_mid + half_range
)

ax.set_zlim(
    z_mid - half_range,
    z_mid + half_range
)


fig.tight_layout()


# =====================================================
# SAVE RESULT
# =====================================================

output_path = (
    results_dir /
    "trajectory_3d.png"
)

fig.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)


# =====================================================
# TERMINAL OUTPUT
# =====================================================

print("=" * 60)
print("END-EFFECTOR TRAJECTORY ANALYSIS")
print("=" * 60)

print(
    f"Number of trajectory points: "
    f"{num_points}"
)

print(
    f"Cartesian trajectory length: "
    f"{trajectory_length:.6f} m"
)

print("\nStart position:")
print(
    positions[0]
)

print("\nEnd position:")
print(
    positions[-1]
)

print(
    "\nSaved figure:"
)

print(
    "results/trajectory_3d.png"
)


plt.show()
