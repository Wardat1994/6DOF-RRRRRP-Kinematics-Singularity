import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from src.trajectory import generate_end_effector_trajectory
from src.singularity import translational_singularity_metrics


# =====================================================
# TRAJECTORY CONFIGURATION
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

# Assumed trajectory execution time
duration = 10.0  # seconds


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

time = np.linspace(
    0.0,
    duration,
    num_points
)


# =====================================================
# CARTESIAN VELOCITY
# =====================================================

velocity = np.gradient(
    positions,
    time,
    axis=0
)

speed = np.linalg.norm(
    velocity,
    axis=1
)


# =====================================================
# SINGULARITY METRICS ALONG TRAJECTORY
# =====================================================

sigma_min_values = np.zeros(
    num_points
)

manipulability_values = np.zeros(
    num_points
)

condition_values = np.zeros(
    num_points
)


for i in range(num_points):

    metrics = translational_singularity_metrics(
        q_trajectory[i],
        d6_trajectory[i]
    )

    sigma_min_values[i] = (
        metrics["sigma_min"]
    )

    manipulability_values[i] = (
        metrics["manipulability"]
    )

    condition_values[i] = (
        metrics["condition_number"]
    )


# =====================================================
# RESULTS DIRECTORY
# =====================================================

results_dir = PROJECT_ROOT / "results"

results_dir.mkdir(
    exist_ok=True
)


# =====================================================
# PLOT 1 — X, Y, Z VS TIME
# =====================================================

fig1, ax1 = plt.subplots(
    figsize=(9, 5)
)

ax1.plot(
    time,
    positions[:, 0],
    label="X"
)

ax1.plot(
    time,
    positions[:, 1],
    label="Y"
)

ax1.plot(
    time,
    positions[:, 2],
    label="Z"
)

ax1.set_xlabel(
    "Time [s]"
)

ax1.set_ylabel(
    "Position [m]"
)

ax1.set_title(
    "End-Effector Cartesian Position"
)

ax1.grid(True)
ax1.legend()

fig1.tight_layout()

fig1.savefig(
    results_dir / "trajectory_xyz_vs_time.png",
    dpi=300,
    bbox_inches="tight"
)


# =====================================================
# PLOT 2 — CARTESIAN SPEED
# =====================================================

fig2, ax2 = plt.subplots(
    figsize=(9, 5)
)

ax2.plot(
    time,
    speed,
    linewidth=2
)

ax2.set_xlabel(
    "Time [s]"
)

ax2.set_ylabel(
    "Speed [m/s]"
)

ax2.set_title(
    "End-Effector Cartesian Speed"
)

ax2.grid(True)

fig2.tight_layout()

fig2.savefig(
    results_dir / "trajectory_speed.png",
    dpi=300,
    bbox_inches="tight"
)


# =====================================================
# PLOT 3 — MANIPULABILITY
# =====================================================

fig3, ax3 = plt.subplots(
    figsize=(9, 5)
)

ax3.plot(
    time,
    manipulability_values,
    linewidth=2
)

ax3.set_xlabel(
    "Time [s]"
)

ax3.set_ylabel(
    "Translational Manipulability"
)

ax3.set_title(
    "Manipulability Along Trajectory"
)

ax3.grid(True)

fig3.tight_layout()

fig3.savefig(
    results_dir / "trajectory_manipulability.png",
    dpi=300,
    bbox_inches="tight"
)


# =====================================================
# PLOT 4 — MINIMUM SINGULAR VALUE
# =====================================================

fig4, ax4 = plt.subplots(
    figsize=(9, 5)
)

ax4.plot(
    time,
    sigma_min_values,
    linewidth=2
)

ax4.set_xlabel(
    "Time [s]"
)

ax4.set_ylabel(
    "Minimum Singular Value"
)

ax4.set_title(
    "Minimum Singular Value Along Trajectory"
)

ax4.grid(True)

fig4.tight_layout()

fig4.savefig(
    results_dir / "trajectory_sigma_min.png",
    dpi=300,
    bbox_inches="tight"
)


# =====================================================
# SUMMARY
# =====================================================

critical_index = int(
    np.argmin(
        sigma_min_values
    )
)

print("=" * 60)
print("TRAJECTORY PERFORMANCE ANALYSIS")
print("=" * 60)

print(
    f"Trajectory duration: "
    f"{duration:.2f} s"
)

print(
    f"Maximum Cartesian speed: "
    f"{np.max(speed):.6f} m/s"
)

print(
    f"Average Cartesian speed: "
    f"{np.mean(speed):.6f} m/s"
)

print(
    f"Minimum sigma_min: "
    f"{sigma_min_values[critical_index]:.6e}"
)

print(
    f"Critical time: "
    f"{time[critical_index]:.3f} s"
)

print(
    f"Minimum manipulability: "
    f"{np.min(manipulability_values):.6e}"
)

print("\nSaved figures:")
print("results/trajectory_xyz_vs_time.png")
print("results/trajectory_speed.png")
print("results/trajectory_manipulability.png")
print("results/trajectory_sigma_min.png")


plt.show()
