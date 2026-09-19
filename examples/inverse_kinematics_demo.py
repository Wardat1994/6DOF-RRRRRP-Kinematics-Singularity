import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from src.kinematics import get_end_effector_pose
from src.inverse_kinematics import solve_position_ik


# =====================================================
# INITIAL CONFIGURATION
# =====================================================

q_initial = np.radians([
    -90.0,
    0.0,
    20.0,
    -10.0,
    0.0
])

d6_initial = 0.15


# =====================================================
# TARGET POSITION
# =====================================================

q_target_reference = np.radians([
    -70.0,
    10.0,
    30.0,
    -20.0,
    15.0
])

d6_target_reference = 0.22

_, target_position, _ = get_end_effector_pose(
    q_target_reference,
    d6_target_reference
)


# =====================================================
# SOLVE INVERSE KINEMATICS
# =====================================================

result = solve_position_ik(
    target_position=target_position,
    q_initial=q_initial,
    d6_initial=d6_initial,
    damping=0.05,
    step_size=0.5,
    tolerance=1e-4,
    max_iterations=1000
)


position_history = result["position_history"]
error_history = result["error_history"]


# =====================================================
# RESULTS DIRECTORY
# =====================================================

results_dir = PROJECT_ROOT / "results"
results_dir.mkdir(exist_ok=True)


# =====================================================
# PLOT 1 — IK CARTESIAN PATH
# =====================================================

fig1 = plt.figure(
    figsize=(8, 7)
)

ax1 = fig1.add_subplot(
    111,
    projection="3d"
)

ax1.plot(
    position_history[:, 0],
    position_history[:, 1],
    position_history[:, 2],
    linewidth=2,
    label="DLS IK Path"
)

ax1.scatter(
    position_history[0, 0],
    position_history[0, 1],
    position_history[0, 2],
    s=80,
    marker="o",
    label="Start"
)

ax1.scatter(
    target_position[0],
    target_position[1],
    target_position[2],
    s=120,
    marker="X",
    label="Target"
)

ax1.scatter(
    result["final_position"][0],
    result["final_position"][1],
    result["final_position"][2],
    s=80,
    marker="s",
    label="Final Position"
)

ax1.set_xlabel("X [m]")
ax1.set_ylabel("Y [m]")
ax1.set_zlabel("Z [m]")

ax1.set_title(
    "Damped Least Squares Inverse Kinematics"
)

ax1.grid(True)
ax1.legend()

fig1.tight_layout()

fig1.savefig(
    results_dir / "inverse_kinematics_path.png",
    dpi=300,
    bbox_inches="tight"
)


# =====================================================
# PLOT 2 — ERROR CONVERGENCE
# =====================================================

iterations = np.arange(
    len(error_history)
)

fig2, ax2 = plt.subplots(
    figsize=(8, 5)
)

ax2.plot(
    iterations,
    error_history,
    linewidth=2
)

ax2.set_xlabel(
    "Iteration"
)

ax2.set_ylabel(
    "Cartesian Position Error [m]"
)

ax2.set_title(
    "DLS Inverse Kinematics Error Convergence"
)

ax2.grid(True)

fig2.tight_layout()

fig2.savefig(
    results_dir / "inverse_kinematics_error.png",
    dpi=300,
    bbox_inches="tight"
)


# =====================================================
# TERMINAL RESULTS
# =====================================================

np.set_printoptions(
    precision=6,
    suppress=True
)

print("=" * 60)
print("DLS INVERSE KINEMATICS RESULTS")
print("=" * 60)

print(
    f"Converged: "
    f"{result['converged']}"
)

print(
    f"Iterations: "
    f"{result['iterations']}"
)

print(
    f"Final Cartesian error: "
    f"{result['final_error']:.8f} m"
)

print("\nTarget position:")
print(
    result["target_position"]
)

print("\nFinal end-effector position:")
print(
    result["final_position"]
)

print("\nFinal revolute joint angles [deg]:")
print(
    np.degrees(
        result["q"]
    )
)

print(
    "\nFinal prismatic joint d6 [m]:"
)

print(
    f"{result['d6']:.6f}"
)

print("\nSaved figures:")
print(
    "results/inverse_kinematics_path.png"
)

print(
    "results/inverse_kinematics_error.png"
)

plt.show()
