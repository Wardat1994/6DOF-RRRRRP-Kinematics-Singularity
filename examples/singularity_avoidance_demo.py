import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from src.kinematics import get_end_effector_pose
from src.adaptive_damping import solve_position_ik_adaptive

from src.singularity_avoidance import (
    translational_sigma_min,
    solve_position_ik_with_singularity_avoidance,
)


# =====================================================
# NEAR-SINGULAR INITIAL CONFIGURATION
# =====================================================

q_initial = np.radians([
    -40.185,
    60.300,
    53.672,
    33.222,
    78.732
])

d6_initial = 0.4647


# =====================================================
# REACHABLE TARGET CONFIGURATION
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
# INITIAL SINGULARITY METRIC
# =====================================================

initial_sigma = translational_sigma_min(
    q_initial,
    d6_initial
)


# =====================================================
# ADAPTIVE DLS WITHOUT SINGULARITY AVOIDANCE
# =====================================================

adaptive_result = solve_position_ik_adaptive(
    target_position=target_position,
    q_initial=q_initial,
    d6_initial=d6_initial,
    step_size=0.5,
    tolerance=1e-4,
    max_iterations=1000,
    sigma_threshold=0.10,
    lambda_min=1e-4,
    lambda_max=0.20
)


# =====================================================
# ADAPTIVE DLS + SINGULARITY AVOIDANCE
# =====================================================

avoidance_result = (
    solve_position_ik_with_singularity_avoidance(
        target_position=target_position,
        q_initial=q_initial,
        d6_initial=d6_initial,
        step_size=0.5,
        avoidance_gain=0.005,
        tolerance=1e-4,
        max_iterations=1000,
        sigma_threshold=0.10,
        lambda_min=1e-4,
        lambda_max=0.20
    )
)


# =====================================================
# RESULTS DIRECTORY
# =====================================================

results_dir = PROJECT_ROOT / "results"
results_dir.mkdir(exist_ok=True)


# =====================================================
# PLOT 1 — SIGMA MIN COMPARISON
# =====================================================

fig1, ax1 = plt.subplots(
    figsize=(9, 5)
)

ax1.plot(
    np.arange(
        len(adaptive_result["sigma_min_history"])
    ),
    adaptive_result["sigma_min_history"],
    linewidth=2,
    label="Adaptive DLS"
)

ax1.plot(
    np.arange(
        len(avoidance_result["sigma_min_history"])
    ),
    avoidance_result["sigma_min_history"],
    linewidth=2,
    label="Adaptive DLS + Singularity Avoidance"
)

ax1.set_xlabel(
    "Iteration"
)

ax1.set_ylabel(
    "Minimum Singular Value"
)

ax1.set_title(
    "Singularity Avoidance — Sigma Minimum Comparison"
)

ax1.grid(True)
ax1.legend()

fig1.tight_layout()

fig1.savefig(
    results_dir / "singularity_avoidance_sigma_min.png",
    dpi=300,
    bbox_inches="tight"
)


# =====================================================
# PLOT 2 — CARTESIAN ERROR
# =====================================================

fig2, ax2 = plt.subplots(
    figsize=(9, 5)
)

ax2.semilogy(
    np.arange(
        len(adaptive_result["error_history"])
    ),
    adaptive_result["error_history"],
    linewidth=2,
    label="Adaptive DLS"
)

ax2.semilogy(
    np.arange(
        len(avoidance_result["error_history"])
    ),
    avoidance_result["error_history"],
    linewidth=2,
    label="With Singularity Avoidance"
)

ax2.set_xlabel(
    "Iteration"
)

ax2.set_ylabel(
    "Cartesian Position Error [m]"
)

ax2.set_title(
    "Singularity Avoidance — Error Convergence"
)

ax2.grid(True)
ax2.legend()

fig2.tight_layout()

fig2.savefig(
    results_dir / "singularity_avoidance_error.png",
    dpi=300,
    bbox_inches="tight"
)


# =====================================================
# PLOT 3 — NULL-SPACE AVOIDANCE ACTIVITY
# =====================================================

fig3, ax3 = plt.subplots(
    figsize=(9, 5)
)

avoidance_history = (
    avoidance_result["avoidance_history"]
)

ax3.plot(
    np.arange(
        len(avoidance_history)
    ),
    avoidance_history,
    linewidth=2
)

ax3.set_xlabel(
    "Iteration"
)

ax3.set_ylabel(
    "||Delta q avoidance||"
)

ax3.set_title(
    "Null-Space Singularity Avoidance Activity"
)

ax3.grid(True)

fig3.tight_layout()

fig3.savefig(
    results_dir / "singularity_avoidance_activity.png",
    dpi=300,
    bbox_inches="tight"
)


# =====================================================
# PLOT 4 — CARTESIAN PATH COMPARISON
# =====================================================

adaptive_positions = (
    adaptive_result["position_history"]
)

avoidance_positions = (
    avoidance_result["position_history"]
)


fig4 = plt.figure(
    figsize=(8, 7)
)

ax4 = fig4.add_subplot(
    111,
    projection="3d"
)

ax4.plot(
    adaptive_positions[:, 0],
    adaptive_positions[:, 1],
    adaptive_positions[:, 2],
    linewidth=2,
    label="Adaptive DLS"
)

ax4.plot(
    avoidance_positions[:, 0],
    avoidance_positions[:, 1],
    avoidance_positions[:, 2],
    linewidth=2,
    label="With Singularity Avoidance"
)

ax4.scatter(
    adaptive_positions[0, 0],
    adaptive_positions[0, 1],
    adaptive_positions[0, 2],
    s=80,
    marker="o",
    label="Start"
)

ax4.scatter(
    target_position[0],
    target_position[1],
    target_position[2],
    s=120,
    marker="X",
    label="Target"
)

ax4.set_xlabel("X [m]")
ax4.set_ylabel("Y [m]")
ax4.set_zlabel("Z [m]")

ax4.set_title(
    "Cartesian Path — Singularity Avoidance"
)

ax4.grid(True)
ax4.legend()

fig4.tight_layout()

fig4.savefig(
    results_dir / "singularity_avoidance_path.png",
    dpi=300,
    bbox_inches="tight"
)


# =====================================================
# FINAL SIGMA VALUES
# =====================================================

adaptive_final_sigma = (
    translational_sigma_min(
        adaptive_result["q"],
        adaptive_result["d6"]
    )
)

avoidance_final_sigma = (
    translational_sigma_min(
        avoidance_result["q"],
        avoidance_result["d6"]
    )
)


# =====================================================
# TERMINAL SUMMARY
# =====================================================

print("=" * 70)
print("SINGULARITY AVOIDANCE COMPARISON")
print("=" * 70)

print(
    f"\nInitial sigma_min: "
    f"{initial_sigma:.8e}"
)

print("\nADAPTIVE DLS")
print(
    f"Converged: "
    f"{adaptive_result['converged']}"
)

print(
    f"Iterations: "
    f"{adaptive_result['iterations']}"
)

print(
    f"Final error: "
    f"{adaptive_result['final_error']:.8e} m"
)

print(
    f"Final sigma_min: "
    f"{adaptive_final_sigma:.8e}"
)


print("\nADAPTIVE DLS + SINGULARITY AVOIDANCE")

print(
    f"Converged: "
    f"{avoidance_result['converged']}"
)

print(
    f"Iterations: "
    f"{avoidance_result['iterations']}"
)

print(
    f"Final error: "
    f"{avoidance_result['final_error']:.8e} m"
)

print(
    f"Final sigma_min: "
    f"{avoidance_final_sigma:.8e}"
)

if len(avoidance_history) > 0:

    print(
        f"Maximum avoidance activity: "
        f"{np.max(avoidance_history):.8e}"
    )


print("\nSaved figures:")

print(
    "results/singularity_avoidance_sigma_min.png"
)

print(
    "results/singularity_avoidance_error.png"
)

print(
    "results/singularity_avoidance_activity.png"
)

print(
    "results/singularity_avoidance_path.png"
)


plt.show()
