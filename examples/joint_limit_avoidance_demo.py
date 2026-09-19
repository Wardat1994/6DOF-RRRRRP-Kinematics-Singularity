import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from src.kinematics import get_end_effector_pose
from src.adaptive_damping import solve_position_ik_adaptive

from src.joint_limit_avoidance import (
    joint_limit_margin,
    joint_limit_cost,
    solve_position_ik_with_joint_limit_avoidance,
)


# =====================================================
# INITIAL CONFIGURATION NEAR JOINT LIMITS
# =====================================================

q_initial = np.radians([
    170.0,
    82.0,
    78.0,
    -82.0,
    80.0
])

d6_initial = 0.46


# =====================================================
# REACHABLE TARGET CONFIGURATION
# =====================================================

q_target_reference = np.radians([
    120.0,
    45.0,
    40.0,
    -40.0,
    45.0
])

d6_target_reference = 0.30


_, target_position, _ = get_end_effector_pose(
    q_target_reference,
    d6_target_reference
)


# =====================================================
# INITIAL JOINT-LIMIT METRICS
# =====================================================

initial_margin = joint_limit_margin(
    q_initial,
    d6_initial
)

initial_cost = joint_limit_cost(
    q_initial,
    d6_initial,
    activation_threshold=0.80
)


# =====================================================
# ADAPTIVE DLS WITHOUT JOINT-LIMIT AVOIDANCE
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
# ADAPTIVE DLS + JOINT-LIMIT AVOIDANCE
# =====================================================

avoidance_result = (
    solve_position_ik_with_joint_limit_avoidance(
        target_position=target_position,
        q_initial=q_initial,
        d6_initial=d6_initial,
        step_size=0.5,
        avoidance_gain=0.01,
        activation_threshold=0.80,
        tolerance=1e-4,
        max_iterations=1000,
        sigma_threshold=0.10,
        lambda_min=1e-4,
        lambda_max=0.20
    )
)


# =====================================================
# CALCULATE BASELINE JOINT-LIMIT HISTORIES
# =====================================================

adaptive_margin_history = []

adaptive_cost_history = []


for q, d6 in zip(
    adaptive_result["q_history"],
    adaptive_result["d6_history"]
):

    adaptive_margin_history.append(
        joint_limit_margin(
            q,
            d6
        )
    )

    adaptive_cost_history.append(
        joint_limit_cost(
            q,
            d6,
            activation_threshold=0.80
        )
    )


adaptive_margin_history = np.asarray(
    adaptive_margin_history
)

adaptive_cost_history = np.asarray(
    adaptive_cost_history
)


# =====================================================
# RESULTS DIRECTORY
# =====================================================

results_dir = PROJECT_ROOT / "results"

results_dir.mkdir(
    exist_ok=True
)


# =====================================================
# PLOT 1 — JOINT-LIMIT MARGIN
# =====================================================

fig1, ax1 = plt.subplots(
    figsize=(9, 5)
)

ax1.plot(
    np.arange(
        len(adaptive_margin_history)
    ),
    adaptive_margin_history,
    linewidth=2,
    label="Adaptive DLS"
)

ax1.plot(
    np.arange(
        len(
            avoidance_result[
                "joint_limit_margin_history"
            ]
        )
    ),
    avoidance_result[
        "joint_limit_margin_history"
    ],
    linewidth=2,
    label="Adaptive DLS + Joint-Limit Avoidance"
)

ax1.set_xlabel(
    "Iteration"
)

ax1.set_ylabel(
    "Minimum Normalized Joint-Limit Margin"
)

ax1.set_title(
    "Joint-Limit Margin Comparison"
)

ax1.grid(True)
ax1.legend()

fig1.tight_layout()

fig1.savefig(
    results_dir / "joint_limit_margin_comparison.png",
    dpi=300,
    bbox_inches="tight"
)


# =====================================================
# PLOT 2 — JOINT-LIMIT COST
# =====================================================

fig2, ax2 = plt.subplots(
    figsize=(9, 5)
)

ax2.plot(
    np.arange(
        len(adaptive_cost_history)
    ),
    adaptive_cost_history,
    linewidth=2,
    label="Adaptive DLS"
)

ax2.plot(
    np.arange(
        len(
            avoidance_result[
                "joint_limit_cost_history"
            ]
        )
    ),
    avoidance_result[
        "joint_limit_cost_history"
    ],
    linewidth=2,
    label="Adaptive DLS + Joint-Limit Avoidance"
)

ax2.set_xlabel(
    "Iteration"
)

ax2.set_ylabel(
    "Joint-Limit Cost"
)

ax2.set_title(
    "Joint-Limit Penalty Comparison"
)

ax2.grid(True)
ax2.legend()

fig2.tight_layout()

fig2.savefig(
    results_dir / "joint_limit_cost_comparison.png",
    dpi=300,
    bbox_inches="tight"
)


# =====================================================
# PLOT 3 — CARTESIAN ERROR
# =====================================================

fig3, ax3 = plt.subplots(
    figsize=(9, 5)
)

ax3.semilogy(
    np.arange(
        len(adaptive_result["error_history"])
    ),
    adaptive_result["error_history"],
    linewidth=2,
    label="Adaptive DLS"
)

ax3.semilogy(
    np.arange(
        len(avoidance_result["error_history"])
    ),
    avoidance_result["error_history"],
    linewidth=2,
    label="With Joint-Limit Avoidance"
)

ax3.set_xlabel(
    "Iteration"
)

ax3.set_ylabel(
    "Cartesian Position Error [m]"
)

ax3.set_title(
    "Joint-Limit Avoidance — Error Convergence"
)

ax3.grid(True)
ax3.legend()

fig3.tight_layout()

fig3.savefig(
    results_dir / "joint_limit_error_comparison.png",
    dpi=300,
    bbox_inches="tight"
)


# =====================================================
# PLOT 4 — CARTESIAN PATH
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
    label="With Joint-Limit Avoidance"
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
    "Cartesian Path — Joint-Limit Avoidance"
)

ax4.grid(True)
ax4.legend()

fig4.tight_layout()

fig4.savefig(
    results_dir / "joint_limit_path_comparison.png",
    dpi=300,
    bbox_inches="tight"
)


# =====================================================
# FINAL METRICS
# =====================================================

adaptive_final_margin = joint_limit_margin(
    adaptive_result["q"],
    adaptive_result["d6"]
)

adaptive_final_cost = joint_limit_cost(
    adaptive_result["q"],
    adaptive_result["d6"],
    activation_threshold=0.80
)


avoidance_final_margin = joint_limit_margin(
    avoidance_result["q"],
    avoidance_result["d6"]
)

avoidance_final_cost = joint_limit_cost(
    avoidance_result["q"],
    avoidance_result["d6"],
    activation_threshold=0.80
)


# =====================================================
# TERMINAL SUMMARY
# =====================================================

print("=" * 70)
print("JOINT-LIMIT AVOIDANCE COMPARISON")
print("=" * 70)

print(
    f"\nInitial joint-limit margin: "
    f"{initial_margin:.8f}"
)

print(
    f"Initial joint-limit cost: "
    f"{initial_cost:.8f}"
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
    f"Final Cartesian error: "
    f"{adaptive_result['final_error']:.8e} m"
)

print(
    f"Final joint-limit margin: "
    f"{adaptive_final_margin:.8f}"
)

print(
    f"Final joint-limit cost: "
    f"{adaptive_final_cost:.8f}"
)


print(
    "\nADAPTIVE DLS + JOINT-LIMIT AVOIDANCE"
)

print(
    f"Converged: "
    f"{avoidance_result['converged']}"
)

print(
    f"Iterations: "
    f"{avoidance_result['iterations']}"
)

print(
    f"Final Cartesian error: "
    f"{avoidance_result['final_error']:.8e} m"
)

print(
    f"Final joint-limit margin: "
    f"{avoidance_final_margin:.8f}"
)

print(
    f"Final joint-limit cost: "
    f"{avoidance_final_cost:.8f}"
)


if len(
    avoidance_result["avoidance_history"]
) > 0:

    print(
        f"Maximum avoidance activity: "
        f"{np.max(avoidance_result['avoidance_history']):.8e}"
    )


print("\nSaved figures:")

print(
    "results/joint_limit_margin_comparison.png"
)

print(
    "results/joint_limit_cost_comparison.png"
)

print(
    "results/joint_limit_error_comparison.png"
)

print(
    "results/joint_limit_path_comparison.png"
)


plt.show()
