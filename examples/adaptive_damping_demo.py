import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from src.kinematics import get_end_effector_pose
from src.jacobian import translational_jacobian
from src.inverse_kinematics import solve_position_ik
from src.adaptive_damping import solve_position_ik_adaptive


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
# TARGET CONFIGURATION
# Used only to generate a reachable Cartesian target
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
# FIXED DLS
# =====================================================

fixed_damping = 0.05

fixed_result = solve_position_ik(
    target_position=target_position,
    q_initial=q_initial,
    d6_initial=d6_initial,
    damping=fixed_damping,
    step_size=0.5,
    tolerance=1e-4,
    max_iterations=1000
)


# =====================================================
# ADAPTIVE DLS
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
# FIXED-DLS SIGMA MIN HISTORY
# =====================================================

fixed_sigma_history = []

for q, d6 in zip(
    fixed_result["q_history"],
    fixed_result["d6_history"]
):

    J = translational_jacobian(
        q,
        d6
    )

    singular_values = np.linalg.svd(
        J,
        compute_uv=False
    )

    fixed_sigma_history.append(
        np.min(singular_values)
    )


fixed_sigma_history = np.asarray(
    fixed_sigma_history
)


# =====================================================
# RESULTS DIRECTORY
# =====================================================

results_dir = PROJECT_ROOT / "results"

results_dir.mkdir(
    exist_ok=True
)


# =====================================================
# PLOT 1 — ERROR CONVERGENCE
# =====================================================

fig1, ax1 = plt.subplots(
    figsize=(9, 5)
)

ax1.semilogy(
    np.arange(
        len(fixed_result["error_history"])
    ),
    fixed_result["error_history"],
    linewidth=2,
    label="Fixed DLS"
)

ax1.semilogy(
    np.arange(
        len(adaptive_result["error_history"])
    ),
    adaptive_result["error_history"],
    linewidth=2,
    label="Adaptive DLS"
)

ax1.set_xlabel(
    "Iteration"
)

ax1.set_ylabel(
    "Cartesian Error [m]"
)

ax1.set_title(
    "Fixed vs Adaptive DLS — Error Convergence"
)

ax1.grid(True)
ax1.legend()

fig1.tight_layout()

fig1.savefig(
    results_dir / "dls_error_comparison.png",
    dpi=300,
    bbox_inches="tight"
)


# =====================================================
# PLOT 2 — DAMPING COEFFICIENT
# =====================================================

fig2, ax2 = plt.subplots(
    figsize=(9, 5)
)

adaptive_damping_history = (
    adaptive_result["damping_history"]
)

ax2.plot(
    np.arange(
        len(adaptive_damping_history)
    ),
    adaptive_damping_history,
    linewidth=2,
    label="Adaptive λ"
)

ax2.axhline(
    y=fixed_damping,
    linestyle="--",
    label="Fixed λ = 0.05"
)

ax2.set_xlabel(
    "Iteration"
)

ax2.set_ylabel(
    "Damping Coefficient λ"
)

ax2.set_title(
    "Fixed vs Adaptive Damping"
)

ax2.grid(True)
ax2.legend()

fig2.tight_layout()

fig2.savefig(
    results_dir / "dls_damping_comparison.png",
    dpi=300,
    bbox_inches="tight"
)


# =====================================================
# PLOT 3 — SIGMA MIN
# =====================================================

fig3, ax3 = plt.subplots(
    figsize=(9, 5)
)

ax3.plot(
    np.arange(
        len(fixed_sigma_history)
    ),
    fixed_sigma_history,
    linewidth=2,
    label="Fixed DLS"
)

ax3.plot(
    np.arange(
        len(adaptive_result["sigma_min_history"])
    ),
    adaptive_result["sigma_min_history"],
    linewidth=2,
    label="Adaptive DLS"
)

ax3.set_xlabel(
    "Iteration"
)

ax3.set_ylabel(
    "Minimum Singular Value"
)

ax3.set_title(
    "Minimum Singular Value During IK"
)

ax3.grid(True)
ax3.legend()

fig3.tight_layout()

fig3.savefig(
    results_dir / "dls_sigma_min_comparison.png",
    dpi=300,
    bbox_inches="tight"
)


# =====================================================
# PLOT 4 — CARTESIAN PATH COMPARISON
# =====================================================

fig4 = plt.figure(
    figsize=(8, 7)
)

ax4 = fig4.add_subplot(
    111,
    projection="3d"
)

fixed_positions = (
    fixed_result["position_history"]
)

adaptive_positions = (
    adaptive_result["position_history"]
)

ax4.plot(
    fixed_positions[:, 0],
    fixed_positions[:, 1],
    fixed_positions[:, 2],
    linewidth=2,
    label="Fixed DLS"
)

ax4.plot(
    adaptive_positions[:, 0],
    adaptive_positions[:, 1],
    adaptive_positions[:, 2],
    linewidth=2,
    label="Adaptive DLS"
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
    "Fixed vs Adaptive DLS — Cartesian Path"
)

ax4.grid(True)
ax4.legend()

fig4.tight_layout()

fig4.savefig(
    results_dir / "dls_path_comparison.png",
    dpi=300,
    bbox_inches="tight"
)


# =====================================================
# TERMINAL SUMMARY
# =====================================================

print("=" * 65)
print("FIXED DLS vs ADAPTIVE DLS")
print("=" * 65)

print("\nFIXED DLS")
print(
    f"Converged: {fixed_result['converged']}"
)
print(
    f"Iterations: {fixed_result['iterations']}"
)
print(
    f"Final error: "
    f"{fixed_result['final_error']:.8e} m"
)
print(
    f"Damping λ: {fixed_damping:.6f}"
)

print("\nADAPTIVE DLS")
print(
    f"Converged: {adaptive_result['converged']}"
)
print(
    f"Iterations: {adaptive_result['iterations']}"
)
print(
    f"Final error: "
    f"{adaptive_result['final_error']:.8e} m"
)

if len(adaptive_damping_history) > 0:

    print(
        f"Minimum λ: "
        f"{np.min(adaptive_damping_history):.6f}"
    )

    print(
        f"Maximum λ: "
        f"{np.max(adaptive_damping_history):.6f}"
    )

if len(adaptive_result["sigma_min_history"]) > 0:

    print(
        f"Minimum sigma_min: "
        f"{np.min(adaptive_result['sigma_min_history']):.6e}"
    )


print("\nSaved figures:")
print("results/dls_error_comparison.png")
print("results/dls_damping_comparison.png")
print("results/dls_sigma_min_comparison.png")
print("results/dls_path_comparison.png")


plt.show()
