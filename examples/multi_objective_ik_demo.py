import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from src.kinematics import get_end_effector_pose

from src.adaptive_damping import (
    solve_position_ik_adaptive,
)

from src.singularity_avoidance import (
    translational_sigma_min,
    solve_position_ik_with_singularity_avoidance,
)

from src.joint_limit_avoidance import (
    joint_limit_margin,
    joint_limit_cost,
    solve_position_ik_with_joint_limit_avoidance,
)

from src.multi_objective_ik import (
    solve_multi_objective_ik,
)


# =====================================================
# CHALLENGING INITIAL CONFIGURATION
# Near singularity and near joint limits
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
# REACHABLE TARGET
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
# INITIAL METRICS
# =====================================================

initial_sigma = translational_sigma_min(
    q_initial,
    d6_initial
)

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
# 1. ADAPTIVE DLS
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
# 2. SINGULARITY AVOIDANCE
# =====================================================

singularity_result = (
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
# 3. JOINT-LIMIT AVOIDANCE
# =====================================================

joint_limit_result = (
    solve_position_ik_with_joint_limit_avoidance(
        target_position=target_position,
        q_initial=q_initial,
        d6_initial=d6_initial,
        step_size=0.5,
        avoidance_gain=0.005,
        activation_threshold=0.80,
        tolerance=1e-4,
        max_iterations=1000,
        sigma_threshold=0.10,
        lambda_min=1e-4,
        lambda_max=0.20
    )
)


# =====================================================
# 4. MULTI-OBJECTIVE IK
# =====================================================

multi_result = solve_multi_objective_ik(
    target_position=target_position,
    q_initial=q_initial,
    d6_initial=d6_initial,
    step_size=0.5,
    singularity_gain=0.005,
    joint_limit_gain=0.005,
    joint_limit_activation=0.80,
    tolerance=1e-4,
    max_iterations=1000,
    sigma_threshold=0.10,
    lambda_min=1e-4,
    lambda_max=0.20
)


# =====================================================
# CONSISTENT METRIC HISTORIES
# =====================================================

def calculate_metric_histories(result):

    sigma_history = []
    margin_history = []
    cost_history = []

    for q, d6 in zip(
        result["q_history"],
        result["d6_history"]
    ):

        sigma_history.append(
            translational_sigma_min(
                q,
                d6
            )
        )

        margin_history.append(
            joint_limit_margin(
                q,
                d6
            )
        )

        cost_history.append(
            joint_limit_cost(
                q,
                d6,
                activation_threshold=0.80
            )
        )

    return (
        np.asarray(sigma_history),
        np.asarray(margin_history),
        np.asarray(cost_history)
    )


adaptive_sigma, adaptive_margin, adaptive_cost = (
    calculate_metric_histories(
        adaptive_result
    )
)

singularity_sigma, singularity_margin, singularity_cost = (
    calculate_metric_histories(
        singularity_result
    )
)

joint_sigma, joint_margin, joint_cost = (
    calculate_metric_histories(
        joint_limit_result
    )
)

multi_sigma, multi_margin, multi_cost = (
    calculate_metric_histories(
        multi_result
    )
)


# =====================================================
# RESULTS DIRECTORY
# =====================================================

results_dir = PROJECT_ROOT / "results"
results_dir.mkdir(exist_ok=True)


# =====================================================
# PLOT 1 — CARTESIAN ERROR
# =====================================================

fig1, ax1 = plt.subplots(
    figsize=(10, 6)
)

for result, label in [
    (adaptive_result, "Adaptive DLS"),
    (
        singularity_result,
        "Singularity Avoidance"
    ),
    (
        joint_limit_result,
        "Joint-Limit Avoidance"
    ),
    (
        multi_result,
        "Multi-Objective IK"
    )
]:

    ax1.semilogy(
        np.arange(
            len(result["error_history"])
        ),
        result["error_history"],
        linewidth=2,
        label=label
    )


ax1.set_xlabel("Iteration")
ax1.set_ylabel(
    "Cartesian Position Error [m]"
)

ax1.set_title(
    "Multi-Objective IK — Error Comparison"
)

ax1.grid(True)
ax1.legend()

fig1.tight_layout()

fig1.savefig(
    results_dir /
    "multi_objective_error_comparison.png",
    dpi=300,
    bbox_inches="tight"
)


# =====================================================
# PLOT 2 — SIGMA MIN
# =====================================================

fig2, ax2 = plt.subplots(
    figsize=(10, 6)
)

for history, label in [
    (
        adaptive_sigma,
        "Adaptive DLS"
    ),
    (
        singularity_sigma,
        "Singularity Avoidance"
    ),
    (
        joint_sigma,
        "Joint-Limit Avoidance"
    ),
    (
        multi_sigma,
        "Multi-Objective IK"
    )
]:

    ax2.plot(
        np.arange(len(history)),
        history,
        linewidth=2,
        label=label
    )


ax2.set_xlabel("Iteration")
ax2.set_ylabel(
    "Minimum Singular Value"
)

ax2.set_title(
    "Multi-Objective IK — Singularity Metric"
)

ax2.grid(True)
ax2.legend()

fig2.tight_layout()

fig2.savefig(
    results_dir /
    "multi_objective_sigma_min_comparison.png",
    dpi=300,
    bbox_inches="tight"
)


# =====================================================
# PLOT 3 — JOINT-LIMIT MARGIN
# =====================================================

fig3, ax3 = plt.subplots(
    figsize=(10, 6)
)

for history, label in [
    (
        adaptive_margin,
        "Adaptive DLS"
    ),
    (
        singularity_margin,
        "Singularity Avoidance"
    ),
    (
        joint_margin,
        "Joint-Limit Avoidance"
    ),
    (
        multi_margin,
        "Multi-Objective IK"
    )
]:

    ax3.plot(
        np.arange(len(history)),
        history,
        linewidth=2,
        label=label
    )


ax3.set_xlabel("Iteration")

ax3.set_ylabel(
    "Minimum Normalized Joint-Limit Margin"
)

ax3.set_title(
    "Multi-Objective IK — Joint-Limit Margin"
)

ax3.grid(True)
ax3.legend()

fig3.tight_layout()

fig3.savefig(
    results_dir /
    "multi_objective_joint_limit_margin.png",
    dpi=300,
    bbox_inches="tight"
)


# =====================================================
# PLOT 4 — CARTESIAN PATH
# =====================================================

fig4 = plt.figure(
    figsize=(9, 8)
)

ax4 = fig4.add_subplot(
    111,
    projection="3d"
)


for result, label in [
    (
        adaptive_result,
        "Adaptive DLS"
    ),
    (
        singularity_result,
        "Singularity Avoidance"
    ),
    (
        joint_limit_result,
        "Joint-Limit Avoidance"
    ),
    (
        multi_result,
        "Multi-Objective IK"
    )
]:

    positions = result[
        "position_history"
    ]

    ax4.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        linewidth=2,
        label=label
    )


start_position = (
    adaptive_result[
        "position_history"
    ][0]
)

ax4.scatter(
    start_position[0],
    start_position[1],
    start_position[2],
    s=80,
    marker="o",
    label="Start"
)

ax4.scatter(
    target_position[0],
    target_position[1],
    target_position[2],
    s=130,
    marker="X",
    label="Target"
)

ax4.set_xlabel("X [m]")
ax4.set_ylabel("Y [m]")
ax4.set_zlabel("Z [m]")

ax4.set_title(
    "Multi-Objective IK — Cartesian Paths"
)

ax4.grid(True)
ax4.legend()

fig4.tight_layout()

fig4.savefig(
    results_dir /
    "multi_objective_path_comparison.png",
    dpi=300,
    bbox_inches="tight"
)


# =====================================================
# FINAL METRIC FUNCTION
# =====================================================

def final_metrics(result):

    q = result["q"]
    d6 = result["d6"]

    return {
        "sigma": translational_sigma_min(
            q,
            d6
        ),

        "margin": joint_limit_margin(
            q,
            d6
        ),

        "cost": joint_limit_cost(
            q,
            d6,
            activation_threshold=0.80
        ),
    }


# =====================================================
# TERMINAL SUMMARY
# =====================================================

print("=" * 72)
print("MULTI-OBJECTIVE IK COMPARISON")
print("=" * 72)

print(
    f"\nInitial sigma_min: "
    f"{initial_sigma:.8e}"
)

print(
    f"Initial joint-limit margin: "
    f"{initial_margin:.8f}"
)

print(
    f"Initial joint-limit cost: "
    f"{initial_cost:.8f}"
)


controllers = [
    (
        "ADAPTIVE DLS",
        adaptive_result
    ),
    (
        "SINGULARITY AVOIDANCE",
        singularity_result
    ),
    (
        "JOINT-LIMIT AVOIDANCE",
        joint_limit_result
    ),
    (
        "MULTI-OBJECTIVE IK",
        multi_result
    ),
]


for name, result in controllers:

    metrics = final_metrics(
        result
    )

    print(
        f"\n{name}"
    )

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
        f"{result['final_error']:.8e} m"
    )

    print(
        f"Final sigma_min: "
        f"{metrics['sigma']:.8e}"
    )

    print(
        f"Final joint-limit margin: "
        f"{metrics['margin']:.8f}"
    )

    print(
        f"Final joint-limit cost: "
        f"{metrics['cost']:.8f}"
    )


print(
    "\nMULTI-OBJECTIVE SECONDARY TASK ACTIVITY"
)

if len(
    multi_result[
        "singularity_activity_history"
    ]
) > 0:

    print(
        "Max singularity activity: "
        f"{np.max(multi_result['singularity_activity_history']):.8e}"
    )

if len(
    multi_result[
        "joint_limit_activity_history"
    ]
) > 0:

    print(
        "Max joint-limit activity: "
        f"{np.max(multi_result['joint_limit_activity_history']):.8e}"
    )

if len(
    multi_result[
        "secondary_activity_history"
    ]
) > 0:

    print(
        "Max projected secondary activity: "
        f"{np.max(multi_result['secondary_activity_history']):.8e}"
    )


print("\nSaved figures:")

print(
    "results/multi_objective_error_comparison.png"
)

print(
    "results/multi_objective_sigma_min_comparison.png"
)

print(
    "results/multi_objective_joint_limit_margin.png"
)

print(
    "results/multi_objective_path_comparison.png"
)


plt.show()
