import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# =====================================================
# PROJECT PATH
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# =====================================================
# IMPORTS
# =====================================================

from src.kinematics import (
    get_end_effector_pose,
    get_joint_positions,
)

from src.multi_objective_ik import (
    solve_multi_objective_ik,
)

from src.obstacle_aware_ik import (
    solve_obstacle_aware_ik,
)

from src.obstacle_avoidance import (
    minimum_obstacle_clearance,
    obstacle_activation,
)


# =====================================================
# INITIAL CONFIGURATION
# =====================================================

q_initial = np.radians([
    -90.0,
    0.0,
    30.0,
    -15.0,
    0.0
])

d6_initial = 0.20


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


_, target_position, _ = (
    get_end_effector_pose(
        q_target_reference,
        d6_target_reference
    )
)


# =====================================================
# COMMON CONTROLLER PARAMETERS
# =====================================================

step_size = 0.5

singularity_gain = 0.005
joint_limit_gain = 0.005

joint_limit_activation = 0.80
safety_margin = 0.05

sigma_threshold = 0.10

lambda_min = 1e-4
lambda_max = 0.20

tolerance = 1e-4
max_iterations = 1000


# =====================================================
# OBSTACLE PARAMETERS
# =====================================================

obstacle_radius = 0.10

link_radius = 0.05

obstacle_safety_distance = 0.10

obstacle_influence_distance = 0.40

obstacle_gain = 0.03


# =====================================================
# BASELINE MULTI-OBJECTIVE IK
# =====================================================

baseline_result = (
    solve_multi_objective_ik(
        target_position=target_position,
        q_initial=q_initial,
        d6_initial=d6_initial,

        step_size=step_size,

        singularity_gain=(
            singularity_gain
        ),

        joint_limit_gain=(
            joint_limit_gain
        ),

        joint_limit_activation=(
            joint_limit_activation
        ),

        safety_margin=(
            safety_margin
        ),

        tolerance=tolerance,

        max_iterations=(
            max_iterations
        ),

        sigma_threshold=(
            sigma_threshold
        ),

        lambda_min=lambda_min,
        lambda_max=lambda_max
    )
)


# =====================================================
# BUILD A CONTROLLED CHALLENGING OBSTACLE
# =====================================================
#
# The obstacle is generated from the baseline
# trajectory so that it lies near a robot link
# during the motion.
#
# We search for a position that:
#
# 1. Does not initially collide with the robot.
# 2. Does not block the final configuration.
# 3. Creates a low clearance during the baseline.
#
# This gives a meaningful comparison between:
#
# Multi-Objective IK
#
# and:
#
# Obstacle-Aware Multi-Objective IK
# =====================================================


def calculate_clearance(
    q,
    d6,
    obstacle_center
):

    return float(
        minimum_obstacle_clearance(
            q=q,
            d6=d6,

            obstacle_center=(
                obstacle_center
            ),

            obstacle_radius=(
                obstacle_radius
            ),

            link_radius=(
                link_radius
            )
        )["clearance"]
    )


def create_perpendicular_direction(
    segment
):

    segment_norm = float(
        np.linalg.norm(
            segment
        )
    )

    if segment_norm <= 1e-12:
        return None

    direction = (
        segment
        / segment_norm
    )

    reference = np.array([
        0.0,
        0.0,
        1.0
    ])

    if abs(
        float(
            direction @ reference
        )
    ) > 0.90:

        reference = np.array([
            1.0,
            0.0,
            0.0
        ])

    perpendicular = np.cross(
        direction,
        reference
    )

    norm = float(
        np.linalg.norm(
            perpendicular
        )
    )

    if norm <= 1e-12:
        return None

    return (
        perpendicular
        / norm
    )


def build_challenging_obstacle():

    q_history = (
        baseline_result[
            "q_history"
        ]
    )

    d6_history = (
        baseline_result[
            "d6_history"
        ]
    )

    number_of_states = len(
        q_history
    )

    candidate_indices = np.unique(
        np.linspace(
            max(
                1,
                int(
                    0.25
                    * number_of_states
                )
            ),
            max(
                1,
                int(
                    0.75
                    * number_of_states
                )
            ),
            15
        ).astype(int)
    )

    candidate_indices = np.clip(
        candidate_indices,
        0,
        number_of_states - 1
    )

    # Non-zero major physical links:
    # points 1->2
    # points 2->3
    # points 3->4
    candidate_link_indices = [
        1,
        2,
        3,
    ]

    desired_baseline_clearance = (
        0.03
    )

    centerline_offset = (
        obstacle_radius
        + link_radius
        + desired_baseline_clearance
    )

    candidates = []

    for state_index in candidate_indices:

        q_state = (
            q_history[
                state_index
            ]
        )

        d6_state = float(
            d6_history[
                state_index
            ]
        )

        points = (
            get_joint_positions(
                q_state,
                d6_state
            )
        )

        for link_index in (
            candidate_link_indices
        ):

            start = (
                points[
                    link_index
                ]
            )

            end = (
                points[
                    link_index + 1
                ]
            )

            segment = (
                end
                - start
            )

            perpendicular = (
                create_perpendicular_direction(
                    segment
                )
            )

            if perpendicular is None:
                continue

            midpoint = (
                0.5
                * (
                    start
                    + end
                )
            )

            for sign in [
                -1.0,
                1.0
            ]:

                center = (
                    midpoint
                    + sign
                    * centerline_offset
                    * perpendicular
                )

                current_clearance = (
                    calculate_clearance(
                        q_state,
                        d6_state,
                        center
                    )
                )

                initial_clearance = (
                    calculate_clearance(
                        q_initial,
                        d6_initial,
                        center
                    )
                )

                final_clearance = (
                    calculate_clearance(
                        baseline_result[
                            "q"
                        ],
                        baseline_result[
                            "d6"
                        ],
                        center
                    )
                )

                # Desired candidate:
                #
                # - positive baseline clearance
                # - below safety distance
                # - start and goal remain comfortably safe
                if (
                    current_clearance
                    > 0.01
                    and current_clearance
                    < obstacle_safety_distance
                    and initial_clearance
                    > obstacle_safety_distance
                    and final_clearance
                    > obstacle_safety_distance
                ):

                    candidates.append({
                        "center": center,
                        "state_index": (
                            state_index
                        ),
                        "link_index": (
                            link_index
                        ),
                        "clearance": (
                            current_clearance
                        ),
                        "initial_clearance": (
                            initial_clearance
                        ),
                        "final_clearance": (
                            final_clearance
                        ),
                    })

    if len(candidates) > 0:

        best = min(
            candidates,
            key=lambda item: abs(
                item[
                    "clearance"
                ]
                - desired_baseline_clearance
            )
        )

        return best

    # ---------------------------------------------
    # FALLBACK
    # ---------------------------------------------

    state_index = (
        number_of_states // 2
    )

    points = (
        get_joint_positions(
            q_history[
                state_index
            ],
            float(
                d6_history[
                    state_index
                ]
            )
        )
    )

    link_index = 2

    start = points[
        link_index
    ]

    end = points[
        link_index + 1
    ]

    midpoint = (
        0.5
        * (
            start
            + end
        )
    )

    perpendicular = (
        create_perpendicular_direction(
            end - start
        )
    )

    if perpendicular is None:

        perpendicular = np.array([
            1.0,
            0.0,
            0.0
        ])

    center = (
        midpoint
        + centerline_offset
        * perpendicular
    )

    return {
        "center": center,
        "state_index": (
            state_index
        ),
        "link_index": (
            link_index
        ),
        "clearance": (
            calculate_clearance(
                q_history[
                    state_index
                ],
                float(
                    d6_history[
                        state_index
                    ]
                ),
                center
            )
        ),
        "initial_clearance": (
            calculate_clearance(
                q_initial,
                d6_initial,
                center
            )
        ),
        "final_clearance": (
            calculate_clearance(
                baseline_result[
                    "q"
                ],
                baseline_result[
                    "d6"
                ],
                center
            )
        ),
    }


obstacle_design = (
    build_challenging_obstacle()
)

obstacle_center = (
    obstacle_design[
        "center"
    ]
)


# =====================================================
# OBSTACLE-AWARE CONTROLLER
# =====================================================

obstacle_result = (
    solve_obstacle_aware_ik(
        target_position=(
            target_position
        ),

        q_initial=q_initial,

        d6_initial=d6_initial,

        obstacle_center=(
            obstacle_center
        ),

        obstacle_radius=(
            obstacle_radius
        ),

        link_radius=(
            link_radius
        ),

        step_size=step_size,

        singularity_gain=(
            singularity_gain
        ),

        joint_limit_gain=(
            joint_limit_gain
        ),

        obstacle_gain=(
            obstacle_gain
        ),

        joint_limit_activation=(
            joint_limit_activation
        ),

        safety_margin=(
            safety_margin
        ),

        obstacle_safety_distance=(
            obstacle_safety_distance
        ),

        obstacle_influence_distance=(
            obstacle_influence_distance
        ),

        tolerance=tolerance,

        max_iterations=(
            max_iterations
        ),

        sigma_threshold=(
            sigma_threshold
        ),

        lambda_min=lambda_min,

        lambda_max=lambda_max
    )
)


# =====================================================
# CLEARANCE HISTORY
# =====================================================

def calculate_clearance_history(
    result
):

    history = []

    for q_state, d6_state in zip(
        result[
            "q_history"
        ],
        result[
            "d6_history"
        ]
    ):

        history.append(
            calculate_clearance(
                q_state,
                float(d6_state),
                obstacle_center
            )
        )

    return np.asarray(
        history,
        dtype=float
    )


baseline_clearance = (
    calculate_clearance_history(
        baseline_result
    )
)

obstacle_clearance = (
    calculate_clearance_history(
        obstacle_result
    )
)


# =====================================================
# ACTIVATION HISTORY
# =====================================================

baseline_activation = np.asarray([
    obstacle_activation(
        clearance=value,
        safety_distance=(
            obstacle_safety_distance
        ),
        influence_distance=(
            obstacle_influence_distance
        )
    )
    for value in baseline_clearance
])


obstacle_activation_history = np.asarray([
    obstacle_activation(
        clearance=value,
        safety_distance=(
            obstacle_safety_distance
        ),
        influence_distance=(
            obstacle_influence_distance
        )
    )
    for value in obstacle_clearance
])


# =====================================================
# MINIMUM CLEARANCE INDICES
# =====================================================

baseline_min_index = int(
    np.argmin(
        baseline_clearance
    )
)

obstacle_min_index = int(
    np.argmin(
        obstacle_clearance
    )
)


baseline_min_clearance = float(
    baseline_clearance[
        baseline_min_index
    ]
)

obstacle_min_clearance = float(
    obstacle_clearance[
        obstacle_min_index
    ]
)


# =====================================================
# RESULTS DIRECTORY
# =====================================================

results_dir = (
    PROJECT_ROOT
    / "results"
)

results_dir.mkdir(
    exist_ok=True
)


# =====================================================
# PLOT 1 — OBSTACLE CLEARANCE
# =====================================================

fig1, ax1 = plt.subplots(
    figsize=(10, 6)
)

ax1.plot(
    np.arange(
        len(
            baseline_clearance
        )
    ),
    baseline_clearance,
    linewidth=2,
    label="Multi-Objective IK"
)

ax1.plot(
    np.arange(
        len(
            obstacle_clearance
        )
    ),
    obstacle_clearance,
    linewidth=2,
    label="Obstacle-Aware IK"
)

ax1.axhline(
    obstacle_safety_distance,
    linestyle="--",
    linewidth=2,
    label="Safety Distance"
)

ax1.axhline(
    obstacle_influence_distance,
    linestyle=":",
    linewidth=2,
    label="Influence Distance"
)

ax1.set_xlabel(
    "Iteration"
)

ax1.set_ylabel(
    "Minimum Robot-Obstacle Clearance [m]"
)

ax1.set_title(
    "Obstacle Clearance Comparison"
)

ax1.grid(True)

ax1.legend()

fig1.tight_layout()

fig1.savefig(
    results_dir
    / "obstacle_aware_clearance_comparison.png",
    dpi=300,
    bbox_inches="tight"
)


# =====================================================
# PLOT 2 — CARTESIAN ERROR
# =====================================================

fig2, ax2 = plt.subplots(
    figsize=(10, 6)
)

ax2.semilogy(
    np.arange(
        len(
            baseline_result[
                "error_history"
            ]
        )
    ),
    baseline_result[
        "error_history"
    ],
    linewidth=2,
    label="Multi-Objective IK"
)

ax2.semilogy(
    np.arange(
        len(
            obstacle_result[
                "error_history"
            ]
        )
    ),
    obstacle_result[
        "error_history"
    ],
    linewidth=2,
    label="Obstacle-Aware IK"
)

ax2.set_xlabel(
    "Iteration"
)

ax2.set_ylabel(
    "Cartesian Position Error [m]"
)

ax2.set_title(
    "Cartesian Error Comparison"
)

ax2.grid(True)

ax2.legend()

fig2.tight_layout()

fig2.savefig(
    results_dir
    / "obstacle_aware_error_comparison.png",
    dpi=300,
    bbox_inches="tight"
)


# =====================================================
# PLOT 3 — OBSTACLE ACTIVATION
# =====================================================

fig3, ax3 = plt.subplots(
    figsize=(10, 6)
)

ax3.plot(
    np.arange(
        len(
            baseline_activation
        )
    ),
    baseline_activation,
    linewidth=2,
    label="Baseline Risk Activation"
)

ax3.plot(
    np.arange(
        len(
            obstacle_activation_history
        )
    ),
    obstacle_activation_history,
    linewidth=2,
    label="Obstacle-Aware Activation"
)

ax3.set_xlabel(
    "Iteration"
)

ax3.set_ylabel(
    "Obstacle Activation"
)

ax3.set_ylim(
    -0.05,
    1.05
)

ax3.set_title(
    "Obstacle Avoidance Activation"
)

ax3.grid(True)

ax3.legend()

fig3.tight_layout()

fig3.savefig(
    results_dir
    / "obstacle_aware_activation.png",
    dpi=300,
    bbox_inches="tight"
)


# =====================================================
# PLOT 4 — MINIMUM SINGULAR VALUE
# =====================================================

fig4, ax4 = plt.subplots(
    figsize=(10, 6)
)

ax4.plot(
    baseline_result[
        "sigma_min_history"
    ],
    linewidth=2,
    label="Multi-Objective IK"
)

ax4.plot(
    obstacle_result[
        "sigma_min_history"
    ],
    linewidth=2,
    label="Obstacle-Aware IK"
)

ax4.set_xlabel(
    "Iteration"
)

ax4.set_ylabel(
    "Minimum Singular Value"
)

ax4.set_title(
    "Singularity Metric Comparison"
)

ax4.grid(True)

ax4.legend()

fig4.tight_layout()

fig4.savefig(
    results_dir
    / "obstacle_aware_sigma_min.png",
    dpi=300,
    bbox_inches="tight"
)


# =====================================================
# PLOT 5 — JOINT-LIMIT MARGIN
# =====================================================

fig5, ax5 = plt.subplots(
    figsize=(10, 6)
)

ax5.plot(
    baseline_result[
        "joint_limit_margin_history"
    ],
    linewidth=2,
    label="Multi-Objective IK"
)

ax5.plot(
    obstacle_result[
        "joint_limit_margin_history"
    ],
    linewidth=2,
    label="Obstacle-Aware IK"
)

ax5.set_xlabel(
    "Iteration"
)

ax5.set_ylabel(
    "Minimum Normalized Joint-Limit Margin"
)

ax5.set_title(
    "Joint-Limit Margin Comparison"
)

ax5.grid(True)

ax5.legend()

fig5.tight_layout()

fig5.savefig(
    results_dir
    / "obstacle_aware_joint_limit_margin.png",
    dpi=300,
    bbox_inches="tight"
)


# =====================================================
# PLOT 6 — 3D CARTESIAN PATH
# =====================================================

fig6 = plt.figure(
    figsize=(10, 8)
)

ax6 = fig6.add_subplot(
    111,
    projection="3d"
)


baseline_positions = (
    baseline_result[
        "position_history"
    ]
)

obstacle_positions = (
    obstacle_result[
        "position_history"
    ]
)


ax6.plot(
    baseline_positions[:, 0],
    baseline_positions[:, 1],
    baseline_positions[:, 2],
    linewidth=2,
    label="Multi-Objective IK"
)

ax6.plot(
    obstacle_positions[:, 0],
    obstacle_positions[:, 1],
    obstacle_positions[:, 2],
    linewidth=2,
    label="Obstacle-Aware IK"
)


# =====================================================
# START AND TARGET
# =====================================================

start_position = (
    baseline_positions[0]
)

ax6.scatter(
    start_position[0],
    start_position[1],
    start_position[2],
    s=90,
    marker="o",
    label="Start"
)

ax6.scatter(
    target_position[0],
    target_position[1],
    target_position[2],
    s=140,
    marker="X",
    label="Target"
)


# =====================================================
# OBSTACLE SPHERE
# =====================================================

u = np.linspace(
    0.0,
    2.0 * np.pi,
    60
)

v = np.linspace(
    0.0,
    np.pi,
    40
)


sphere_x = (
    obstacle_center[0]
    + obstacle_radius
    * np.outer(
        np.cos(u),
        np.sin(v)
    )
)

sphere_y = (
    obstacle_center[1]
    + obstacle_radius
    * np.outer(
        np.sin(u),
        np.sin(v)
    )
)

sphere_z = (
    obstacle_center[2]
    + obstacle_radius
    * np.outer(
        np.ones_like(u),
        np.cos(v)
    )
)


ax6.plot_surface(
    sphere_x,
    sphere_y,
    sphere_z,
    alpha=0.35
)


# =====================================================
# ROBOT AT BASELINE MINIMUM CLEARANCE
# =====================================================

baseline_robot = (
    get_joint_positions(
        baseline_result[
            "q_history"
        ][
            baseline_min_index
        ],
        float(
            baseline_result[
                "d6_history"
            ][
                baseline_min_index
            ]
        )
    )
)

ax6.plot(
    baseline_robot[:, 0],
    baseline_robot[:, 1],
    baseline_robot[:, 2],
    "--",
    linewidth=1.5,
    label="Baseline Robot at Min Clearance"
)


# =====================================================
# ROBOT AT OBSTACLE-AWARE MINIMUM CLEARANCE
# =====================================================

obstacle_robot = (
    get_joint_positions(
        obstacle_result[
            "q_history"
        ][
            obstacle_min_index
        ],
        float(
            obstacle_result[
                "d6_history"
            ][
                obstacle_min_index
            ]
        )
    )
)

ax6.plot(
    obstacle_robot[:, 0],
    obstacle_robot[:, 1],
    obstacle_robot[:, 2],
    "-.",
    linewidth=1.5,
    label="Obstacle-Aware Robot at Min Clearance"
)


ax6.set_xlabel(
    "X [m]"
)

ax6.set_ylabel(
    "Y [m]"
)

ax6.set_zlabel(
    "Z [m]"
)

ax6.set_title(
    "Obstacle-Aware Multi-Objective IK"
)

ax6.grid(True)

ax6.legend()

fig6.tight_layout()

fig6.savefig(
    results_dir
    / "obstacle_aware_path_comparison.png",
    dpi=300,
    bbox_inches="tight"
)


# =====================================================
# TERMINAL SUMMARY
# =====================================================

print("=" * 78)

print(
    "OBSTACLE-AWARE MULTI-OBJECTIVE IK COMPARISON"
)

print("=" * 78)


print(
    "\nGenerated obstacle center [m]:"
)

print(
    obstacle_center
)

print(
    "\nObstacle radius: "
    f"{obstacle_radius:.4f} m"
)

print(
    "Link radius: "
    f"{link_radius:.4f} m"
)

print(
    "Safety distance: "
    f"{obstacle_safety_distance:.4f} m"
)

print(
    "Influence distance: "
    f"{obstacle_influence_distance:.4f} m"
)


print(
    "\nObstacle generated near baseline "
    f"iteration: "
    f"{obstacle_design['state_index']}"
)

print(
    "Reference link number: "
    f"{obstacle_design['link_index'] + 1}"
)


print(
    "\n"
    + "-" * 78
)

print(
    "BASELINE MULTI-OBJECTIVE IK"
)

print(
    "-" * 78
)

print(
    "Converged: "
    f"{baseline_result['converged']}"
)

print(
    "Iterations: "
    f"{baseline_result['iterations']}"
)

print(
    "Final Cartesian error: "
    f"{baseline_result['final_error']:.8e} m"
)

print(
    "Minimum obstacle clearance: "
    f"{baseline_min_clearance:.8f} m"
)

print(
    "Maintained safety distance: "
    f"{baseline_min_clearance >= obstacle_safety_distance}"
)


print(
    "\n"
    + "-" * 78
)

print(
    "OBSTACLE-AWARE MULTI-OBJECTIVE IK"
)

print(
    "-" * 78
)

print(
    "Converged: "
    f"{obstacle_result['converged']}"
)

print(
    "Iterations: "
    f"{obstacle_result['iterations']}"
)

print(
    "Final Cartesian error: "
    f"{obstacle_result['final_error']:.8e} m"
)

print(
    "Minimum obstacle clearance: "
    f"{obstacle_min_clearance:.8f} m"
)

print(
    "Final obstacle clearance: "
    f"{obstacle_result['final_obstacle_clearance']:.8f} m"
)

print(
    "Maintained safety distance: "
    f"{obstacle_min_clearance >= obstacle_safety_distance}"
)

print(
    "Collision at final state: "
    f"{obstacle_result['collision']}"
)

print(
    "Final sigma_min: "
    f"{obstacle_result['final_sigma_min']:.8e}"
)

print(
    "Final joint-limit margin: "
    f"{obstacle_result['final_joint_limit_margin']:.8f}"
)


if len(
    obstacle_result[
        "obstacle_activity_history"
    ]
) > 0:

    print(
        "Maximum obstacle activity: "
        f"{np.max(obstacle_result['obstacle_activity_history']):.8e}"
    )


print(
    "\nSaved figures:"
)

print(
    "results/"
    "obstacle_aware_clearance_comparison.png"
)

print(
    "results/"
    "obstacle_aware_error_comparison.png"
)

print(
    "results/"
    "obstacle_aware_activation.png"
)

print(
    "results/"
    "obstacle_aware_sigma_min.png"
)

print(
    "results/"
    "obstacle_aware_joint_limit_margin.png"
)

print(
    "results/"
    "obstacle_aware_path_comparison.png"
)


plt.show()
