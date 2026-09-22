import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# =====================================================
# PROJECT PATH
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


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

from src.multi_obstacle_aware_ik import (
    solve_multi_obstacle_aware_ik,
)

from src.multi_obstacle_avoidance import (
    minimum_multi_obstacle_clearance,
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

# Smaller influence region than the original
# experiment.
#
# The safety distance remains 0.10 m.
obstacle_influence_distance = 0.25

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
# SINGLE-OBSTACLE CLEARANCE HELPER
# =====================================================

def calculate_single_clearance(
    q,
    d6,
    obstacle_center
):

    obstacle = [
        {
            "name": "Temporary",

            "center": (
                obstacle_center
            ),

            "radius": (
                obstacle_radius
            ),
        }
    ]

    result = (
        minimum_multi_obstacle_clearance(
            q=q,
            d6=d6,

            obstacles=obstacle,

            link_radius=(
                link_radius
            )
        )
    )

    return float(
        result[
            "clearance"
        ]
    )


# =====================================================
# PERPENDICULAR DIRECTION
# =====================================================

def create_perpendicular_direction(
    segment
):

    segment = np.asarray(
        segment,
        dtype=float
    )

    norm = float(
        np.linalg.norm(
            segment
        )
    )

    if norm <= 1e-12:
        return None

    direction = (
        segment
        / norm
    )

    reference = np.array([
        0.0,
        0.0,
        1.0
    ])

    if abs(
        float(
            direction
            @ reference
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

    perpendicular_norm = float(
        np.linalg.norm(
            perpendicular
        )
    )

    if perpendicular_norm <= 1e-12:
        return None

    return (
        perpendicular
        / perpendicular_norm
    )


# =====================================================
# GENERATE CHALLENGING OBSTACLE CANDIDATES
# =====================================================

def generate_obstacle_candidates():

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

    candidate_indices = np.arange(
        1,
        max(
            2,
            number_of_states - 1
        )
    )

    # Major physical robot links.
    candidate_link_indices = [
        1,
        2,
        3,
    ]

    # Desired baseline clearances.
    desired_clearances = [
        0.02,
        0.03,
        0.04,
        0.05,
    ]

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

            for desired_clearance in (
                desired_clearances
            ):

                centerline_offset = (
                    obstacle_radius
                    + link_radius
                    + desired_clearance
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
                        calculate_single_clearance(
                            q_state,
                            d6_state,
                            center
                        )
                    )

                    initial_clearance = (
                        calculate_single_clearance(
                            q_initial,
                            d6_initial,
                            center
                        )
                    )

                    final_clearance = (
                        calculate_single_clearance(
                            baseline_result[
                                "q"
                            ],
                            baseline_result[
                                "d6"
                            ],
                            center
                        )
                    )

                    # ---------------------------------
                    # VALID CHALLENGING CANDIDATE
                    # ---------------------------------
                    #
                    # During the baseline trajectory:
                    #
                    #     clearance < safety distance
                    #
                    # while the initial and final
                    # configurations remain safe.
                    # ---------------------------------

                    if (
                        current_clearance
                        > 0.005

                        and current_clearance
                        < obstacle_safety_distance

                        and initial_clearance
                        > (
                            obstacle_safety_distance
                            + 0.015
                        )

                        and final_clearance
                        > (
                            obstacle_safety_distance
                            + 0.015
                        )
                    ):

                        candidates.append({
                            "center": (
                                center.copy()
                            ),

                            "state_index": (
                                int(
                                    state_index
                                )
                            ),

                            "link_index": (
                                int(
                                    link_index
                                )
                            ),

                            "clearance": float(
                                current_clearance
                            ),

                            "initial_clearance": float(
                                initial_clearance
                            ),

                            "final_clearance": float(
                                final_clearance
                            ),
                        })

    return candidates


# =====================================================
# SELECT TWO DISTINCT CHALLENGING OBSTACLES
# =====================================================

def select_two_obstacles():

    candidates = (
        generate_obstacle_candidates()
    )

    if len(candidates) < 2:

        raise RuntimeError(
            "Could not generate at least two "
            "valid obstacle candidates."
        )

    number_of_states = len(
        baseline_result[
            "q_history"
        ]
    )

    trajectory_span = max(
        number_of_states - 1,
        1
    )

    # =================================================
    # PREFERRED TEMPORAL LOCATIONS
    # =================================================
    #
    # Obstacle 1:
    # around the first third of the trajectory.
    #
    # Obstacle 2:
    # around the last third of the trajectory.
    #
    # These are preferences rather than hard
    # requirements.
    # =================================================

    first_target_index = (
        0.30
        * trajectory_span
    )

    second_target_index = (
        0.72
        * trajectory_span
    )

    # =================================================
    # FIRST OBSTACLE
    # =================================================

    first_candidates = [
        candidate
        for candidate in candidates
        if (
            candidate[
                "state_index"
            ]
            <= (
                0.50
                * trajectory_span
            )
        )
    ]

    if len(
        first_candidates
    ) == 0:

        first_candidates = (
            candidates
        )

    first = min(
        first_candidates,

        key=lambda item: (

            abs(
                item[
                    "state_index"
                ]
                - first_target_index
            )
            / trajectory_span

            + 3.0
            * abs(
                item[
                    "clearance"
                ]
                - 0.03
            )
        )
    )

    # =================================================
    # SECOND OBSTACLE CANDIDATES
    # =================================================

    second_candidates = []

    for candidate in candidates:

        center_separation = float(
            np.linalg.norm(
                candidate[
                    "center"
                ]
                - first[
                    "center"
                ]
            )
        )

        index_separation = abs(
            candidate[
                "state_index"
            ]
            - first[
                "state_index"
            ]
        )

        # Reject an effectively identical candidate.
        if (
            center_separation
            <= 1e-3
            and index_separation == 0
        ):
            continue

        second_candidates.append({
            **candidate,

            "_center_separation": (
                center_separation
            ),

            "_index_separation": (
                index_separation
            ),
        })

    if len(
        second_candidates
    ) == 0:

        raise RuntimeError(
            "Could not generate two distinct "
            "obstacle candidates."
        )

    # =================================================
    # SECOND OBSTACLE SCORE
    # =================================================
    #
    # We prefer:
    #
    # 1. A candidate near the final third.
    # 2. A challenging baseline clearance.
    # 3. Large temporal separation.
    # 4. Large Cartesian separation.
    #
    # No hard 0.40 m requirement is imposed.
    # =================================================

    def second_score(
        item
    ):

        temporal_target_error = (
            abs(
                item[
                    "state_index"
                ]
                - second_target_index
            )
            / trajectory_span
        )

        clearance_error = abs(
            item[
                "clearance"
            ]
            - 0.03
        )

        temporal_separation = (
            item[
                "_index_separation"
            ]
            / trajectory_span
        )

        spatial_separation = (
            item[
                "_center_separation"
            ]
        )

        return (
            temporal_target_error

            + 3.0
            * clearance_error

            - 1.5
            * temporal_separation

            - 2.0
            * spatial_separation
        )

    second = min(
        second_candidates,
        key=second_score
    )

    # Remove temporary scoring fields.
    second = {
        key: value
        for key, value
        in second.items()
        if not key.startswith("_")
    }

    # =================================================
    # ORDER BY TRAJECTORY LOCATION
    # =================================================

    if (
        second[
            "state_index"
        ]
        < first[
            "state_index"
        ]
    ):

        first, second = (
            second,
            first
        )

    return (
        first,
        second
    )


# =====================================================
# SELECT OBSTACLES
# =====================================================

first_design, second_design = (
    select_two_obstacles()
)


# =====================================================
# FINAL OBSTACLE SET
# =====================================================

obstacles = [
    {
        "name": "Obstacle 1",

        "center": (
            first_design[
                "center"
            ]
        ),

        "radius": (
            obstacle_radius
        ),
    },

    {
        "name": "Obstacle 2",

        "center": (
            second_design[
                "center"
            ]
        ),

        "radius": (
            obstacle_radius
        ),
    },
]


# =====================================================
# OBSTACLE CENTER SEPARATION
# =====================================================

obstacle_center_separation = float(
    np.linalg.norm(
        np.asarray(
            obstacles[0][
                "center"
            ],
            dtype=float
        )
        -
        np.asarray(
            obstacles[1][
                "center"
            ],
            dtype=float
        )
    )
)


# =====================================================
# TEMPORAL SEPARATION
# =====================================================

obstacle_iteration_separation = abs(
    second_design[
        "state_index"
    ]
    - first_design[
        "state_index"
    ]
)


# =====================================================
# MULTI-OBSTACLE SAFETY-CONSTRAINED IK
# =====================================================

multi_result = (
    solve_multi_obstacle_aware_ik(
        target_position=(
            target_position
        ),

        q_initial=q_initial,

        d6_initial=d6_initial,

        obstacles=obstacles,

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

        use_safety_filter=True,

        safety_filter_buffer=0.005,

        safety_filter_samples=20,

        safety_filter_projection_passes=5,

        safety_filter_backtracking_factor=0.5,

        safety_filter_max_backtracking_steps=12,

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
# BASELINE GLOBAL CLEARANCE HISTORY
# =====================================================

baseline_clearance_history = []

for q_state, d6_state in zip(
    baseline_result[
        "q_history"
    ],
    baseline_result[
        "d6_history"
    ]
):

    result = (
        minimum_multi_obstacle_clearance(
            q=q_state,

            d6=float(
                d6_state
            ),

            obstacles=obstacles,

            link_radius=(
                link_radius
            )
        )
    )

    baseline_clearance_history.append(
        float(
            result[
                "clearance"
            ]
        )
    )


baseline_clearance_history = np.asarray(
    baseline_clearance_history,
    dtype=float
)


baseline_min_clearance = float(
    np.min(
        baseline_clearance_history
    )
)


multi_clearance_history = (
    multi_result[
        "minimum_obstacle_clearance_history"
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
# PLOT 1 — GLOBAL CLEARANCE
# =====================================================

fig1, ax1 = plt.subplots(
    figsize=(10, 6)
)

ax1.plot(
    baseline_clearance_history,

    linewidth=2,

    label=(
        "Baseline Multi-Objective IK"
    )
)

ax1.plot(
    multi_clearance_history,

    linewidth=2,

    label=(
        "Multi-Obstacle Safety IK"
    )
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

    linewidth=1.5,

    label="Influence Distance"
)

ax1.set_xlabel(
    "Iteration"
)

ax1.set_ylabel(
    "Global Minimum Clearance [m]"
)

ax1.set_title(
    "Multi-Obstacle Clearance Comparison"
)

ax1.grid(
    True
)

ax1.legend()

fig1.tight_layout()

fig1.savefig(
    results_dir
    / "multi_obstacle_clearance_comparison.png",

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
    baseline_result[
        "error_history"
    ],

    linewidth=2,

    label=(
        "Baseline Multi-Objective IK"
    )
)

ax2.semilogy(
    multi_result[
        "error_history"
    ],

    linewidth=2,

    label=(
        "Multi-Obstacle Safety IK"
    )
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

ax2.grid(
    True
)

ax2.legend()

fig2.tight_layout()

fig2.savefig(
    results_dir
    / "multi_obstacle_error_comparison.png",

    dpi=300,

    bbox_inches="tight"
)


# =====================================================
# PLOT 3 — ACTIVE OBSTACLES
# =====================================================

fig3, ax3 = plt.subplots(
    figsize=(10, 6)
)

active_history = np.asarray(
    multi_result[
        "active_obstacle_count_history"
    ],
    dtype=int
)

ax3.step(
    np.arange(
        len(
            active_history
        )
    ),

    active_history,

    where="post",

    linewidth=2
)

ax3.set_xlabel(
    "Iteration"
)

ax3.set_ylabel(
    "Active Obstacle Count"
)

ax3.set_title(
    "Number of Active Obstacles"
)

ax3.set_yticks([
    0,
    1,
    2,
])

ax3.set_ylim(
    -0.15,
    2.15
)

ax3.grid(
    True
)

fig3.tight_layout()

fig3.savefig(
    results_dir
    / "multi_obstacle_active_count.png",

    dpi=300,

    bbox_inches="tight"
)


# =====================================================
# PLOT 4 — SAFETY FILTER ACTIVITY
# =====================================================

fig4, ax4 = plt.subplots(
    figsize=(10, 6)
)

safety_scale_history = np.asarray(
    multi_result[
        "safety_filter_scale_history"
    ],
    dtype=float
)

corrected_history = np.asarray(
    multi_result[
        "safety_filter_corrected_history"
    ],
    dtype=bool
)

steps = np.arange(
    len(
        safety_scale_history
    )
)

ax4.plot(
    steps,

    safety_scale_history,

    linewidth=2,

    label="Backtracking Scale"
)


# Mark steps where the linear safety filter
# modified the nominal motion.
corrected_indices = np.flatnonzero(
    corrected_history
)

if len(
    corrected_indices
) > 0:

    ax4.scatter(
        corrected_indices,

        safety_scale_history[
            corrected_indices
        ],

        s=60,

        marker="x",

        label=(
            "Safety-Corrected Step"
        )
    )


ax4.set_xlabel(
    "IK Step"
)

ax4.set_ylabel(
    "Safety Filter Scale"
)

ax4.set_title(
    "Multi-Obstacle Safety Filter Activity"
)

ax4.grid(
    True
)

ax4.legend()

fig4.tight_layout()

fig4.savefig(
    results_dir
    / "multi_obstacle_safety_filter_scale.png",

    dpi=300,

    bbox_inches="tight"
)


# =====================================================
# PLOT 5 — SIGMA MIN
# =====================================================

fig5, ax5 = plt.subplots(
    figsize=(10, 6)
)

ax5.plot(
    baseline_result[
        "sigma_min_history"
    ],

    linewidth=2,

    label=(
        "Baseline Multi-Objective IK"
    )
)

ax5.plot(
    multi_result[
        "sigma_min_history"
    ],

    linewidth=2,

    label=(
        "Multi-Obstacle Safety IK"
    )
)

ax5.set_xlabel(
    "Iteration"
)

ax5.set_ylabel(
    "Minimum Singular Value"
)

ax5.set_title(
    "Singularity Metric Comparison"
)

ax5.grid(
    True
)

ax5.legend()

fig5.tight_layout()

fig5.savefig(
    results_dir
    / "multi_obstacle_sigma_min.png",

    dpi=300,

    bbox_inches="tight"
)


# =====================================================
# PLOT 6 — 3D CARTESIAN PATH
# =====================================================

fig6 = plt.figure(
    figsize=(11, 9)
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

multi_positions = (
    multi_result[
        "position_history"
    ]
)


ax6.plot(
    baseline_positions[:, 0],

    baseline_positions[:, 1],

    baseline_positions[:, 2],

    linewidth=2,

    label="Baseline"
)


ax6.plot(
    multi_positions[:, 0],

    multi_positions[:, 1],

    multi_positions[:, 2],

    linewidth=2,

    label=(
        "Multi-Obstacle Safety IK"
    )
)


# =====================================================
# START POSITION
# =====================================================

ax6.scatter(
    baseline_positions[
        0,
        0
    ],

    baseline_positions[
        0,
        1
    ],

    baseline_positions[
        0,
        2
    ],

    s=90,

    marker="o",

    label="Start"
)


# =====================================================
# TARGET POSITION
# =====================================================

ax6.scatter(
    target_position[0],

    target_position[1],

    target_position[2],

    s=140,

    marker="X",

    label="Target"
)


# =====================================================
# DRAW OBSTACLE SPHERES
# =====================================================

u = np.linspace(
    0.0,
    2.0 * np.pi,
    50
)

v = np.linspace(
    0.0,
    np.pi,
    30
)


for obstacle_index, obstacle in enumerate(
    obstacles
):

    center = np.asarray(
        obstacle[
            "center"
        ],
        dtype=float
    )

    radius = float(
        obstacle[
            "radius"
        ]
    )

    sphere_x = (
        center[0]
        + radius
        * np.outer(
            np.cos(u),
            np.sin(v)
        )
    )

    sphere_y = (
        center[1]
        + radius
        * np.outer(
            np.sin(u),
            np.sin(v)
        )
    )

    sphere_z = (
        center[2]
        + radius
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

    # Slightly offset labels vertically.
    label_offset = np.array([
        0.0,
        0.0,
        0.13
        + 0.03
        * obstacle_index
    ])

    label_position = (
        center
        + label_offset
    )

    ax6.text(
        label_position[0],

        label_position[1],

        label_position[2],

        obstacle[
            "name"
        ]
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
    "Multi-Obstacle Safety-Constrained IK"
)

ax6.grid(
    True
)

ax6.legend()

fig6.tight_layout()

fig6.savefig(
    results_dir
    / "multi_obstacle_path_comparison.png",

    dpi=300,

    bbox_inches="tight"
)


# =====================================================
# TERMINAL SUMMARY
# =====================================================

print(
    "=" * 78
)

print(
    "MULTI-OBSTACLE SAFETY-CONSTRAINED IK COMPARISON"
)

print(
    "=" * 78
)


# =====================================================
# OBSTACLE INFORMATION
# =====================================================

for index, design in enumerate(
    [
        first_design,
        second_design
    ],
    start=1
):

    print(
        f"\nObstacle {index} center [m]:"
    )

    print(
        design[
            "center"
        ]
    )

    print(
        "Generated near baseline iteration: "
        f"{design['state_index']}"
    )

    print(
        "Reference link number: "
        f"{design['link_index'] + 1}"
    )

    print(
        "Baseline design clearance: "
        f"{design['clearance']:.8f} m"
    )

    print(
        "Initial clearance: "
        f"{design['initial_clearance']:.8f} m"
    )

    print(
        "Baseline final clearance: "
        f"{design['final_clearance']:.8f} m"
    )


print(
    "\nObstacle-center separation: "
    f"{obstacle_center_separation:.8f} m"
)

print(
    "Obstacle baseline-iteration separation: "
    f"{obstacle_iteration_separation}"
)

print(
    "Obstacle radius: "
    f"{obstacle_radius:.4f} m"
)

print(
    "Link radius: "
    f"{link_radius:.4f} m"
)

print(
    "Required safety distance: "
    f"{obstacle_safety_distance:.4f} m"
)

print(
    "Influence distance: "
    f"{obstacle_influence_distance:.4f} m"
)


# =====================================================
# BASELINE SUMMARY
# =====================================================

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
    "Minimum global obstacle clearance: "
    f"{baseline_min_clearance:.8f} m"
)

print(
    "Maintained safety distance: "
    f"{baseline_min_clearance >= obstacle_safety_distance}"
)


# =====================================================
# MULTI-OBSTACLE SUMMARY
# =====================================================

print(
    "\n"
    + "-" * 78
)

print(
    "MULTI-OBSTACLE SAFETY-CONSTRAINED IK"
)

print(
    "-" * 78
)

print(
    "Converged: "
    f"{multi_result['converged']}"
)

print(
    "Iterations: "
    f"{multi_result['iterations']}"
)

print(
    "Final Cartesian error: "
    f"{multi_result['final_error']:.8e} m"
)

print(
    "Minimum sampled trajectory clearance: "
    f"{multi_result['minimum_trajectory_clearance']:.8f} m"
)

print(
    "Final obstacle clearance: "
    f"{multi_result['final_obstacle_clearance']:.8f} m"
)

print(
    "Maintained safety distance: "
    f"{multi_result['trajectory_safety_satisfied']}"
)

print(
    "Collision: "
    f"{multi_result['collision']}"
)

print(
    "Final sigma_min: "
    f"{multi_result['final_sigma_min']:.8e}"
)

print(
    "Final joint-limit margin: "
    f"{multi_result['final_joint_limit_margin']:.8f}"
)


# =====================================================
# OBSTACLE ACTIVATION STATISTICS
# =====================================================

if len(
    active_history
) > 0:

    print(
        "Minimum simultaneously active obstacles: "
        f"{np.min(active_history)}"
    )

    print(
        "Maximum simultaneously active obstacles: "
        f"{np.max(active_history)}"
    )

    unique_active_counts = np.unique(
        active_history
    )

    print(
        "Observed active-obstacle counts: "
        f"{unique_active_counts.tolist()}"
    )


# =====================================================
# SAFETY FILTER STATISTICS
# =====================================================

if len(
    corrected_history
) > 0:

    corrected_steps = int(
        np.count_nonzero(
            corrected_history
        )
    )

    print(
        "Safety-filter corrected steps: "
        f"{corrected_steps}"
    )

    print(
        "Total controlled IK steps: "
        f"{len(corrected_history)}"
    )

    print(
        "Safety-filter correction ratio: "
        f"{corrected_steps / len(corrected_history):.4f}"
    )


if len(
    safety_scale_history
) > 0:

    backtracked_steps = int(
        np.count_nonzero(
            safety_scale_history
            < (
                1.0
                - 1e-12
            )
        )
    )

    print(
        "Backtracked steps: "
        f"{backtracked_steps}"
    )

    print(
        "Minimum safety-filter scale: "
        f"{np.min(safety_scale_history):.8f}"
    )


# =====================================================
# SAVED FIGURES
# =====================================================

print(
    "\nSaved figures:"
)

print(
    "results/"
    "multi_obstacle_clearance_comparison.png"
)

print(
    "results/"
    "multi_obstacle_error_comparison.png"
)

print(
    "results/"
    "multi_obstacle_active_count.png"
)

print(
    "results/"
    "multi_obstacle_safety_filter_scale.png"
)

print(
    "results/"
    "multi_obstacle_sigma_min.png"
)

print(
    "results/"
    "multi_obstacle_path_comparison.png"
)


plt.show()
