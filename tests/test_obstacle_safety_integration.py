import numpy as np

from src.kinematics import (
    get_end_effector_pose,
)

from src.multi_objective_ik import (
    solve_multi_objective_ik,
)

from src.obstacle_aware_ik import (
    solve_obstacle_aware_ik,
)

from src.obstacle_avoidance import (
    minimum_obstacle_clearance,
)


# =====================================================
# REGRESSION SCENARIO
# =====================================================
#
# This obstacle was obtained from the controlled
# comparison experiment.
#
# In this scenario:
#
# Baseline Multi-Objective IK:
#     violates the obstacle safety distance.
#
# Safety-Constrained Obstacle-Aware IK:
#     reaches the Cartesian target while maintaining
#     the required sampled trajectory clearance.
#
# =====================================================


Q_INITIAL = np.radians([
    -90.0,
    0.0,
    30.0,
    -15.0,
    0.0
])

D6_INITIAL = 0.20


Q_TARGET = np.radians([
    -70.0,
    10.0,
    30.0,
    -20.0,
    15.0
])

D6_TARGET = 0.22


_, TARGET_POSITION, _ = (
    get_end_effector_pose(
        Q_TARGET,
        D6_TARGET
    )
)


OBSTACLE_CENTER = np.array([
    0.39172654,
    -1.84165972,
    1.22558445
])

OBSTACLE_RADIUS = 0.10

LINK_RADIUS = 0.05

SAFETY_DISTANCE = 0.10


# =====================================================
# CLEARANCE HELPER
# =====================================================

def calculate_clearance(
    q,
    d6
):

    return float(
        minimum_obstacle_clearance(
            q=q,
            d6=d6,
            obstacle_center=(
                OBSTACLE_CENTER
            ),
            obstacle_radius=(
                OBSTACLE_RADIUS
            ),
            link_radius=(
                LINK_RADIUS
            )
        )["clearance"]
    )


# =====================================================
# SAFETY-CONSTRAINED INTEGRATION TEST
# =====================================================

def test_safety_constrained_obstacle_aware_ik():

    # -------------------------------------------------
    # BASELINE
    # -------------------------------------------------

    baseline = (
        solve_multi_objective_ik(
            target_position=(
                TARGET_POSITION
            ),

            q_initial=(
                Q_INITIAL
            ),

            d6_initial=(
                D6_INITIAL
            ),

            step_size=0.5,

            singularity_gain=0.005,

            joint_limit_gain=0.005,

            joint_limit_activation=0.80,

            safety_margin=0.05,

            tolerance=1e-4,

            max_iterations=1000,

            sigma_threshold=0.10,

            lambda_min=1e-4,

            lambda_max=0.20
        )
    )

    baseline_clearance_history = []

    for q_state, d6_state in zip(
        baseline[
            "q_history"
        ],
        baseline[
            "d6_history"
        ]
    ):

        baseline_clearance_history.append(
            calculate_clearance(
                q_state,
                float(d6_state)
            )
        )

    baseline_min_clearance = float(
        np.min(
            baseline_clearance_history
        )
    )

    # The regression scenario must remain
    # challenging for the baseline controller.
    assert baseline[
        "converged"
    ] is True

    assert (
        baseline_min_clearance
        < SAFETY_DISTANCE
    )

    # -------------------------------------------------
    # SAFETY-CONSTRAINED CONTROLLER
    # -------------------------------------------------

    result = (
        solve_obstacle_aware_ik(
            target_position=(
                TARGET_POSITION
            ),

            q_initial=(
                Q_INITIAL
            ),

            d6_initial=(
                D6_INITIAL
            ),

            obstacle_center=(
                OBSTACLE_CENTER
            ),

            obstacle_radius=(
                OBSTACLE_RADIUS
            ),

            link_radius=(
                LINK_RADIUS
            ),

            step_size=0.5,

            singularity_gain=0.005,

            joint_limit_gain=0.005,

            obstacle_gain=0.03,

            joint_limit_activation=0.80,

            safety_margin=0.05,

            obstacle_safety_distance=(
                SAFETY_DISTANCE
            ),

            obstacle_influence_distance=0.40,

            use_safety_filter=True,

            safety_filter_buffer=0.005,

            safety_filter_samples=20,

            safety_filter_backtracking_factor=0.5,

            safety_filter_max_backtracking_steps=12,

            tolerance=1e-4,

            max_iterations=1000,

            sigma_threshold=0.10,

            lambda_min=1e-4,

            lambda_max=0.20
        )
    )

    # =================================================
    # TARGET REACHED
    # =================================================

    assert result[
        "converged"
    ] is True

    assert (
        result[
            "final_error"
        ]
        <= 1e-4
    )

    assert np.allclose(
        result[
            "final_position"
        ],
        TARGET_POSITION,
        atol=1e-4
    )

    # =================================================
    # TRAJECTORY SAFETY
    # =================================================

    assert result[
        "trajectory_safety_satisfied"
    ] is True

    assert (
        result[
            "minimum_trajectory_clearance"
        ]
        >= (
            SAFETY_DISTANCE
            - 1e-9
        )
    )

    assert (
        result[
            "final_obstacle_clearance"
        ]
        >= (
            SAFETY_DISTANCE
            - 1e-9
        )
    )

    assert result[
        "collision"
    ] is False

    # =================================================
    # SAFETY FILTER ACTUALLY PARTICIPATED
    # =================================================

    corrected_history = (
        result[
            "safety_filter_corrected_history"
        ]
    )

    assert len(
        corrected_history
    ) > 0

    assert np.any(
        corrected_history
    )

    # Every accepted sampled path must satisfy
    # the requested safety distance.
    path_clearance_history = (
        result[
            "minimum_path_clearance_history"
        ]
    )

    assert len(
        path_clearance_history
    ) > 0

    assert np.all(
        path_clearance_history
        >= (
            SAFETY_DISTANCE
            - 1e-9
        )
    )

    # =================================================
    # NUMERICAL VALIDITY
    # =================================================

    assert np.isfinite(
        result[
            "final_sigma_min"
        ]
    )

    assert (
        result[
            "final_sigma_min"
        ]
        >= 0.0
    )

    assert np.isfinite(
        result[
            "final_joint_limit_margin"
        ]
    )
