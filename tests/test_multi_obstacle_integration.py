import numpy as np

from src.kinematics import (
    get_end_effector_pose,
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
# FIXED REGRESSION SCENARIO
# =====================================================

Q_INITIAL = np.radians([
    -90.0,
    0.0,
    30.0,
    -15.0,
    0.0
])

D6_INITIAL = 0.20

Q_TARGET_REFERENCE = np.radians([
    -70.0,
    10.0,
    30.0,
    -20.0,
    15.0
])

D6_TARGET_REFERENCE = 0.22


_, TARGET_POSITION, _ = get_end_effector_pose(
    Q_TARGET_REFERENCE,
    D6_TARGET_REFERENCE
)


# =====================================================
# CONTROLLER PARAMETERS
# =====================================================

STEP_SIZE = 0.5

SINGULARITY_GAIN = 0.005

JOINT_LIMIT_GAIN = 0.005

JOINT_LIMIT_ACTIVATION = 0.80

SAFETY_MARGIN = 0.05

SIGMA_THRESHOLD = 0.10

LAMBDA_MIN = 1e-4

LAMBDA_MAX = 0.20

TOLERANCE = 1e-4

MAX_ITERATIONS = 1000


# =====================================================
# OBSTACLE PARAMETERS
# =====================================================

OBSTACLE_RADIUS = 0.10

LINK_RADIUS = 0.05

OBSTACLE_SAFETY_DISTANCE = 0.10

OBSTACLE_INFLUENCE_DISTANCE = 0.25

OBSTACLE_GAIN = 0.03


# =====================================================
# FIXED TWO-OBSTACLE SCENARIO
# =====================================================

OBSTACLES = [
    {
        "name": "Obstacle 1",
        "center": np.array([
            0.34824360,
            -1.91517415,
            1.04807986
        ]),
        "radius": OBSTACLE_RADIUS,
    },
    {
        "name": "Obstacle 2",
        "center": np.array([
            0.38218904,
            -1.84466575,
            1.22558445
        ]),
        "radius": OBSTACLE_RADIUS,
    },
]


# =====================================================
# BASELINE CLEARANCE HELPER
# =====================================================

def compute_baseline_clearance_history(
    q_history,
    d6_history
):

    clearances = []

    for q_state, d6_state in zip(
        q_history,
        d6_history
    ):

        result = minimum_multi_obstacle_clearance(
            q=q_state,
            d6=float(d6_state),
            obstacles=OBSTACLES,
            link_radius=LINK_RADIUS
        )

        clearances.append(
            float(result["clearance"])
        )

    return np.asarray(
        clearances,
        dtype=float
    )


# =====================================================
# INTEGRATION / REGRESSION TEST
# =====================================================

def test_multi_obstacle_safety_regression():

    baseline_result = solve_multi_objective_ik(
        target_position=TARGET_POSITION,

        q_initial=Q_INITIAL,
        d6_initial=D6_INITIAL,

        step_size=STEP_SIZE,

        singularity_gain=SINGULARITY_GAIN,
        joint_limit_gain=JOINT_LIMIT_GAIN,

        joint_limit_activation=JOINT_LIMIT_ACTIVATION,
        safety_margin=SAFETY_MARGIN,

        tolerance=TOLERANCE,
        max_iterations=MAX_ITERATIONS,

        sigma_threshold=SIGMA_THRESHOLD,
        lambda_min=LAMBDA_MIN,
        lambda_max=LAMBDA_MAX
    )

    baseline_clearance_history = (
        compute_baseline_clearance_history(
            baseline_result["q_history"],
            baseline_result["d6_history"]
        )
    )

    baseline_min_clearance = float(
        np.min(baseline_clearance_history)
    )

    multi_result = solve_multi_obstacle_aware_ik(
        target_position=TARGET_POSITION,

        q_initial=Q_INITIAL,
        d6_initial=D6_INITIAL,

        obstacles=OBSTACLES,
        link_radius=LINK_RADIUS,

        step_size=STEP_SIZE,

        singularity_gain=SINGULARITY_GAIN,
        joint_limit_gain=JOINT_LIMIT_GAIN,
        obstacle_gain=OBSTACLE_GAIN,

        joint_limit_activation=JOINT_LIMIT_ACTIVATION,
        safety_margin=SAFETY_MARGIN,

        obstacle_safety_distance=OBSTACLE_SAFETY_DISTANCE,
        obstacle_influence_distance=OBSTACLE_INFLUENCE_DISTANCE,

        use_safety_filter=True,
        safety_filter_buffer=0.005,
        safety_filter_samples=20,
        safety_filter_projection_passes=5,
        safety_filter_backtracking_factor=0.5,
        safety_filter_max_backtracking_steps=12,

        tolerance=TOLERANCE,
        max_iterations=MAX_ITERATIONS,

        sigma_threshold=SIGMA_THRESHOLD,
        lambda_min=LAMBDA_MIN,
        lambda_max=LAMBDA_MAX
    )

    active_counts = np.asarray(
        multi_result["active_obstacle_count_history"],
        dtype=int
    )

    corrected_history = np.asarray(
        multi_result["safety_filter_corrected_history"],
        dtype=bool
    )

    scale_history = np.asarray(
        multi_result["safety_filter_scale_history"],
        dtype=float
    )

    # -------------------------------------------------
    # BASELINE SHOULD VIOLATE SAFETY
    # -------------------------------------------------

    assert baseline_result["converged"] is True
    assert baseline_result["final_error"] <= TOLERANCE
    assert baseline_min_clearance < OBSTACLE_SAFETY_DISTANCE

    # -------------------------------------------------
    # MULTI-OBSTACLE CONTROLLER SHOULD SUCCEED SAFELY
    # -------------------------------------------------

    assert multi_result["converged"] is True
    assert multi_result["final_error"] <= TOLERANCE

    assert multi_result["trajectory_safety_satisfied"] is True
    assert multi_result["minimum_trajectory_clearance"] >= OBSTACLE_SAFETY_DISTANCE
    assert multi_result["final_obstacle_clearance"] >= OBSTACLE_SAFETY_DISTANCE

    assert multi_result["collision"] is False

    # -------------------------------------------------
    # MULTI-OBSTACLE ACTIVITY
    # -------------------------------------------------

    assert len(active_counts) > 0
    assert np.max(active_counts) == 2
    assert 1 in active_counts
    assert 2 in active_counts

    # -------------------------------------------------
    # SAFETY FILTER SHOULD ACTUALLY INTERVENE
    # -------------------------------------------------

    assert len(corrected_history) > 0
    assert np.count_nonzero(corrected_history) > 0

    assert len(scale_history) > 0
    assert np.all(scale_history <= 1.0 + 1e-12)
    assert np.min(scale_history) >= 0.0

    # In the current successful scenario no backtracking
    # is needed; only linear safety correction is active.
    assert np.count_nonzero(scale_history < (1.0 - 1e-12)) == 0
