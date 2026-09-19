import numpy as np

from src.kinematics import get_end_effector_pose
from src.jacobian import translational_jacobian
from src.adaptive_damping import adaptive_dls_pseudoinverse

from src.inverse_kinematics import (
    Q_MIN,
    Q_MAX,
    D6_MIN,
    D6_MAX,
)

from src.singularity_avoidance import (
    translational_sigma_min,
    sigma_min_gradient,
)

from src.joint_limit_avoidance import (
    joint_limit_avoidance_direction,
    joint_limit_cost,
    joint_limit_margin,
)


# =====================================================
# NORMALIZE GENERALIZED JOINT-SPACE VECTOR
# =====================================================

def normalize_vector(vector):
    """
    Normalize a generalized joint-space vector.

    Returns a zero vector if the norm is too small.
    """

    vector = np.asarray(
        vector,
        dtype=float
    )

    norm = float(
        np.linalg.norm(vector)
    )

    if norm <= 1e-12:
        return np.zeros_like(vector)

    return vector / norm


# =====================================================
# SINGLE MULTI-OBJECTIVE IK STEP
# =====================================================

def multi_objective_ik_step(
    q,
    d6,
    target_position,
    step_size=0.5,
    singularity_gain=0.005,
    joint_limit_gain=0.005,
    joint_limit_activation=0.80,
    sigma_threshold=0.10,
    lambda_min=1e-4,
    lambda_max=0.20,
    max_revolute_step=np.radians(5.0),
    max_prismatic_step=0.02
):
    """
    Perform one multi-objective IK iteration.

    Primary task:
        Cartesian end-effector position tracking.

    Secondary tasks:
        1. Singularity avoidance.
        2. Joint-limit avoidance.

    Adaptive Damped Least Squares is used
    for the primary task.
    """

    q = np.asarray(
        q,
        dtype=float
    ).copy()

    target_position = np.asarray(
        target_position,
        dtype=float
    )

    if q.shape != (5,):
        raise ValueError(
            "q must contain five revolute joints."
        )

    if target_position.shape != (3,):
        raise ValueError(
            "target_position must have shape (3,)."
        )

    if singularity_gain < 0.0:
        raise ValueError(
            "singularity_gain must be non-negative."
        )

    if joint_limit_gain < 0.0:
        raise ValueError(
            "joint_limit_gain must be non-negative."
        )

    # =================================================
    # CURRENT END-EFFECTOR STATE
    # =================================================

    _, current_position, _ = (
        get_end_effector_pose(
            q,
            d6
        )
    )

    error = (
        target_position
        - current_position
    )

    error_norm = float(
        np.linalg.norm(error)
    )

    # =================================================
    # TRANSLATIONAL JACOBIAN
    # =================================================

    J = translational_jacobian(
        q,
        d6
    )

    (
        J_dls,
        damping,
        sigma_min
    ) = adaptive_dls_pseudoinverse(
        J,
        sigma_threshold=sigma_threshold,
        lambda_min=lambda_min,
        lambda_max=lambda_max
    )

    # =================================================
    # PRIMARY TASK
    # CARTESIAN POSITION TRACKING
    # =================================================

    primary_delta = (
        step_size
        * J_dls
        @ error
    )

    # =================================================
    # SECONDARY TASK 1
    # SINGULARITY AVOIDANCE
    # =================================================

    singularity_gradient = (
        sigma_min_gradient(
            q,
            d6
        )
    )

    singularity_direction = (
        normalize_vector(
            singularity_gradient
        )
    )

    singularity_command = (
        singularity_gain
        * singularity_direction
    )

    # =================================================
    # SECONDARY TASK 2
    # JOINT-LIMIT AVOIDANCE
    # =================================================

    limit_direction_raw = (
        joint_limit_avoidance_direction(
            q,
            d6,
            activation_threshold=joint_limit_activation
        )
    )

    limit_direction = normalize_vector(
        limit_direction_raw
    )

    joint_limit_command = (
        joint_limit_gain
        * limit_direction
    )

    # =================================================
    # COMBINED SECONDARY OBJECTIVE
    # =================================================

    secondary_command = (
        singularity_command
        + joint_limit_command
    )

    # Damped task-priority projector
    null_space_projector = (
        np.eye(6)
        - J_dls @ J
    )

    secondary_delta = (
        null_space_projector
        @ secondary_command
    )

    # =================================================
    # TOTAL GENERALIZED JOINT UPDATE
    # =================================================

    total_delta = (
        primary_delta
        + secondary_delta
    )

    # Maximum revolute step
    total_delta[:5] = np.clip(
        total_delta[:5],
        -max_revolute_step,
        max_revolute_step
    )

    # Maximum prismatic step
    total_delta[5] = np.clip(
        total_delta[5],
        -max_prismatic_step,
        max_prismatic_step
    )

    q_new = (
        q
        + total_delta[:5]
    )

    d6_new = float(
        d6
        + total_delta[5]
    )

    # =================================================
    # HARD JOINT LIMITS
    # =================================================

    q_new = np.clip(
        q_new,
        Q_MIN,
        Q_MAX
    )

    d6_new = float(
        np.clip(
            d6_new,
            D6_MIN,
            D6_MAX
        )
    )

    return {
        "q": q_new,
        "d6": d6_new,

        "error": error,
        "error_norm": error_norm,

        "sigma_min": sigma_min,
        "damping": damping,

        "joint_limit_cost": joint_limit_cost(
            q,
            d6,
            joint_limit_activation
        ),

        "joint_limit_margin": joint_limit_margin(
            q,
            d6
        ),

        "primary_delta": primary_delta,

        "singularity_command": singularity_command,

        "joint_limit_command": joint_limit_command,

        "secondary_command": secondary_command,

        "secondary_delta": secondary_delta,

        "total_delta": total_delta,

        "singularity_gradient": singularity_gradient,
    }


# =====================================================
# COMPLETE MULTI-OBJECTIVE IK SOLVER
# =====================================================

def solve_multi_objective_ik(
    target_position,
    q_initial,
    d6_initial,
    step_size=0.5,
    singularity_gain=0.005,
    joint_limit_gain=0.005,
    joint_limit_activation=0.80,
    tolerance=1e-4,
    max_iterations=1000,
    sigma_threshold=0.10,
    lambda_min=1e-4,
    lambda_max=0.20
):
    """
    Solve Cartesian position IK using:

        Adaptive DLS
        +
        Singularity Avoidance
        +
        Joint-Limit Avoidance
    """

    target_position = np.asarray(
        target_position,
        dtype=float
    )

    q = np.asarray(
        q_initial,
        dtype=float
    ).copy()

    d6 = float(
        d6_initial
    )

    if target_position.shape != (3,):
        raise ValueError(
            "target_position must have shape (3,)."
        )

    if q.shape != (5,):
        raise ValueError(
            "q_initial must contain five revolute joints."
        )

    if tolerance <= 0.0:
        raise ValueError(
            "tolerance must be positive."
        )

    if max_iterations < 1:
        raise ValueError(
            "max_iterations must be at least 1."
        )

    # =================================================
    # HISTORIES
    # =================================================

    position_history = []
    error_history = []

    sigma_min_history = []
    damping_history = []

    joint_limit_cost_history = []
    joint_limit_margin_history = []

    singularity_activity_history = []
    joint_limit_activity_history = []
    secondary_activity_history = []

    q_history = []
    d6_history = []

    converged = False

    # =================================================
    # ITERATIVE IK
    # =================================================

    for iteration in range(
        max_iterations
    ):

        _, current_position, _ = (
            get_end_effector_pose(
                q,
                d6
            )
        )

        error = (
            target_position
            - current_position
        )

        error_norm = float(
            np.linalg.norm(error)
        )

        current_sigma_min = (
            translational_sigma_min(
                q,
                d6
            )
        )

        current_limit_cost = (
            joint_limit_cost(
                q,
                d6,
                joint_limit_activation
            )
        )

        current_limit_margin = (
            joint_limit_margin(
                q,
                d6
            )
        )

        # ---------------------------------------------
        # SAVE CURRENT STATE
        # ---------------------------------------------

        position_history.append(
            current_position.copy()
        )

        error_history.append(
            error_norm
        )

        sigma_min_history.append(
            current_sigma_min
        )

        joint_limit_cost_history.append(
            current_limit_cost
        )

        joint_limit_margin_history.append(
            current_limit_margin
        )

        q_history.append(
            q.copy()
        )

        d6_history.append(
            d6
        )

        # ---------------------------------------------
        # CONVERGENCE
        # ---------------------------------------------

        if error_norm <= tolerance:
            converged = True
            break

        result = multi_objective_ik_step(
            q=q,
            d6=d6,
            target_position=target_position,
            step_size=step_size,
            singularity_gain=singularity_gain,
            joint_limit_gain=joint_limit_gain,
            joint_limit_activation=joint_limit_activation,
            sigma_threshold=sigma_threshold,
            lambda_min=lambda_min,
            lambda_max=lambda_max
        )

        q = result["q"]
        d6 = result["d6"]

        damping_history.append(
            result["damping"]
        )

        singularity_activity_history.append(
            float(
                np.linalg.norm(
                    result[
                        "singularity_command"
                    ]
                )
            )
        )

        joint_limit_activity_history.append(
            float(
                np.linalg.norm(
                    result[
                        "joint_limit_command"
                    ]
                )
            )
        )

        secondary_activity_history.append(
            float(
                np.linalg.norm(
                    result[
                        "secondary_delta"
                    ]
                )
            )
        )

    # =================================================
    # FINAL STATE
    # =================================================

    _, final_position, _ = (
        get_end_effector_pose(
            q,
            d6
        )
    )

    final_error_vector = (
        target_position
        - final_position
    )

    final_error = float(
        np.linalg.norm(
            final_error_vector
        )
    )

    final_sigma_min = (
        translational_sigma_min(
            q,
            d6
        )
    )

    final_limit_margin = (
        joint_limit_margin(
            q,
            d6
        )
    )

    final_limit_cost = (
        joint_limit_cost(
            q,
            d6,
            joint_limit_activation
        )
    )

    return {
        "q": q,
        "d6": d6,

        "target_position": target_position,
        "final_position": final_position,

        "final_error_vector": final_error_vector,
        "final_error": final_error,

        "final_sigma_min": final_sigma_min,

        "final_joint_limit_margin": (
            final_limit_margin
        ),

        "final_joint_limit_cost": (
            final_limit_cost
        ),

        "converged": converged,
        "iterations": iteration + 1,

        "position_history": np.asarray(
            position_history
        ),

        "error_history": np.asarray(
            error_history
        ),

        "sigma_min_history": np.asarray(
            sigma_min_history
        ),

        "damping_history": np.asarray(
            damping_history
        ),

        "joint_limit_cost_history": np.asarray(
            joint_limit_cost_history
        ),

        "joint_limit_margin_history": np.asarray(
            joint_limit_margin_history
        ),

        "singularity_activity_history": np.asarray(
            singularity_activity_history
        ),

        "joint_limit_activity_history": np.asarray(
            joint_limit_activity_history
        ),

        "secondary_activity_history": np.asarray(
            secondary_activity_history
        ),

        "q_history": np.asarray(
            q_history
        ),

        "d6_history": np.asarray(
            d6_history
        ),
    }
