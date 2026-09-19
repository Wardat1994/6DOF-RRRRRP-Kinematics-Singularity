import numpy as np

from src.kinematics import get_end_effector_pose


# =====================================================
# JOINT LIMITS
# =====================================================

Q_MIN = np.radians([
    -180,
    -90,
    -90,
    -90,
    -90
])

Q_MAX = np.radians([
    180,
    90,
    90,
    90,
    90
])

D6_MIN = 0.03
D6_MAX = 0.48


# =====================================================
# WORKSPACE GENERATION
# =====================================================

def generate_workspace(
    number_of_samples=20000,
    seed=42
):
    """
    Generate reachable end-effector positions of the
    6-DOF RRRRRP robotic manipulator.

    Random joint configurations are sampled within
    the defined joint limits.

    Parameters
    ----------
    number_of_samples : int
        Number of random robot configurations.

    seed : int
        Random seed for reproducible results.

    Returns
    -------
    workspace_points : ndarray
        Array with shape (N, 3) containing
        end-effector Cartesian positions.

    joint_samples : ndarray
        Sampled robot configurations:
        [theta1, theta2, theta3, theta4, theta5, d6]
    """

    rng = np.random.default_rng(seed)

    # Sample revolute joints
    q_samples = rng.uniform(
        Q_MIN,
        Q_MAX,
        size=(number_of_samples, 5)
    )

    # Sample prismatic joint
    d6_samples = rng.uniform(
        D6_MIN,
        D6_MAX,
        size=number_of_samples
    )

    workspace_points = np.zeros(
        (number_of_samples, 3),
        dtype=float
    )

    # Calculate end-effector position
    for i in range(number_of_samples):

        _, position, _ = get_end_effector_pose(
            q_samples[i],
            d6_samples[i]
        )

        workspace_points[i] = position

    # Store complete sampled joint configurations
    joint_samples = np.column_stack([
        q_samples,
        d6_samples
    ])

    return workspace_points, joint_samples


# =====================================================
# WORKSPACE STATISTICS
# =====================================================

def workspace_statistics(workspace_points):
    """
    Calculate basic Cartesian workspace statistics.
    """

    workspace_points = np.asarray(
        workspace_points,
        dtype=float
    )

    minimum = np.min(
        workspace_points,
        axis=0
    )

    maximum = np.max(
        workspace_points,
        axis=0
    )

    span = maximum - minimum

    radial_distance = np.linalg.norm(
        workspace_points,
        axis=1
    )

    maximum_reach = np.max(
        radial_distance
    )

    minimum_reach = np.min(
        radial_distance
    )

    return {
        "x_min": minimum[0],
        "x_max": maximum[0],

        "y_min": minimum[1],
        "y_max": maximum[1],

        "z_min": minimum[2],
        "z_max": maximum[2],

        "x_span": span[0],
        "y_span": span[1],
        "z_span": span[2],

        "minimum_reach": minimum_reach,
        "maximum_reach": maximum_reach
    }


# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":

    points, configurations = generate_workspace(
        number_of_samples=5000
    )

    stats = workspace_statistics(
        points
    )

    print(
        "Generated workspace points:",
        len(points)
    )

    print("\nWorkspace Statistics:")

    for key, value in stats.items():

        print(
            f"{key}: {value:.4f}"
        )
