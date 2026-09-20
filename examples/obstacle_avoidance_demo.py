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
    get_joint_positions,
)

from src.obstacle_avoidance import (
    robot_to_sphere_clearances,
    minimum_obstacle_clearance,
    obstacle_avoidance_state,
)


# =====================================================
# ROBOT CONFIGURATION
# =====================================================

q = np.radians([
    -90.0,
    0.0,
    30.0,
    -15.0,
    0.0
])

d6 = 0.20


# =====================================================
# SPHERICAL OBSTACLE
# =====================================================
#
# The obstacle is intentionally positioned near
# Link 3 without initially intersecting the robot.
#
# This creates an active obstacle-avoidance scenario.
# =====================================================

obstacle_center = np.array([
    0.35,
    -1.34641016,
    0.70
])

obstacle_radius = 0.15

# Approximate physical radius of robot links
link_radius = 0.05


# =====================================================
# OBSTACLE CONTROL DISTANCES
# =====================================================

safety_distance = 0.10

influence_distance = 0.40


# =====================================================
# JOINT POSITIONS
# =====================================================

joint_positions = (
    get_joint_positions(
        q,
        d6
    )
)


# =====================================================
# ALL LINK CLEARANCES
# =====================================================

link_clearances = (
    robot_to_sphere_clearances(
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
    )
)


# =====================================================
# MINIMUM CLEARANCE
# =====================================================

minimum_result = (
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
    )
)


# =====================================================
# COMPLETE OBSTACLE STATE
# =====================================================

state = (
    obstacle_avoidance_state(
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
        ),
        safety_distance=(
            safety_distance
        ),
        influence_distance=(
            influence_distance
        )
    )
)


# =====================================================
# TERMINAL RESULTS
# =====================================================

print("=" * 72)

print(
    "OBSTACLE AVOIDANCE GEOMETRY DEMO"
)

print("=" * 72)


print(
    "\nObstacle center [m]:"
)

print(
    obstacle_center
)


print(
    "\nObstacle radius [m]: "
    f"{obstacle_radius:.4f}"
)

print(
    "Robot link radius [m]: "
    f"{link_radius:.4f}"
)

print(
    "Safety distance [m]: "
    f"{safety_distance:.4f}"
)

print(
    "Influence distance [m]: "
    f"{influence_distance:.4f}"
)


# =====================================================
# LINK-BY-LINK CLEARANCES
# =====================================================

print(
    "\nLINK CLEARANCES"
)

print("-" * 72)


for result in link_clearances:

    print(
        f"Link "
        f"{result['link_number']}: "
        f"{result['clearance']:.8f} m"
    )


# =====================================================
# MINIMUM RESULT
# =====================================================

print(
    "\nMINIMUM CLEARANCE"
)

print("-" * 72)

print(
    "Closest link: "
    f"{minimum_result['link_number']}"
)

print(
    "Minimum clearance: "
    f"{minimum_result['clearance']:.8f} m"
)

print(
    "Center distance: "
    f"{minimum_result['center_distance']:.8f} m"
)

print(
    "Closest point:"
)

print(
    minimum_result[
        "closest_point"
    ]
)


# =====================================================
# AVOIDANCE STATE
# =====================================================

print(
    "\nOBSTACLE AVOIDANCE STATE"
)

print("-" * 72)

print(
    "Activation: "
    f"{state['activation']:.8f}"
)

print(
    "Collision: "
    f"{state['collision']}"
)

print(
    "Clearance gradient:"
)

print(
    state[
        "gradient"
    ]
)

print(
    "Normalized avoidance direction:"
)

print(
    state[
        "direction"
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
# CREATE 3D FIGURE
# =====================================================

fig = plt.figure(
    figsize=(10, 8)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)


# =====================================================
# ROBOT LINKS
# =====================================================

ax.plot(
    joint_positions[:, 0],
    joint_positions[:, 1],
    joint_positions[:, 2],
    "o-",
    linewidth=3,
    markersize=7,
    label="RRRRRP Manipulator"
)


# =====================================================
# BASE
# =====================================================

ax.scatter(
    joint_positions[0, 0],
    joint_positions[0, 1],
    joint_positions[0, 2],
    s=100,
    marker="s",
    label="Base"
)


# =====================================================
# END EFFECTOR
# =====================================================

ax.scatter(
    joint_positions[-1, 0],
    joint_positions[-1, 1],
    joint_positions[-1, 2],
    s=130,
    marker="X",
    label="End-Effector"
)


# =====================================================
# JOINT LABELS
# =====================================================

for index in range(
    len(joint_positions)
):

    if index == 0:

        label = "Base"

    elif index == 6:

        label = "EE"

    else:

        label = (
            f"J{index + 1}"
        )

    ax.text(
        joint_positions[
            index,
            0
        ],
        joint_positions[
            index,
            1
        ],
        joint_positions[
            index,
            2
        ],
        label
    )


# =====================================================
# SPHERICAL OBSTACLE
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

ax.plot_surface(
    sphere_x,
    sphere_y,
    sphere_z,
    alpha=0.40
)


# =====================================================
# OBSTACLE CENTER
# =====================================================

ax.scatter(
    obstacle_center[0],
    obstacle_center[1],
    obstacle_center[2],
    s=100,
    marker="X",
    label="Obstacle Center"
)


# =====================================================
# CLOSEST POINT
# =====================================================

closest_point = (
    minimum_result[
        "closest_point"
    ]
)

ax.scatter(
    closest_point[0],
    closest_point[1],
    closest_point[2],
    s=100,
    marker="o",
    label="Closest Robot Point"
)


# =====================================================
# DISTANCE LINE
# =====================================================

ax.plot(
    [
        closest_point[0],
        obstacle_center[0]
    ],
    [
        closest_point[1],
        obstacle_center[1]
    ],
    [
        closest_point[2],
        obstacle_center[2]
    ],
    "--",
    linewidth=2,
    label="Minimum Distance"
)


# =====================================================
# SAFETY SPHERE
# =====================================================

safety_radius = (
    obstacle_radius
    + link_radius
    + safety_distance
)

safety_x = (
    obstacle_center[0]
    + safety_radius
    * np.outer(
        np.cos(u),
        np.sin(v)
    )
)

safety_y = (
    obstacle_center[1]
    + safety_radius
    * np.outer(
        np.sin(u),
        np.sin(v)
    )
)

safety_z = (
    obstacle_center[2]
    + safety_radius
    * np.outer(
        np.ones_like(u),
        np.cos(v)
    )
)

ax.plot_wireframe(
    safety_x,
    safety_y,
    safety_z,
    linewidth=0.5,
    alpha=0.35
)


# =====================================================
# INFLUENCE SPHERE
# =====================================================

influence_radius = (
    obstacle_radius
    + link_radius
    + influence_distance
)

influence_x = (
    obstacle_center[0]
    + influence_radius
    * np.outer(
        np.cos(u),
        np.sin(v)
    )
)

influence_y = (
    obstacle_center[1]
    + influence_radius
    * np.outer(
        np.sin(u),
        np.sin(v)
    )
)

influence_z = (
    obstacle_center[2]
    + influence_radius
    * np.outer(
        np.ones_like(u),
        np.cos(v)
    )
)

ax.plot_wireframe(
    influence_x,
    influence_y,
    influence_z,
    linewidth=0.4,
    alpha=0.15
)


# =====================================================
# AXIS LABELS
# =====================================================

ax.set_xlabel(
    "X [m]"
)

ax.set_ylabel(
    "Y [m]"
)

ax.set_zlabel(
    "Z [m]"
)


ax.set_title(
    "RRRRRP Manipulator — "
    "Obstacle Clearance Geometry"
)


# =====================================================
# BALANCED 3D LIMITS
# =====================================================

environment_points = np.vstack([
    joint_positions,
    obstacle_center
    + np.array([
        influence_radius,
        influence_radius,
        influence_radius
    ]),
    obstacle_center
    - np.array([
        influence_radius,
        influence_radius,
        influence_radius
    ])
])


minimum_xyz = np.min(
    environment_points,
    axis=0
)

maximum_xyz = np.max(
    environment_points,
    axis=0
)

center_xyz = (
    minimum_xyz
    + maximum_xyz
) / 2.0

maximum_range = float(
    np.max(
        maximum_xyz
        - minimum_xyz
    )
)

if maximum_range < 1e-6:

    maximum_range = 1.0


half_range = (
    maximum_range
    / 2.0
)


ax.set_xlim(
    center_xyz[0]
    - half_range,
    center_xyz[0]
    + half_range
)

ax.set_ylim(
    center_xyz[1]
    - half_range,
    center_xyz[1]
    + half_range
)

ax.set_zlim(
    center_xyz[2]
    - half_range,
    center_xyz[2]
    + half_range
)


ax.grid(True)

ax.legend(
    loc="best"
)


fig.tight_layout()


# =====================================================
# SAVE RESULT
# =====================================================

output_file = (
    results_dir
    / "obstacle_avoidance_geometry.png"
)

fig.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)


print(
    "\nSaved figure:"
)

print(
    "results/"
    "obstacle_avoidance_geometry.png"
)


plt.show()
