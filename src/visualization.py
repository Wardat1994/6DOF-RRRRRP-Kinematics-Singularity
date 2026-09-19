import numpy as np
import matplotlib.pyplot as plt

from src.kinematics import get_joint_positions


def plot_robot(q, d6, show_frames=False):
    """
    Visualize the 6-DOF RRRRRP robotic manipulator in 3D.

    Parameters
    ----------
    q : array-like
        Revolute joint angles [theta1 ... theta5] in radians.

    d6 : float
        Prismatic displacement of joint 6 [m].

    show_frames : bool
        Reserved for future coordinate-frame visualization.
    """

    points = get_joint_positions(q, d6)

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    fig = plt.figure(figsize=(8, 7))

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    # Robot links
    ax.plot(
        x,
        y,
        z,
        "o-",
        linewidth=3,
        markersize=7,
        label="RRRRRP Manipulator"
    )

    # Base marker
    ax.scatter(
        x[0],
        y[0],
        z[0],
        s=100,
        marker="s",
        label="Base"
    )

    # End-effector marker
    ax.scatter(
        x[-1],
        y[-1],
        z[-1],
        s=120,
        marker="X",
        label="End-Effector"
    )

    # =====================================================
    # PHYSICAL JOINT LABELS
    # =====================================================

    # Base / Joint 1
    ax.text(
        x[0],
        y[0],
        z[0],
        "J1 / Base"
    )

    # Joint 2
    ax.text(
        x[1],
        y[1],
        z[1],
        "J2"
    )

    # Joint 3
    ax.text(
        x[2],
        y[2],
        z[2],
        "J3"
    )

    # Joint 4
    ax.text(
        x[3],
        y[3],
        z[3],
        "J4"
    )

    # Joint 5
    ax.text(
        x[4],
        y[4],
        z[4],
        "J5"
    )

    # Joint 6 (Prismatic) lies along the final segment
    j6_mid = 0.5 * (points[5] + points[6])

    ax.text(
        j6_mid[0],
        j6_mid[1],
        j6_mid[2],
        "J6 (P)"
    )

    # End-effector
    ax.text(
        x[-1],
        y[-1],
        z[-1],
        "EE"
    )

    # Axis labels
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    ax.set_title(
        "6-DOF RRRRRP Robotic Manipulator"
    )

    # Keep a visually balanced 3D scale
    max_range = np.max(
        np.ptp(points, axis=0)
    )

    if max_range < 1e-6:
        max_range = 1.0

    center = np.mean(points, axis=0)

    ax.set_xlim(
        center[0] - max_range / 2,
        center[0] + max_range / 2
    )

    ax.set_ylim(
        center[1] - max_range / 2,
        center[1] + max_range / 2
    )

    ax.set_zlim(
        max(0.0, center[2] - max_range / 2),
        center[2] + max_range / 2
    )

    ax.legend()

    plt.tight_layout()

    return fig, ax


if __name__ == "__main__":

    q = np.radians([
        -90,
        0,
        30,
        -15,
        0
    ])

    d6 = 0.20

    plot_robot(
        q,
        d6
    )

    plt.show()
