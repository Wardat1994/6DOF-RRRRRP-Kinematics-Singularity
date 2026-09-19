# 6-Joint RRRRRP Robotic Manipulator
## Kinematics, Singularity Analysis, Inverse Kinematics, and Multi-Objective Control

A research-oriented Python implementation for the modeling, simulation, and numerical control of a **6-joint RRRRRP serial robotic manipulator**, consisting of:

- Five revolute joints
- One prismatic joint

The repository provides a modular framework for studying:

- Denavit–Hartenberg modeling
- Forward kinematics
- 3D robot visualization
- Reachable workspace analysis
- Translational and geometric Jacobians
- Numerical Jacobian validation
- Singular-value analysis
- Translational manipulability
- Joint-space trajectory generation
- Cartesian trajectory analysis
- Damped Least Squares inverse kinematics
- Adaptive damping
- Singularity avoidance
- Joint-limit avoidance
- Null-space secondary objectives
- Multi-objective inverse kinematics
- Protected joint safety margins
- Automated numerical testing
- GitHub Actions continuous integration

---

# Robot Architecture

The manipulator contains six joints:

| Joint | Type | Variable |
|---|---|---|
| J1 | Revolute | θ1 |
| J2 | Revolute | θ2 |
| J3 | Revolute | θ3 |
| J4 | Revolute | θ4 |
| J5 | Revolute | θ5 |
| J6 | Prismatic | d6 |

The generalized joint vector is:

```math
q =
\begin{bmatrix}
\theta_1 &
\theta_2 &
\theta_3 &
\theta_4 &
\theta_5 &
d_6
\end{bmatrix}^{T}
```

The first five generalized coordinates are angular variables, while `d6` represents the displacement of the final prismatic joint.

---

# Geometric Parameters

| Parameter | Value |
|---|---:|
| H | 0.5 m |
| a2 | 1.0 m |
| a3 | 0.8 m |
| a4 | 0.6 m |

---

# Denavit–Hartenberg Model

The robot is modeled using the standard Denavit–Hartenberg convention.

| Joint | aᵢ | αᵢ | dᵢ | θᵢ | Type |
|---|---:|---:|---:|---:|---|
| J1 | 0 | π/2 | H | θ1 | Revolute |
| J2 | a2 | 0 | 0 | θ2 | Revolute |
| J3 | a3 | 0 | 0 | θ3 | Revolute |
| J4 | a4 | 0 | 0 | θ4 | Revolute |
| J5 | 0 | -π/2 | 0 | θ5 | Revolute |
| J6 | 0 | 0 | d6 | 0 | Prismatic |

The standard homogeneous transformation matrix is:

```math
{}^{i-1}T_i =
\begin{bmatrix}
\cos\theta_i &
-\sin\theta_i\cos\alpha_i &
\sin\theta_i\sin\alpha_i &
a_i\cos\theta_i
\\
\sin\theta_i &
\cos\theta_i\cos\alpha_i &
-\cos\theta_i\sin\alpha_i &
a_i\sin\theta_i
\\
0 &
\sin\alpha_i &
\cos\alpha_i &
d_i
\\
0 & 0 & 0 & 1
\end{bmatrix}
```

The complete transformation from the base frame to the end-effector is:

```math
{}^{0}T_6 =
{}^{0}T_1
{}^{1}T_2
{}^{2}T_3
{}^{3}T_4
{}^{4}T_5
{}^{5}T_6
```

---

# Forward Kinematics

The end-effector pose is represented by:

```math
{}^{0}T_6 =
\begin{bmatrix}
R_0^6 & p_e
\\
0 & 1
\end{bmatrix}
```

where the Cartesian end-effector position is:

```math
p_e =
\begin{bmatrix}
x \\
y \\
z
\end{bmatrix}
```

The implementation calculates all intermediate transformation matrices and joint positions.

Main implementation:

```text
src/kinematics.py
```

Run:

```bash
python examples/forward_kinematics_demo.py
```

---

# 3D Robot Visualization

The robot configuration is reconstructed directly from the homogeneous transformation matrices.

![3D Robot Configuration](results/robot_configuration.png)

The visualization includes:

- Base frame
- Revolute joints
- Serial links
- Prismatic section
- End-effector

Main implementation:

```text
src/visualization.py
```

Run:

```bash
python examples/visualization_demo.py
```

---

# Reachable Workspace

The reachable workspace is estimated by randomly sampling valid robot configurations inside the implemented joint limits.

For every sampled configuration, forward kinematics is evaluated to obtain the Cartesian end-effector position.

![Reachable Workspace](results/workspace.png)

Workspace analysis is useful for:

- Reachability studies
- Trajectory feasibility
- Motion planning
- Workspace-boundary analysis
- Future obstacle avoidance
- Constraint-aware control

Main implementation:

```text
src/workspace.py
```

Run:

```bash
python examples/workspace_demo.py
```

---

# Jacobian Modeling

The project implements both:

- Translational Jacobian
- Full geometric Jacobian

For the current position-control and avoidance algorithms, the **translational Jacobian** is the primary differential kinematic model.

---

## Translational Jacobian

The translational Jacobian has dimensions:

```math
J_v \in \mathbb{R}^{3\times6}
```

and relates generalized joint rates to Cartesian end-effector velocity:

```math
v_e = J_v(q)\dot{q}
```

For a revolute joint:

```math
J_{v_i}
=
z_{i-1}
\times
\left(
o_6-o_{i-1}
\right)
```

For the final prismatic joint:

```math
J_{v_6}=z_5
```

---

## Angular Jacobian

For revolute joints:

```math
J_{\omega_i}=z_{i-1}
```

The final prismatic joint has no angular contribution:

```math
J_{\omega_6}=0
```

---

## Geometric Jacobian

The full geometric Jacobian is assembled as:

```math
J_g =
\begin{bmatrix}
J_v \\
J_\omega
\end{bmatrix}
```

It relates generalized joint rates to the end-effector twist:

```math
\begin{bmatrix}
v_e \\
\omega_e
\end{bmatrix}
=
J_g(q)\dot{q}
```

Main implementation:

```text
src/jacobian.py
```

---

# Jacobian Validation

The analytical translational Jacobian is independently validated using central finite differences of the forward kinematic model.

For each generalized coordinate:

```math
\frac{\partial p_e}{\partial q_i}
\approx
\frac{
p_e(q_i+\varepsilon)
-
p_e(q_i-\varepsilon)
}{
2\varepsilon
}
```

The numerical derivative is compared with the analytical Jacobian implementation through automated unit tests.

This provides numerical validation of the differential kinematic model.

---

# Translational Singularity Analysis

The current position-control algorithms use the translational Jacobian:

```math
J_v \in \mathbb{R}^{3\times6}
```

A regular Cartesian position configuration has:

```math
\operatorname{rank}(J_v)=3
```

A translational singularity occurs when:

```math
\operatorname{rank}(J_v)<3
```

Singularity analysis is performed using Singular Value Decomposition:

```math
J_v =
U\Sigma V^T
```

The singular values are:

```math
\sigma_1,\sigma_2,\sigma_3
```

and the minimum singular value is:

```math
\sigma_{\min}
=
\min
\left(
\sigma_1,\sigma_2,\sigma_3
\right)
```

As:

```math
\sigma_{\min}\rightarrow0
```

the robot approaches a translational singular configuration.

---

# Translational Manipulability

The translational manipulability measure is:

```math
w(q)
=
\sqrt{
\det
\left(
J_vJ_v^T
\right)
}
```

It is also equal to the product of the singular values of `Jv`:

```math
w(q)
=
\sigma_1\sigma_2\sigma_3
```

![Manipulability](results/manipulability_vs_theta3.png)

Lower manipulability indicates reduced Cartesian dexterity.

Main implementation:

```text
src/singularity.py
```

---

# Joint-Space Trajectory Generation

The repository includes linear interpolation between initial and final generalized robot configurations.

For:

```math
q_s \rightarrow q_f
```

intermediate joint configurations are generated and evaluated through forward kinematics.

The resulting Cartesian end-effector trajectory is:

```math
p_e(t)
=
\begin{bmatrix}
x(t) \\
y(t) \\
z(t)
\end{bmatrix}
```

Main implementation:

```text
src/trajectory.py
```

---

# 3D End-Effector Trajectory

![3D End-Effector Trajectory](results/trajectory_3d.png)

Run:

```bash
python examples/trajectory_demo.py
```

---

# Cartesian Position Along the Trajectory

The Cartesian coordinates are evaluated throughout the simulated motion:

```math
x(t),\qquad y(t),\qquad z(t)
```

![Trajectory XYZ](results/trajectory_xyz_vs_time.png)

---

# Cartesian Velocity Analysis

The Cartesian velocity is estimated numerically:

```math
v(t)
=
\frac{dp_e(t)}{dt}
```

The Cartesian speed is:

```math
\|v(t)\|
=
\sqrt{
v_x^2+v_y^2+v_z^2
}
```

![Trajectory Speed](results/trajectory_speed.png)

---

# Manipulability Along the Trajectory

The translational manipulability measure is calculated at every sampled robot configuration.

![Trajectory Manipulability](results/trajectory_manipulability.png)

This makes it possible to monitor dexterity degradation during motion.

---

# Minimum Singular Value Along the Trajectory

The minimum singular value of the translational Jacobian is evaluated throughout the trajectory.

![Trajectory Minimum Singular Value](results/trajectory_sigma_min.png)

This provides a trajectory-level singularity indicator.

Run:

```bash
python examples/trajectory_analysis_demo.py
```

---

# Inverse Kinematics

The repository implements iterative Cartesian position inverse kinematics using Damped Least Squares.

The Cartesian position error is:

```math
e =
p_d-p_e
```

where:

- `p_d` is the desired Cartesian position
- `p_e` is the current Cartesian position

The Damped Least Squares pseudoinverse is:

```math
J_\lambda^\#
=
J_v^T
\left(
J_vJ_v^T+\lambda^2I
\right)^{-1}
```

The generalized joint correction is:

```math
\Delta q
=
\alpha
J_\lambda^\#
e
```

where:

- `λ` is the damping coefficient
- `α` is the iteration step size

Main implementation:

```text
src/inverse_kinematics.py
```

Run:

```bash
python examples/inverse_kinematics_demo.py
```

Generated results include:

![Inverse Kinematics Path](results/inverse_kinematics_path.png)

![Inverse Kinematics Error](results/inverse_kinematics_error.png)

---

# Adaptive Damped Least Squares

A fixed damping coefficient can produce unnecessarily slow convergence away from singularities or insufficient stabilization near singular configurations.

The repository therefore implements adaptive damping based on the minimum singular value.

Conceptually:

```math
\lambda
=
f(\sigma_{\min})
```

where:

- damping increases near singular configurations
- damping approaches a small minimum value in well-conditioned configurations

Main implementation:

```text
src/adaptive_damping.py
```

Run:

```bash
python examples/adaptive_damping_demo.py
```

Comparison results:

![DLS Error Comparison](results/dls_error_comparison.png)

![DLS Damping Comparison](results/dls_damping_comparison.png)

![DLS Sigma Minimum Comparison](results/dls_sigma_min_comparison.png)

![DLS Path Comparison](results/dls_path_comparison.png)

---

# Singularity Avoidance

The inverse kinematics controller was extended with a secondary objective that attempts to increase the minimum singular value of the translational Jacobian.

The minimum singular value gradient is estimated numerically:

```math
\nabla_q\sigma_{\min}
```

The primary Cartesian task is solved using Adaptive DLS.

The secondary singularity-avoidance motion is projected through a damped task-priority projector:

```math
N_\lambda
=
I-J_\lambda^\#J_v
```

The generalized update has the form:

```math
\Delta q
=
\Delta q_{\text{primary}}
+
N_\lambda
\Delta q_{\text{singularity}}
```

Main implementation:

```text
src/singularity_avoidance.py
```

Run:

```bash
python examples/singularity_avoidance_demo.py
```

Generated results:

![Singularity Avoidance Sigma Minimum](results/singularity_avoidance_sigma_min.png)

![Singularity Avoidance Error](results/singularity_avoidance_error.png)

![Singularity Avoidance Activity](results/singularity_avoidance_activity.png)

![Singularity Avoidance Path](results/singularity_avoidance_path.png)

---

# Joint-Limit Avoidance

The repository also implements a secondary objective that moves generalized joint variables away from their physical limits.

Each generalized coordinate is normalized relative to its allowable range.

The normalized coordinate satisfies approximately:

```math
-1
\leq
\bar{q}_i
\leq
1
```

where:

```math
\bar{q}_i=0
```

represents the center of the allowable joint range.

A joint-limit penalty activates when a joint enters a predefined near-limit region.

The secondary correction is projected through the damped task-priority projector:

```math
\Delta q
=
\Delta q_{\text{primary}}
+
N_\lambda
\Delta q_{\text{limit}}
```

Main implementation:

```text
src/joint_limit_avoidance.py
```

Run:

```bash
python examples/joint_limit_avoidance_demo.py
```

Generated results:

![Joint-Limit Margin Comparison](results/joint_limit_margin_comparison.png)

![Joint-Limit Cost Comparison](results/joint_limit_cost_comparison.png)

![Joint-Limit Error Comparison](results/joint_limit_error_comparison.png)

![Joint-Limit Path Comparison](results/joint_limit_path_comparison.png)

---

# Multi-Objective Inverse Kinematics

The current controller combines:

1. Cartesian position tracking
2. Adaptive damping
3. Singularity avoidance
4. Joint-limit avoidance
5. Protected internal joint limits

The generalized update is structured as:

```math
\Delta q
=
J_\lambda^\# e
+
N_\lambda
\left(
k_s a_s g_s
+
k_l a_l g_l
\right)
```

where:

- `Jλ#` is the adaptive DLS pseudoinverse
- `e` is the Cartesian position error
- `Nλ` is the damped task-priority projector
- `gs` is the singularity-avoidance direction
- `gl` is the joint-limit-avoidance direction
- `as` is the singularity activation level
- `al` is the joint-limit activation level
- `ks` and `kl` are secondary-task gains

---

## Singularity Activation

Singularity avoidance is activated only when:

```math
\sigma_{\min}
<
\sigma_{\text{threshold}}
```

A continuous activation factor is used so that the avoidance contribution decreases smoothly as the robot moves away from the singular region.

---

## Joint-Limit Activation

Joint-limit avoidance becomes stronger as a generalized joint coordinate approaches its physical boundary.

The controller therefore provides increasing secondary-task activity near dangerous joint configurations.

---

# Protected Joint Safety Margin

In addition to the null-space joint-limit avoidance task, the improved controller includes a protected internal safety region.

The current default value is:

```math
m_{\text{safety}}=0.05
```

which reserves approximately 5% of each joint's normalized range near the physical limits.

The controller therefore combines:

```text
Null-space joint-limit avoidance
+
Protected joint safety bounds
```

The protected layer prevents the generalized coordinates from crossing the configured internal safety boundary during the multi-objective IK experiment.

Main implementation:

```text
src/multi_objective_ik.py
```

Run:

```bash
python examples/multi_objective_ik_demo.py
```

---

# Multi-Objective Controller Results

The current comparison evaluates four control strategies:

- Adaptive DLS
- Adaptive DLS + Singularity Avoidance
- Adaptive DLS + Joint-Limit Avoidance
- Multi-Objective IK

---

## Cartesian Error Comparison

![Multi-Objective Error Comparison](results/multi_objective_error_comparison.png)

The multi-objective controller maintains Cartesian convergence while simultaneously activating secondary tasks.

---

## Singularity Metric Comparison

![Multi-Objective Sigma Minimum](results/multi_objective_sigma_min_comparison.png)

The controller moves rapidly away from the initial near-singular configuration while maintaining Cartesian position control.

---

## Joint-Limit Margin Comparison

![Multi-Objective Joint-Limit Margin](results/multi_objective_joint_limit_margin.png)

The improved multi-objective controller maintains the configured protected joint safety margin during the evaluated motion.

---

## Cartesian Path Comparison

![Multi-Objective Cartesian Path](results/multi_objective_path_comparison.png)

The Cartesian trajectory demonstrates the effect of redundancy resolution and secondary objectives on the path followed by the end-effector.

---

# Representative Multi-Objective Experiment

For the current test scenario, the initial configuration was intentionally selected close to both:

- a translational singularity
- one or more joint limits

Initial minimum singular value:

```text
1.93697243e-05
```

Initial normalized joint-limit margin:

```text
0.06800000
```

The evaluated Multi-Objective IK controller converged in:

```text
33 iterations
```

with final Cartesian position error:

```text
7.18180001e-05 m
```

Final minimum singular value:

```text
1.01351868
```

Final normalized joint-limit margin:

```text
0.13176671
```

Final joint-limit cost:

```text
0.05972685
```

These values describe the current controlled numerical experiment and should not be interpreted as universal performance guarantees for all robot configurations.

---

# Automated Testing

The repository contains automated tests for the major mathematical and control components.

Current testing includes:

- DH transformation matrices
- Forward kinematics
- End-effector pose extraction
- Rotation matrix validation
- Translational Jacobian dimensions
- Angular Jacobian dimensions
- Geometric Jacobian dimensions
- Prismatic-joint angular contribution
- Finite-difference Jacobian validation
- Singularity metrics
- Manipulability
- Joint-space trajectories
- Cartesian trajectories
- Damped Least Squares inverse kinematics
- Adaptive damping
- Singularity avoidance
- Joint-limit avoidance
- Multi-objective IK
- Joint-limit safety checks
- Convergence checks
- Controller history validation

Run all tests:

```bash
python -m pytest tests -v
```

The complete test suite is also executed automatically using **GitHub Actions**.

---

# Project Structure

```text
6DOF-RRRRRP-Kinematics-Singularity/
│
├── src/
│   ├── kinematics.py
│   ├── visualization.py
│   ├── workspace.py
│   ├── jacobian.py
│   ├── singularity.py
│   ├── trajectory.py
│   ├── inverse_kinematics.py
│   ├── adaptive_damping.py
│   ├── singularity_avoidance.py
│   ├── joint_limit_avoidance.py
│   └── multi_objective_ik.py
│
├── examples/
│   ├── forward_kinematics_demo.py
│   ├── visualization_demo.py
│   ├── workspace_demo.py
│   ├── singularity_demo.py
│   ├── trajectory_demo.py
│   ├── trajectory_analysis_demo.py
│   ├── inverse_kinematics_demo.py
│   ├── adaptive_damping_demo.py
│   ├── singularity_avoidance_demo.py
│   ├── joint_limit_avoidance_demo.py
│   └── multi_objective_ik_demo.py
│
├── tests/
│   ├── test_kinematics.py
│   ├── test_jacobian.py
│   ├── test_singularity.py
│   ├── test_trajectory.py
│   ├── test_inverse_kinematics.py
│   ├── test_adaptive_damping.py
│   ├── test_singularity_avoidance.py
│   ├── test_joint_limit_avoidance.py
│   └── test_multi_objective_ik.py
│
├── results/
│   ├── robot_configuration.png
│   ├── workspace.png
│   ├── manipulability_vs_theta3.png
│   ├── trajectory_3d.png
│   ├── trajectory_xyz_vs_time.png
│   ├── trajectory_speed.png
│   ├── trajectory_manipulability.png
│   ├── trajectory_sigma_min.png
│   ├── inverse_kinematics_path.png
│   ├── inverse_kinematics_error.png
│   ├── dls_error_comparison.png
│   ├── dls_damping_comparison.png
│   ├── dls_sigma_min_comparison.png
│   ├── dls_path_comparison.png
│   ├── singularity_avoidance_sigma_min.png
│   ├── singularity_avoidance_error.png
│   ├── singularity_avoidance_activity.png
│   ├── singularity_avoidance_path.png
│   ├── joint_limit_margin_comparison.png
│   ├── joint_limit_cost_comparison.png
│   ├── joint_limit_error_comparison.png
│   ├── joint_limit_path_comparison.png
│   ├── multi_objective_error_comparison.png
│   ├── multi_objective_sigma_min_comparison.png
│   ├── multi_objective_joint_limit_margin.png
│   └── multi_objective_path_comparison.png
│
├── .github/
│   └── workflows/
│       └── python-tests.yml
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

# Installation

Clone the repository and install the required Python packages:

```bash
pip install -r requirements.txt
```

Current dependencies include:

```text
NumPy
Matplotlib
Pytest
```

---

# Running the Examples

Forward kinematics:

```bash
python examples/forward_kinematics_demo.py
```

Robot visualization:

```bash
python examples/visualization_demo.py
```

Workspace analysis:

```bash
python examples/workspace_demo.py
```

Singularity analysis:

```bash
python examples/singularity_demo.py
```

Trajectory generation:

```bash
python examples/trajectory_demo.py
```

Trajectory analysis:

```bash
python examples/trajectory_analysis_demo.py
```

Inverse kinematics:

```bash
python examples/inverse_kinematics_demo.py
```

Adaptive damping:

```bash
python examples/adaptive_damping_demo.py
```

Singularity avoidance:

```bash
python examples/singularity_avoidance_demo.py
```

Joint-limit avoidance:

```bash
python examples/joint_limit_avoidance_demo.py
```

Multi-objective inverse kinematics:

```bash
python examples/multi_objective_ik_demo.py
```

---

# Current Capabilities

| Capability | Status |
|---|---|
| DH Modeling | ✅ Implemented |
| Forward Kinematics | ✅ Implemented |
| 3D Robot Visualization | ✅ Implemented |
| Workspace Generation | ✅ Implemented |
| Translational Jacobian | ✅ Implemented |
| Geometric Jacobian | ✅ Implemented |
| Numerical Jacobian Validation | ✅ Implemented |
| Translational Singularity Analysis | ✅ Implemented |
| Translational Manipulability | ✅ Implemented |
| Joint-Space Trajectory Generation | ✅ Implemented |
| Cartesian Trajectory Analysis | ✅ Implemented |
| Cartesian Velocity Analysis | ✅ Implemented |
| Damped Least Squares IK | ✅ Implemented |
| Adaptive Damping | ✅ Implemented |
| Singularity Avoidance | ✅ Implemented |
| Joint-Limit Avoidance | ✅ Implemented |
| Null-Space Secondary Objectives | ✅ Implemented |
| Multi-Objective IK | ✅ Implemented |
| Protected Joint Safety Margin | ✅ Implemented |
| Automated Unit Testing | ✅ Implemented |
| GitHub Actions CI | ✅ Implemented |
| Obstacle Avoidance | 🔄 Planned |
| Constraint-Aware Motion Planning | 🔄 Planned |
| MATLAB Validation | 🔄 Planned |
| Robotics Toolbox Comparison | 🔄 Planned |
| Hardware / Arduino Integration | 🔄 Planned |

---

# Next Development Stage

The next major development stage is:

## Obstacle Avoidance

The planned obstacle-avoidance framework will extend the current controller with:

- Obstacle representation
- Robot-to-obstacle distance calculation
- Collision detection
- Safety-distance monitoring
- Repulsive potential fields
- Obstacle-avoidance gradients
- Null-space obstacle avoidance
- Cartesian trajectory obstacle avoidance
- Combination with singularity avoidance
- Combination with joint-limit avoidance

The long-term multi-objective controller is expected to combine:

```math
\text{Cartesian Tracking}
+
\text{Singularity Avoidance}
+
\text{Joint-Limit Avoidance}
+
\text{Obstacle Avoidance}
```

subject to robot constraints and safety requirements.

---

# Future Research Development

Planned extensions include:

- Obstacle detection and avoidance
- Artificial potential fields
- Distance-based collision constraints
- Constraint-aware trajectory optimization
- Hierarchical task-priority control
- Weighted pseudoinverse methods
- Cartesian trajectory tracking
- Joint-velocity constraints
- Joint-acceleration constraints
- Acceleration analysis
- Jerk analysis
- MATLAB validation
- MATLAB Robotics Toolbox comparison
- Real-time controller implementation
- Python–Arduino communication
- Experimental robotic validation

---

# Research Scope

This repository is intended as a modular research platform for robotic manipulator studies involving:

- Kinematic modeling
- Differential kinematics
- Jacobian-based control
- Inverse kinematics
- Singular-value analysis
- Singularity detection
- Singularity avoidance
- Redundancy resolution
- Null-space optimization
- Joint-limit avoidance
- Multi-objective control
- Motion planning
- Trajectory analysis
- Obstacle avoidance
- Constraint-aware robotics
- Real-time robotic control

---

# Important Modeling Note

The manipulator contains **six joints**, but the current avoidance and inverse-kinematics experiments focus primarily on the **three-dimensional Cartesian position task**.

Therefore, position-control singularity analysis in this repository is based primarily on:

```math
J_v \in \mathbb{R}^{3\times6}
```

rather than assuming that every six-joint configuration necessarily provides unrestricted six-dimensional Cartesian pose control.

This distinction is important when interpreting Jacobian rank, singular values, manipulability, and controller performance.

---

# Author

**Mohammad Yasin Alwardat**

Research interests:

- Robotic manipulator kinematics
- Jacobian-based control
- Damped Least Squares methods
- Singularity detection and avoidance
- Joint-limit avoidance
- Redundancy resolution
- Motion planning
- Trajectory optimization
- Obstacle avoidance
- Constraint-aware robotic control
- Python and MATLAB robotic modeling
- Experimental robotic control
