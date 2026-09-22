# 6-Joint RRRRRP Robotic Manipulator

## Kinematics, Singularity Analysis, Multi-Objective IK, and Obstacle-Aware Safety Control

A research-oriented Python framework for the modeling, simulation, analysis, and numerical control of a **6-joint RRRRRP serial robotic manipulator** consisting of:

- Five revolute joints
- One prismatic joint

The repository progressively develops the manipulator from basic forward kinematics to safety-constrained inverse kinematics with multiple static obstacles.

The current implementation includes:

- Denavit–Hartenberg modeling
- Forward kinematics
- 3D robot visualization
- Reachable workspace estimation
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
- Damped task-priority secondary objectives
- Multi-objective inverse kinematics
- Protected internal joint limits
- Sphere-based obstacle representation
- Link-to-obstacle clearance calculation
- Single-obstacle avoidance
- Single-obstacle safety filtering
- Safety-constrained obstacle-aware IK
- Multi-obstacle avoidance
- Multi-obstacle safety filtering
- Multi-obstacle safety-constrained IK
- Sampled nonlinear step validation
- Backtracking safety logic
- Automated unit, integration, and regression testing
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

For analytical purposes, the generalized coordinate vector can be written as:

```math
q_g =
\begin{bmatrix}
\theta_1 &
\theta_2 &
\theta_3 &
\theta_4 &
\theta_5 &
d_6
\end{bmatrix}^{T}
```

In the Python implementation, the five revolute coordinates are commonly stored in:

```text
q
```

while the prismatic displacement is passed separately as:

```text
d6
```

---

# Geometric Parameters

The nominal manipulator dimensions are:

| Parameter | Value |
|---|---:|
| H | 0.5 m |
| a2 | 1.0 m |
| a3 | 0.8 m |
| a4 | 0.6 m |

---

# Joint Limits

The implemented revolute-joint limits are:

```text
q_min = [-180°, -90°, -90°, -90°, -90°]
q_max = [ 180°,  90°,  90°,  90°,  90°]
```

The prismatic joint range is:

```text
0.03 m <= d6 <= 0.48 m
```

The controller can additionally reserve an internal protected region near these physical limits.

---

# Denavit–Hartenberg Model

The manipulator is modeled using the standard Denavit–Hartenberg convention.

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

The complete base-to-end-effector transformation is:

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

where:

```math
p_e =
\begin{bmatrix}
x \\
y \\
z
\end{bmatrix}
```

The implementation calculates:

- Individual DH transformations
- Complete forward transformation
- End-effector position
- End-effector orientation
- Intermediate joint positions

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

Robot geometry is reconstructed from the forward-kinematic chain.

![3D Robot Configuration](results/robot_configuration.png)

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

The manipulator workspace is estimated numerically by sampling valid configurations inside the implemented joint ranges.

Forward kinematics is evaluated for every sampled configuration.

![Reachable Workspace](results/workspace.png)

The workspace model can support:

- Reachability analysis
- Target feasibility studies
- Motion-planning experiments
- Workspace-boundary studies
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

The project implements:

- Translational Jacobian
- Angular Jacobian
- Full geometric Jacobian

For the current inverse-kinematics and avoidance controllers, the **translational Jacobian** is the primary differential kinematic model.

---

## Translational Jacobian

The translational Jacobian is:

```math
J_v \in \mathbb{R}^{3\times6}
```

and relates generalized joint velocity to Cartesian linear velocity:

```math
v_e = J_v(q_g)\dot{q}_g
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

For the prismatic joint:

```math
J_{\omega_6}=0
```

---

## Geometric Jacobian

The complete geometric Jacobian is assembled as:

```math
J_g =
\begin{bmatrix}
J_v \\
J_\omega
\end{bmatrix}
```

and relates generalized joint rates to the end-effector twist:

```math
\begin{bmatrix}
v_e \\
\omega_e
\end{bmatrix}
=
J_g(q_g)\dot{q}_g
```

Main implementation:

```text
src/jacobian.py
```

---

# Numerical Jacobian Validation

The analytical translational Jacobian is independently checked using central finite differences of the forward-kinematic model.

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

The numerical result is compared against the analytical implementation using automated tests.

This provides an independent numerical consistency check of the differential kinematics.

---

# Translational Singularity Analysis

The implemented Cartesian position controllers operate primarily on:

```math
J_v \in \mathbb{R}^{3\times6}
```

A regular configuration for the three-dimensional Cartesian position task has:

```math
\operatorname{rank}(J_v)=3
```

A translational singularity occurs when:

```math
\operatorname{rank}(J_v)<3
```

Singular Value Decomposition is used:

```math
J_v =
U\Sigma V^T
```

with singular values:

```math
\sigma_1,\sigma_2,\sigma_3
```

The minimum singular value is:

```math
\sigma_{\min}
=
\min(
\sigma_1,
\sigma_2,
\sigma_3
)
```

When:

```math
\sigma_{\min}\rightarrow0
```

the manipulator approaches a translational singularity.

Generated analyses include:

![Minimum Singular Value](results/sigma_min_vs_theta3.png)

![Condition Number](results/condition_number_vs_theta3.png)

---

# Translational Manipulability

The translational manipulability measure is:

```math
w(q)
=
\sqrt{
\det(
J_vJ_v^T
)
}
```

and equivalently:

```math
w(q)
=
\sigma_1\sigma_2\sigma_3
```

![Manipulability](results/manipulability_vs_theta3.png)

Main implementation:

```text
src/singularity.py
```

Run:

```bash
python examples/singularity_demo.py
```

---

# Joint-Space Trajectory Generation

The repository supports interpolation between initial and final generalized robot configurations.

For:

```math
q_s \rightarrow q_f
```

intermediate configurations are generated and mapped through forward kinematics.

Main implementation:

```text
src/trajectory.py
```

Run:

```bash
python examples/trajectory_demo.py
```

---

# Cartesian Trajectory Analysis

The Cartesian trajectory is:

```math
p_e(t)
=
\begin{bmatrix}
x(t) \\
y(t) \\
z(t)
\end{bmatrix}
```

### 3D trajectory

![3D End-Effector Trajectory](results/trajectory_3d.png)

### Cartesian coordinates

![Trajectory XYZ](results/trajectory_xyz_vs_time.png)

### Cartesian speed

The Cartesian velocity is estimated numerically:

```math
v(t)
=
\frac{dp_e(t)}{dt}
```

and:

```math
\|v(t)\|
=
\sqrt{
v_x^2+v_y^2+v_z^2
}
```

![Trajectory Speed](results/trajectory_speed.png)

### Manipulability along the trajectory

![Trajectory Manipulability](results/trajectory_manipulability.png)

### Minimum singular value along the trajectory

![Trajectory Minimum Singular Value](results/trajectory_sigma_min.png)

Run:

```bash
python examples/trajectory_analysis_demo.py
```

---

# Damped Least Squares Inverse Kinematics

The repository implements iterative Cartesian position inverse kinematics using Damped Least Squares.

The Cartesian position error is:

```math
e =
p_d-p_e
```

The damped pseudoinverse is:

```math
J_\lambda^\#
=
J_v^T
\left(
J_vJ_v^T+\lambda^2I
\right)^{-1}
```

The primary generalized correction is:

```math
\Delta q_g
=
\alpha
J_\lambda^\#
e
```

where:

- `λ` is the damping coefficient
- `α` is the numerical step size

Main implementation:

```text
src/inverse_kinematics.py
```

Run:

```bash
python examples/inverse_kinematics_demo.py
```

Generated results:

![Inverse Kinematics Path](results/inverse_kinematics_path.png)

![Inverse Kinematics Error](results/inverse_kinematics_error.png)

---

# Adaptive Damping

Fixed damping can be inefficient because the conditioning of the translational Jacobian changes throughout the motion.

The repository therefore adapts damping according to the minimum singular value:

```math
\lambda
=
f(\sigma_{\min})
```

The intended behavior is:

- Higher damping near poorly conditioned configurations
- Lower damping away from singular regions

Main implementation:

```text
src/adaptive_damping.py
```

Run:

```bash
python examples/adaptive_damping_demo.py
```

Generated results:

![DLS Error Comparison](results/dls_error_comparison.png)

![DLS Damping Comparison](results/dls_damping_comparison.png)

![DLS Sigma Minimum Comparison](results/dls_sigma_min_comparison.png)

![DLS Path Comparison](results/dls_path_comparison.png)

---

# Singularity Avoidance

A secondary objective is used to move the manipulator away from low values of the translational minimum singular value.

The numerical gradient is approximated as:

```math
\nabla_{q_g}\sigma_{\min}
```

The primary Cartesian task uses adaptive DLS.

The secondary singularity contribution is introduced through the damped task-priority projection:

```math
N_\lambda
=
I-J_\lambda^\#J_v
```

giving a controller of the form:

```math
\Delta q_g
=
\Delta q_{\text{primary}}
+
N_\lambda
\Delta q_{\text{singularity}}
```

Because the pseudoinverse is damped, `Nλ` should be interpreted as a **damped task-priority projection**, rather than an exact mathematical null-space projector.

Main implementation:

```text
src/singularity_avoidance.py
```

Run:

```bash
python examples/singularity_avoidance_demo.py
```

Generated results:

![Singularity Avoidance Sigma](results/singularity_avoidance_sigma_min.png)

![Singularity Avoidance Error](results/singularity_avoidance_error.png)

![Singularity Avoidance Activity](results/singularity_avoidance_activity.png)

![Singularity Avoidance Path](results/singularity_avoidance_path.png)

---

# Joint-Limit Avoidance

A second secondary objective moves generalized coordinates away from their physical limits.

Each generalized coordinate is normalized relative to its valid range.

The normalized coordinate approximately satisfies:

```math
-1
\leq
\bar{q}_i
\leq
1
```

The center of the valid range corresponds approximately to:

```math
\bar{q}_i=0
```

The joint-limit term becomes active as one or more generalized coordinates approach the configured boundary region.

The controller has the structure:

```math
\Delta q_g
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

# Protected Joint Safety Margin

The multi-objective controller includes an internal protected region in addition to the secondary joint-limit objective.

The current default normalized safety margin is:

```math
m_{\text{safety}}=0.05
```

The protected limits restrict accepted numerical updates from crossing the configured internal safety boundary.

This mechanism complements, rather than replaces, the continuous joint-limit avoidance objective.

---

# Multi-Objective Inverse Kinematics

The multi-objective controller combines:

1. Cartesian position tracking
2. Adaptive damping
3. Singularity avoidance
4. Joint-limit avoidance
5. Protected internal joint limits

A simplified representation is:

```math
\Delta q_g
=
\Delta q_{\text{primary}}
+
N_\lambda
\left(
k_s a_s g_s
+
k_l a_l g_l
\right)
```

where:

- `gs` is the singularity-avoidance direction
- `gl` is the joint-limit-avoidance direction
- `as` is the singularity activation
- `al` is the joint-limit activation
- `ks` and `kl` are secondary-task gains

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

### Cartesian error

![Multi-Objective Error Comparison](results/multi_objective_error_comparison.png)

### Minimum singular value

![Multi-Objective Sigma Minimum](results/multi_objective_sigma_min_comparison.png)

### Joint-limit margin

![Multi-Objective Joint-Limit Margin](results/multi_objective_joint_limit_margin.png)

### Cartesian path

![Multi-Objective Cartesian Path](results/multi_objective_path_comparison.png)

The experiment demonstrates that Cartesian convergence can be combined with secondary singularity and joint-limit objectives in the evaluated scenario.

The reported experiment should be interpreted as a numerical case study rather than a universal performance guarantee.

---

# Obstacle Representation

The obstacle-avoidance framework models static obstacles as spheres defined by:

```text
center = [x, y, z]
radius = r
```

Robot links are approximated geometrically as line segments with configurable link radius.

This produces a capsule-like link representation for clearance calculations.

---

# Link-to-Sphere Clearance

For each robot link, the closest point on the link segment to the obstacle center is computed.

The signed clearance conceptually has the form:

```math
c
=
d_{\text{centerline}}
-
r_{\text{obstacle}}
-
r_{\text{link}}
```

Interpretation:

```text
c > 0   separated
c = 0   contact boundary
c < 0   geometric intersection
```

The minimum clearance across all robot links defines the robot-to-obstacle clearance.

Main implementation:

```text
src/obstacle_avoidance.py
```

---

# Obstacle Activation

Obstacle avoidance uses a distance-dependent activation function.

Conceptually:

```text
clearance >= influence distance
    -> obstacle contribution inactive

safety distance < clearance < influence distance
    -> obstacle contribution gradually active

clearance <= safety distance
    -> maximum avoidance activation
```

This provides a smooth transition between inactive and active avoidance regions.

---

# Obstacle Clearance Gradient

The obstacle-clearance gradient is estimated numerically using finite differences:

```math
\nabla_{q_g} c
```

The gradient identifies a local generalized-coordinate direction that tends to increase obstacle clearance.

Because the generalized coordinates contain both radians and meters, numerical gradient normalization is unit-sensitive and should be interpreted accordingly.

---

# Single-Obstacle Avoidance

The initial obstacle-avoidance implementation provides:

- Sphere obstacle geometry
- Link-to-obstacle distance
- Signed clearance
- Collision detection
- Activation calculation
- Numerical clearance gradient
- Obstacle-avoidance direction

Main implementation:

```text
src/obstacle_avoidance.py
```

Run:

```bash
python examples/obstacle_avoidance_demo.py
```

Generated geometry example:

![Obstacle Avoidance Geometry](results/obstacle_avoidance_geometry.png)

---

# Obstacle-Aware Inverse Kinematics

The Cartesian IK controller was extended with an obstacle-avoidance secondary objective.

A simplified representation is:

```math
\Delta q_g
=
\Delta q_{\text{primary}}
+
N_\lambda
\left(
\Delta q_{\text{singularity}}
+
\Delta q_{\text{joint-limit}}
+
\Delta q_{\text{obstacle}}
\right)
```

The obstacle component attempts to increase robot-to-obstacle clearance while preserving the primary Cartesian task as much as possible.

Main implementation:

```text
src/obstacle_aware_ik.py
```

---

# Single-Obstacle Safety Filter

Secondary avoidance alone does not guarantee that a finite numerical IK step will satisfy a required clearance.

A separate safety layer was therefore implemented.

The single-obstacle safety filter includes:

- Linearized clearance checking
- Clearance-gradient based step correction
- Joint-step validation
- Sampling along the proposed generalized step
- Backtracking when required
- Rejection of unsafe initial configurations when safety enforcement is enabled

Main implementation:

```text
src/obstacle_safety_filter.py
```

The safety filter acts on the proposed controller update before the state is accepted.

---

# Single-Obstacle Safety-Constrained Comparison

Run:

```bash
python examples/obstacle_aware_comparison_demo.py
```

Generated results include:

### Clearance comparison

![Single-Obstacle Clearance Comparison](results/obstacle_aware_clearance_comparison.png)

### Cartesian error

![Single-Obstacle Error Comparison](results/obstacle_aware_error_comparison.png)

### Singularity metric

![Single-Obstacle Sigma Minimum](results/obstacle_aware_sigma_min.png)

### Joint-limit margin

![Single-Obstacle Joint-Limit Margin](results/obstacle_aware_joint_limit_margin.png)

### Obstacle activation

![Single-Obstacle Activation](results/obstacle_aware_activation.png)

### Cartesian path

![Single-Obstacle Path Comparison](results/obstacle_aware_path_comparison.png)

The comparison is designed to contrast a baseline IK trajectory with a safety-constrained obstacle-aware trajectory under the same target task.

---

# Multi-Obstacle Avoidance

The framework was extended from one obstacle to an arbitrary non-empty collection of static spherical obstacles.

For each obstacle, the implementation evaluates:

- Minimum robot clearance
- Closest robot link
- Obstacle activation
- Numerical clearance gradient
- Local avoidance direction
- Collision state

The controller also identifies the globally critical obstacle.

Main implementation:

```text
src/multi_obstacle_avoidance.py
```

---

# Combined Multi-Obstacle Avoidance Direction

Individual obstacle-avoidance directions are combined using activation-weighted normalized contributions.

Conceptually:

```math
g_{\text{obs}}
=
\sum_i
a_i
\hat{g}_i
```

where:

- `ai` is the activation level of obstacle `i`
- `ĝi` is the normalized local avoidance direction

This provides a practical combined secondary direction.

Because contributions from different obstacles can partially cancel, this combined vector is not treated as the final safety guarantee.

Harder step-level protection is handled by the dedicated safety filter.

---

# Multi-Obstacle Safety Filter

The multi-obstacle safety layer evaluates all obstacles during each proposed generalized IK update.

Main implementation:

```text
src/multi_obstacle_safety_filter.py
```

The filter includes:

- Per-obstacle linearized clearance constraints
- Sequential projection of the proposed generalized step
- Multiple projection passes
- Sampled nonlinear clearance validation
- Backtracking when necessary
- Global minimum-clearance monitoring

The sequential projection procedure is an approximate numerical feasibility mechanism rather than a quadratic-programming optimizer.

---

# Multi-Obstacle Safety-Constrained IK

The complete controller combines:

```text
Cartesian position tracking
+
Adaptive damping
+
Singularity avoidance
+
Joint-limit avoidance
+
Protected joint limits
+
Multi-obstacle avoidance
+
Multi-obstacle safety filtering
```

Main implementation:

```text
src/multi_obstacle_aware_ik.py
```

The controller records histories for:

- Cartesian error
- Minimum singular value
- Joint-limit margin
- Global obstacle clearance
- Critical obstacle
- Critical robot link
- Active-obstacle count
- Obstacle activity
- Safety-filter intervention
- Accepted step scale
- Backtracking activity
- Minimum sampled path clearance

---

# Representative Multi-Obstacle Experiment

Run:

```bash
python examples/multi_obstacle_comparison_demo.py
```

The current deterministic comparison uses two static spherical obstacles.

For the evaluated scenario:

| Metric | Baseline Multi-Objective IK | Multi-Obstacle Safety IK |
|---|---:|---:|
| Converged | Yes | Yes |
| Iterations | 15 | 22 |
| Final Cartesian error | 9.2871×10⁻⁵ m | 6.5308×10⁻⁵ m |
| Minimum obstacle clearance | -0.04855 m | 0.10561 m |
| Required safety distance | 0.10 m | 0.10 m |
| Maintained safety distance | No | Yes |
| Final collision state | — | False |

For the safety-constrained controller:

```text
Maximum simultaneously active obstacles: 2
Observed active-obstacle counts: [1, 2]
Safety-filter corrected steps: 9
Total controlled IK steps: 21
Backtracked steps: 0
Minimum accepted backtracking scale: 1.0
```

In this specific experiment, the linearized safety correction was sufficient and no backtracking reduction was required.

These values describe the current deterministic numerical experiment and are not universal performance guarantees.

---

# Multi-Obstacle Clearance Comparison

![Multi-Obstacle Clearance Comparison](results/multi_obstacle_clearance_comparison.png)

The baseline trajectory enters a negative signed-clearance region in this scenario.

The safety-constrained controller remains above the configured:

```text
0.10 m
```

sampled safety threshold.

---

# Multi-Obstacle Cartesian Error

![Multi-Obstacle Cartesian Error](results/multi_obstacle_error_comparison.png)

The safety-constrained controller follows a modified trajectory while still reaching the Cartesian position target within the configured numerical tolerance.

---

# Multi-Obstacle Activation History

![Multi-Obstacle Active Count](results/multi_obstacle_active_count.png)

The current experiment includes periods with one and two simultaneously active obstacles.

---

# Safety-Filter Intervention History

![Multi-Obstacle Safety Filter Activity](results/multi_obstacle_safety_filter_activity.png)

This figure identifies controller steps for which the multi-obstacle safety layer modified the proposed generalized update.

A correction does not necessarily imply backtracking.

In the current representative experiment:

```text
9 steps were safety-corrected
0 steps required backtracking
```

---

# Multi-Obstacle Singularity Metric

![Multi-Obstacle Sigma Minimum](results/multi_obstacle_sigma_min.png)

The singular-value history is monitored alongside Cartesian tracking and obstacle clearance.

The final singular-value result should be interpreted as scenario-specific rather than as evidence that obstacle avoidance universally improves singularity metrics.

---

# Multi-Obstacle 3D Path

![Multi-Obstacle Path Comparison](results/multi_obstacle_path_comparison.png)

The 3D comparison illustrates how the safety-constrained controller modifies the Cartesian path in response to the obstacle configuration.

---

# Safety Interpretation

The implemented obstacle safety mechanism combines:

```text
Linearized safety correction
+
Finite sampling along each accepted generalized step
+
Optional backtracking
```

This provides a practical discrete numerical safety check for the simulated trajectory.

However, the current implementation should **not** be interpreted as a formal proof of continuous-time collision avoidance between every pair of sampled configurations.

The current safety claims therefore refer to the implemented sampled numerical validation procedure.

---

# Static-Obstacle Scope

The current obstacle framework assumes:

- Static obstacles
- Spherical obstacle geometry
- Capsule-like robot-link approximation
- Numerically sampled finite IK steps

Dynamic obstacle prediction and continuous-time safety certification are outside the current implementation.

---

# Automated Testing

The project includes automated testing for the mathematical, kinematic, control, obstacle, and safety components.

Testing covers:

- DH transformation matrices
- Forward kinematics
- End-effector pose extraction
- Rotation matrix properties
- Translational Jacobian
- Angular Jacobian
- Geometric Jacobian
- Prismatic-joint Jacobian behavior
- Numerical finite-difference Jacobian validation
- Singular-value calculations
- Manipulability
- Trajectory generation
- Cartesian trajectory calculations
- Damped Least Squares IK
- Adaptive damping
- Singularity avoidance
- Joint-limit avoidance
- Multi-objective IK
- Protected joint limits
- Single-obstacle geometry
- Link-to-sphere clearance
- Obstacle activation
- Collision detection
- Obstacle-clearance gradients
- Single-obstacle safety filtering
- Obstacle-aware IK
- Single-obstacle integration behavior
- Multi-obstacle normalization
- Multi-obstacle clearance
- Combined obstacle avoidance
- Multi-obstacle safety filtering
- Multi-obstacle-aware IK
- Multi-obstacle integration and regression behavior

Run the complete test suite:

```bash
python -m pytest tests -v
```

Latest reported complete local validation:

```text
141 passed in 5.37s
```

Environment used for that local test run:

```text
Python 3.14.2
pytest 9.1.1
```

---

# Continuous Integration

GitHub Actions automatically executes the complete Python test suite on:

```text
push -> main
pull request -> main
```

The current CI workflow uses:

```text
Python 3.12
```

Workflow:

```text
.github/workflows/python-tests.yml
```

---

# Project Structure

```text
6DOF-RRRRRP-Kinematics-Singularity/
│
├── src/
│   ├── adaptive_damping.py
│   ├── inverse_kinematics.py
│   ├── jacobian.py
│   ├── joint_limit_avoidance.py
│   ├── kinematics.py
│   ├── multi_objective_ik.py
│   ├── multi_obstacle_avoidance.py
│   ├── multi_obstacle_aware_ik.py
│   ├── multi_obstacle_safety_filter.py
│   ├── obstacle_avoidance.py
│   ├── obstacle_aware_ik.py
│   ├── obstacle_safety_filter.py
│   ├── singularity.py
│   ├── singularity_avoidance.py
│   ├── trajectory.py
│   ├── visualization.py
│   └── workspace.py
│
├── examples/
│   ├── adaptive_damping_demo.py
│   ├── forward_kinematics_demo.py
│   ├── inverse_kinematics_demo.py
│   ├── joint_limit_avoidance_demo.py
│   ├── multi_objective_ik_demo.py
│   ├── multi_obstacle_comparison_demo.py
│   ├── obstacle_avoidance_demo.py
│   ├── obstacle_aware_comparison_demo.py
│   ├── singularity_avoidance_demo.py
│   ├── singularity_demo.py
│   ├── trajectory_analysis_demo.py
│   ├── trajectory_demo.py
│   ├── visualization_demo.py
│   └── workspace_demo.py
│
├── tests/
│   ├── test_adaptive_damping.py
│   ├── test_inverse_kinematics.py
│   ├── test_jacobian.py
│   ├── test_joint_limit_avoidance.py
│   ├── test_kinematics.py
│   ├── test_multi_objective_ik.py
│   ├── test_multi_obstacle_avoidance.py
│   ├── test_multi_obstacle_aware_ik.py
│   ├── test_multi_obstacle_integration.py
│   ├── test_multi_obstacle_safety_filter.py
│   ├── test_obstacle_avoidance.py
│   ├── test_obstacle_aware_ik.py
│   ├── test_obstacle_safety_filter.py
│   ├── test_obstacle_safety_integration.py
│   ├── test_singularity.py
│   ├── test_singularity_avoidance.py
│   └── test_trajectory.py
│
├── results/
│   ├── robot_configuration.png
│   ├── workspace.png
│   ├── sigma_min_vs_theta3.png
│   ├── condition_number_vs_theta3.png
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
│   ├── multi_objective_path_comparison.png
│   ├── obstacle_avoidance_geometry.png
│   ├── obstacle_aware_clearance_comparison.png
│   ├── obstacle_aware_error_comparison.png
│   ├── obstacle_aware_sigma_min.png
│   ├── obstacle_aware_joint_limit_margin.png
│   ├── obstacle_aware_activation.png
│   ├── obstacle_aware_path_comparison.png
│   ├── multi_obstacle_clearance_comparison.png
│   ├── multi_obstacle_error_comparison.png
│   ├── multi_obstacle_active_count.png
│   ├── multi_obstacle_safety_filter_activity.png
│   ├── multi_obstacle_safety_filter_scale.png
│   ├── multi_obstacle_sigma_min.png
│   └── multi_obstacle_path_comparison.png
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

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Current requirements:

```text
numpy>=1.24
pytest>=8.0
matplotlib>=3.7
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

Multi-objective IK:

```bash
python examples/multi_objective_ik_demo.py
```

Obstacle geometry and avoidance:

```bash
python examples/obstacle_avoidance_demo.py
```

Single-obstacle safety comparison:

```bash
python examples/obstacle_aware_comparison_demo.py
```

Multi-obstacle safety comparison:

```bash
python examples/multi_obstacle_comparison_demo.py
```

Run all tests:

```bash
python -m pytest tests -v
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
| Damped Task-Priority Secondary Objectives | ✅ Implemented |
| Multi-Objective IK | ✅ Implemented |
| Protected Joint Safety Margin | ✅ Implemented |
| Static Sphere Obstacle Modeling | ✅ Implemented |
| Robot-to-Obstacle Clearance | ✅ Implemented |
| Collision Detection | ✅ Implemented |
| Single-Obstacle Avoidance | ✅ Implemented |
| Single-Obstacle Safety Filter | ✅ Implemented |
| Safety-Constrained Obstacle-Aware IK | ✅ Implemented |
| Multi-Obstacle Avoidance | ✅ Implemented |
| Multi-Obstacle Safety Filter | ✅ Implemented |
| Multi-Obstacle Safety-Constrained IK | ✅ Implemented |
| Sampled Nonlinear Step Validation | ✅ Implemented |
| Backtracking Safety Logic | ✅ Implemented |
| Integration / Regression Testing | ✅ Implemented |
| Automated Unit Testing | ✅ Implemented |
| GitHub Actions CI | ✅ Implemented |
| Global Path Planning | ⏳ Future Extension |
| Dynamic Obstacles | ⏳ Future Extension |
| Continuous-Time Collision Certification | ⏳ Future Extension |
| QP / CBF Safety Control | ⏳ Future Extension |
| MATLAB Validation | ⏳ Future Extension |
| Robotics Toolbox Comparison | ⏳ Future Extension |
| Hardware / Arduino Integration | ⏳ Future Extension |

---

# Important Modeling Notes

## 1. Position-Control Scope

The manipulator contains six joints, but the current IK and avoidance experiments focus primarily on the three-dimensional Cartesian position task.

Therefore, the principal singularity metric used by these controllers is based on:

```math
J_v \in \mathbb{R}^{3\times6}
```

A regular position-control configuration requires:

```math
\operatorname{rank}(J_v)=3
```

This distinction is important when interpreting rank, singular values, manipulability, and controller performance.

---

## 2. Damped Task-Priority Projection

The matrix:

```math
N_\lambda
=
I-J_\lambda^\#J_v
```

uses a damped pseudoinverse.

It is therefore not generally an exact orthogonal null-space projector.

The project describes it as a **damped task-priority projection**.

---

## 3. Mixed Generalized Units

The generalized state contains:

```text
revolute coordinates -> radians
prismatic coordinate -> meters
```

Numerical gradients that operate across all generalized coordinates are consequently unit-sensitive.

This is particularly relevant for:

- Singularity gradients
- Joint-limit gradients
- Obstacle-clearance gradients
- Gradient normalization

---

## 4. Minimum-Clearance Nonsmoothness

The global obstacle-clearance function is obtained by selecting the minimum clearance across robot links and obstacles.

The identity of the closest link or critical obstacle can change during motion.

The resulting minimum-clearance function can therefore be locally nonsmooth at switching points.

---

## 5. Multi-Obstacle Projection

The current multi-obstacle safety filter uses sequential projections onto locally linearized clearance constraints.

This is a practical approximate procedure.

It is not equivalent to solving an exact global constrained optimization or quadratic programming problem.

---

## 6. Sampled Safety Validation

Accepted finite IK steps are evaluated using discrete samples along the proposed generalized motion.

The implementation therefore provides sampled numerical safety verification.

It does not constitute a formal continuous-time collision-free proof.

---

## 7. Unsafe Initial States

When safety enforcement is enabled, the current controller expects the initial configuration to satisfy the configured obstacle safety requirement.

The implementation is not currently designed as a dedicated recovery controller for initially unsafe states.

---

# Research Scope

This repository is intended as a modular research platform for studying:

- Robotic manipulator kinematics
- Differential kinematics
- Jacobian-based control
- Numerical inverse kinematics
- Singular-value analysis
- Translational singularity detection
- Singularity avoidance
- Redundancy resolution
- Joint-limit avoidance
- Damped task-priority control
- Multi-objective control
- Static obstacle avoidance
- Numerical collision checking
- Clearance-based safety constraints
- Safety-filtered inverse kinematics
- Multi-obstacle interaction
- Constraint-aware robotic control
- Trajectory analysis
- Numerical robotics validation

---

# Future Research Development

Potential future extensions include:

- Dynamic obstacle tracking
- Moving-obstacle prediction
- Global Cartesian or configuration-space path planning
- Trajectory optimization
- Quadratic-programming based safety constraints
- Control Barrier Functions
- Exact hierarchical task-priority optimization
- Weighted pseudoinverse methods
- Joint-velocity constraints
- Joint-acceleration constraints
- Acceleration analysis
- Jerk analysis
- Continuous collision detection
- More general obstacle geometry
- MATLAB cross-validation
- MATLAB Robotics Toolbox comparison
- Real-time controller implementation
- Python–Arduino communication
- Sensor-based obstacle detection
- Experimental robotic validation

---

# Validation Philosophy

The project follows a progressive validation strategy:

```text
Forward kinematics
        ↓
Analytical Jacobian
        ↓
Finite-difference validation
        ↓
Singularity metrics
        ↓
Inverse kinematics
        ↓
Adaptive damping
        ↓
Secondary objectives
        ↓
Multi-objective IK
        ↓
Single-obstacle avoidance
        ↓
Single-obstacle safety filtering
        ↓
Multi-obstacle avoidance
        ↓
Multi-obstacle safety filtering
        ↓
Integration and regression testing
```

This structure allows individual mathematical components to be tested before they are combined into higher-level controllers.

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
- Safety-constrained inverse kinematics
- Constraint-aware robotic control
- Python and MATLAB robotic modeling
- Experimental robotic control
