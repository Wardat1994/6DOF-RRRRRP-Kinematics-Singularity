import sys
from pathlib import Path

import numpy as np


# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.kinematics import get_end_effector_pose


# =====================================================
# TEST ROBOT CONFIGURATION
# =====================================================

# Revolute joints [degrees -> radians]
q = np.radians([
    -90,
    0,
    30,
    -15,
    0
])

# Prismatic joint displacement [m]
d6 = 0.20


# =====================================================
# FORWARD KINEMATICS
# =====================================================

T06, position, rotation = get_end_effector_pose(q, d6)


# =====================================================
# DISPLAY RESULTS
# =====================================================

np.set_printoptions(
    precision=5,
    suppress=True
)

print("=" * 60)
print("6-DOF RRRRRP ROBOT - FORWARD KINEMATICS")
print("=" * 60)

print("\nJoint Configuration:")

for i, angle in enumerate(np.degrees(q), start=1):
    print(f"theta{i} = {angle:.2f} deg")

print(f"d6 = {d6:.3f} m")


print("\nT06 - End-Effector Transformation Matrix:")
print(T06)


print("\nEnd-Effector Position [m]:")

print(f"x = {position[0]:.5f} m")
print(f"y = {position[1]:.5f} m")
print(f"z = {position[2]:.5f} m")


print("\nEnd-Effector Rotation Matrix R06:")
print(rotation)


# =====================================================
# ROTATION MATRIX CHECK
# =====================================================

orthogonality_error = np.linalg.norm(
    rotation.T @ rotation - np.eye(3)
)

determinant = np.linalg.det(rotation)


print("\nRotation Matrix Validation:")

print(
    f"||R^T R - I|| = "
    f"{orthogonality_error:.3e}"
)

print(
    f"det(R) = "
    f"{determinant:.6f}"
)

print("\nForward kinematics calculation completed successfully.")
