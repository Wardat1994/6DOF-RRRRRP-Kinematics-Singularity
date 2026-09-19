import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization import plot_robot


# =====================================================
# ROBOT CONFIGURATION
# =====================================================

q = np.radians([
    -90,
    0,
    30,
    -15,
    0
])

d6 = 0.20


# =====================================================
# 3D VISUALIZATION
# =====================================================

fig, ax = plot_robot(q, d6)

plt.show()
