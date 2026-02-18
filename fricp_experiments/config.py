import os

# Experiment configurations
SKIP_EXISTING = False  # If True, won't rerun FRICP if output files exist
RUN_MIXED_EXPERIMENT = True  # Set to True to run the 2D Noise/Outlier grid
VERBOSE = True  # If True, prints more debug info during experiments

# Get the directory of the current script (config.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Base configuration
BASE_CONFIG = {
    "data_path": os.path.join(PROJECT_ROOT, "data", "target.ply"),
    "output_root": os.path.join(BASE_DIR, "results"),
    "exe_path": os.path.join(PROJECT_ROOT, "build", "Release", "FRICP.exe")
}

# Noise experiment grid (1D)
noise_grid = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]

# Outlier experiment grid (1D)
outlier_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

# Mixed experiment grids (2D) - Keep small for performance
noise_grid_mixed = [0.0, 0.005, 0.01, 0.02]
outlier_grid_mixed = [0.0, 0.1, 0.3, 0.5]

# Methods to test
# 0: ICP, 1: AA-ICP, 2: Fast ICP, 3: Robust ICP, 4: ICP P2Pl
methods = [0, 3, 6] # Standard ICP (0), Robust ICP (3), and Sparse ICP (6)
method_names = {
    0: "Standard ICP",
    1: "AA-ICP",
    2: "Fast ICP",
    3: "Robust ICP",
    4: "ICP (Pt-to-Pl)",
    6: "Sparse ICP"
}
