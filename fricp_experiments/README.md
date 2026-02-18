# FRICP Experiment Framework (Python)

This directory provides a Python framework to benchmark the **Fast and Robust Iterative Closest Point** algorithm. It automates the generation of synthetic datasets, executes the C++ solver, and analyzes performance across various conditions (Noise, Outliers, etc.).

## 🚀 Getting Started

If you have just cloned this repository:

1.  **Build the C++ Core**:
    Ensure you have complied the main project. The framework expects the executable to be located at:
    `build/Release/FRICP.exe` (on Windows) or `build/FRICP` (on Linux, update `config.py` accordingly).
    
    > **⚠️ CRITICAL NOTE**: This framework relies on patched versions of `main.cpp` AND `FRICP.h` (included in this repo). You MUST compile using these specific files, as they include fixes for exporting convergence statistics (iterations & energy) that are missing in the original author's code.

2.  **Install Python Dependencies**:
    ```bash
    pip install numpy pandas scipy matplotlib
    ```

3.  **Run Experiments**:
    ```bash
    python fricp_experiments/main.py
    ```

## 📁 Repository Structure

*   **`main.py`**: The entry point. Manages the experiment loops and metric collection.
*   **`config.py`**: Configuration hub. You can adjust:
    *   `SKIP_EXISTING`: If `True`, the script won't re-run the C++ binary if results exist. It correctly handles persistent ground truths (see Reproducibility).
    *   `VERBOSE`: If `True`, prints detailed rotation/translation errors for each run and warns if a method returns an `IDENTITY` matrix.
    *   `RUN_MIXED_EXPERIMENT`: Set to `True` to enable the 2D stress test.
...
*   **`viz/`**: Generates comparative plots. Includes **horizontal jitter** to ensure multiple overlapping curves (e.g., when methods yield identical errors) remain visible.

## 🔄 Reproducibility & Stableness
To solve "drifting" metrics when re-running experiments:
- **RNG Seeding**: `main.py` uses `np.random.seed(42)` for fixed perturbations.
- **Persistent Ground Truth**: The ground truth transform used for an experiment is saved as `results/*_gt_transform.txt`. If `SKIP_EXISTING` is used, the script loads this specific matrix to ensure metrics are computed against the correct reference, even if the RNG state changed.

## 📊 Understanding the results

The framework generates plots in the `results/` directory:
1.  **`noise_plot.png`**: Evaluation under increasing noise.
2.  **`outlier_plot.png`**: Evaluation under increasing outliers.
3.  **`mixed_heatmap.png`**: Heatmaps for combined Noise vs Outliers.

### Why do some curves have jitter?
If multiple methods fail in the same way (e.g., they all return Identity), they would overlap perfectly. We add a small horizontal jitter to the x-axis values in the plots so you can see that all methods are actually represented.

### Why two lines per method?
- **Solid line (FRICP)**: Error from the matrix exported by the C++.
- **Dashed line (Kabsch)**: Error from an estimation based on the actual point positions.

## 📁 Data Folder Note
The top-level `data/` folder contains the base models:
- **`target.ply`**: The reference model used to generate experiment datasets.
- **`source.ply`**: Original source model from the project.
- **`res_*/` directories**: These are typically legacy results from direct C++ executions. For this Python framework, all fresh results are consolidated in `fricp_experiments/results/`.

## 👁️ Visual Validation
Point clouds are saved in `results/`. Open `target.ply` and `mXreg_pc.ply` in **CloudCompare** to visually inspect the alignment.
