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
    python fricp_experiments/main.py --trials 1
    ```

    To run with multiple trials per condition (e.g., for averaging convergence stats):
    ```bash
    python fricp_experiments/main.py --trials 20
    ```

## 📂 Repository Structure & Configuration

*   **`main.py`**: The entry point. Manages the experiment loops and metric collection.
*   **`config.py`**: Configuration hub. You can adjust:
    *   `SKIP_EXISTING`: If `True`, the script won't re-run the C++ binary if result files exist. Correctly handles persistent ground truths (see Reproducibility).
    *   `VERBOSE`: If `True`, prints detailed results for each condition even in multi-trial runs.
    *   `RUN_MIXED_EXPERIMENT`: Set to `True` (default) to enable the 2D stress test grid (Noise vs Outliers).
    *   `methods`: List of IDs to test (0: ICP, 3: Robust ICP, 6: Sparse ICP).
*   **`fricp/runner.py`**: Python wrapper for the C++ executable. Handles stats parsing of `stdout` and caches results.
*   **`perturb/`**: Contains the logic for adding Gaussian noise, substituting outliers, and applying random rigid transforms.
*   **`metrics/`**: Implementation of `Rotation Error`, `Translation Error`, `Comparable RMS`, and `Welsch Energy`.
*   **`viz/`**: Plotting scripts. Generates 1D curves with error bars and 2D heatmaps.

## 📊 Tracked Metrics & Results Interpretation

The framework tracks multiple metrics to provide a 360° view of performance. Results include the **Mean** (average performance) and **Standard Deviation** (`_std`, representing stability).

### 1. Performance Metrics (`*_plot.png`)
*   **Rotation Error (degrees)**: Angular difference between estimate and Ground Truth. 0.0 is perfect.
*   **Comparable Metric Reference (Dashed Lines)**: In most plots, you will see a **Dashed Line (Kabsch reference)**. This represents the *optimal* registration possible (computed on 100% true inliers with perfect correspondences). It represents the lower-bound error for a given noise level.
*   **Comparable RMS Error**: Euclidean distance between registered source and target. 
    *   *Note*: In outlier tests, this is calculated **only on true inliers** for a fair comparison between methods that reject outliers and those that don't.
*   **Execution Time (s)**: Total CPU time. Vital for comparing **FRICP** vs **Sparse ICP**.

### 2. Convergence Metrics (`*_convergence.png`)
*   **Iterations**: Steps taken until convergence. Faster is better.
*   **Internal Energy**: Final value of the method's objective function. ⚠️ **Non-comparable** between different ICP types (ICP vs Robust ICP) as they minimize different kernels.

### 📊 Plot Characteristics
*   **Horizontal Jitter**: If multiple methods have identical errors (e.g., they all fail and return an Identity matrix), they would overlap perfectly. We add a small horizontal shift (jitter) so every method's marker remains visible.
*   **Error Bars**: Shaded areas or vertical bars show the **Standard Deviation**. Small bars indicate high **stability**; large bars mean the method depends on the luck of the random seed.

## 🔄 Reproducibility & Ground Truths

*   **Persistent Ground Truth**: Ground truth transforms are saved as `*_gt_transform.txt` in `results/`. 
    *   **If `--trials 1`**: The script loads these files to ensure you always test against the *exact same* perturbation, even if you restart the script.
    *   **If `--trials > N`**: The script generates new, seed-controlled transforms per trial for statistical significance.
*   **RNG Seeding**: `main.py` uses fixed seeds per condition (`42 + index`) to ensure data generation is deterministic.

### 3. Mixed Grid Analysis (Heatmaps)
*   **`mixed_heatmap.png`**: Displays Rotation Error across a 2D grid of Noise Sigma and Outlier Ratio. This identifies the "breakdown point" where algorithms start to fail.
*   **`mixed_convergence_heatmap.png`**: Shows how computational effort (iterations) increases as the environment becomes more challenging.

## 👁️ Visual Validation
Open `target.ply` and `method_X/mXreg_pc.ply` in **CloudCompare** to visually inspect alignment quality. Results are organized in folders named after the perturbation (e.g., `noise_0.02_t0/`).
