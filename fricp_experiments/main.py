import os
import numpy as np
import pandas as pd
import argparse
from io_pc.ply import read_ply, write_ply
from perturb.noise import add_gaussian_noise
from perturb.outliers import substitute_outliers
from perturb.transform import apply_transform, get_random_rotation, get_random_translation
from fricp.runner import FRICPRunner
from metrics.transform_error import compute_transform_error, estimate_kabsch
from metrics.reconstruction import compute_rms, compute_welsch_energy
from viz.plots import plot_experiment_results, plot_mixed_results
from viz.convergence import plot_convergence_metrics, plot_iteration_heatmap
import config

# Seed for reproducibility - reset per trial in the loops
np.random.seed(42)

def get_args():
    parser = argparse.ArgumentParser(description="FRICP Benchmark Suite")
    parser.add_argument("--trials", type=int, default=1, help="Number of trials per point (set to 20 for paper results)")
    return parser.parse_args()

def run_experiment_mixed(n_trials=1):
    print(f"\n--- Mixed Stress Test: Noise + Outliers ({n_trials} trials) ---")
    target_xyz, target_nor = read_ply(config.BASE_CONFIG["data_path"])
    runner = FRICPRunner(config.BASE_CONFIG["exe_path"])
    results = []

    T_gt_path = os.path.join(config.BASE_CONFIG["output_root"], "mixed_gt_transform.txt")
    
    # Handle ground truth: load existing if single run, or generate per trial
    if config.SKIP_EXISTING and os.path.exists(T_gt_path):
        T_gt = np.loadtxt(T_gt_path)
        T_gt_inv = np.linalg.inv(T_gt)
        R_gt = T_gt_inv[:3, :3]
        t_gt = T_gt_inv[:3, 3]
    else:
        # Initial random pose for the 'single-trial' consistency
        R_gt = get_random_rotation(max_angle_deg=12)
        t_gt = get_random_translation(max_dist=0.08)
        T_gt_inv = np.eye(4)
        T_gt_inv[:3, :3], T_gt_inv[:3, 3] = R_gt, t_gt
        T_gt = np.linalg.inv(T_gt_inv)
        np.savetxt(T_gt_path, T_gt)

    total_iters = len(config.noise_grid_mixed) * len(config.outlier_grid_mixed)
    current_iter = 0

    for sigma in config.noise_grid_mixed:
        for ratio in config.outlier_grid_mixed:
            current_iter += 1
            print(f"  Step {current_iter}/{total_iters}: sigma={sigma}, outliers={ratio*100}%")
            
            # Accumulators for stats
            metrics_sum = {m: {"time": [], "iters": [], "final_energy": [], "rot_err_fricp": [], "trans_err_fricp": [], "rot_err_kabsch": [], "rms": [], "energy": []} for m in config.methods}
            
            for trial in range(n_trials):
                if n_trials > 1:
                    # New random transform per trial for statistical significance
                    seed = 42 + int(ratio*1000) + int(sigma*10000) + trial
                    np.random.seed(seed)
                    R_trial = get_random_rotation(max_angle_deg=15)
                    t_trial = get_random_translation(max_dist=0.1)
                    T_trial_inv = np.eye(4)
                    T_trial_inv[:3, :3], T_trial_inv[:3, 3] = R_trial, t_trial
                    T_curr = np.linalg.inv(T_trial_inv)
                    R_curr, t_curr = R_trial, t_trial
                else:
                    # Persistent single transformation for reproducibility/visual comparison
                    R_curr, t_curr, T_curr = R_gt, t_gt, T_gt

                # 1. Perturb
                source_init_xyz, source_init_nor = apply_transform(target_xyz, R_curr, t_curr, target_nor)
                source_perturbed_xyz, source_perturbed_nor, inlier_mask = substitute_outliers(source_init_xyz, source_init_nor, ratio=ratio)
                source_perturbed_xyz, source_perturbed_nor = add_gaussian_noise(source_perturbed_xyz, source_perturbed_nor, sigma=sigma)
                
                # 2. Save
                work_dir = os.path.join(config.BASE_CONFIG["output_root"], f"mixed_s{sigma}_r{ratio}_t{trial}")
                if not os.path.exists(work_dir): os.makedirs(work_dir)
                
                target_path = os.path.join(work_dir, "target.ply")
                source_path = os.path.join(work_dir, "source.ply")
                write_ply(target_path, target_xyz, target_nor)
                write_ply(source_path, source_perturbed_xyz, source_perturbed_nor)
                
                for m in config.methods:
                    method_name = config.method_names.get(m, f"Method_{m}")
                    res_dir = os.path.join(work_dir, f"method_{m}")
                    
                    skip_if_exists = config.SKIP_EXISTING if n_trials == 1 else False
                    res = runner.run(target_path, source_path, res_dir, method=m, skip_if_exists=skip_if_exists)
                    
                    T_fricp = res["transformation"]
                    rot_err_fricp, trans_err_fricp = compute_transform_error(T_fricp, T_curr)
                    
                    # Optimal registration on inliers only (reference lower bound)
                    T_kabsch = estimate_kabsch(source_perturbed_xyz[inlier_mask], target_xyz[inlier_mask])
                    rot_err_kabsch, _ = compute_transform_error(T_kabsch, T_curr)

                    metrics_sum[m]["time"].append(res["time"])
                    metrics_sum[m]["iters"].append(res["iters"] if not np.isnan(res["iters"]) else 0)
                    metrics_sum[m]["rot_err_fricp"].append(rot_err_fricp)
                    metrics_sum[m]["trans_err_fricp"].append(trans_err_fricp)
                    metrics_sum[m]["rot_err_kabsch"].append(rot_err_kabsch)
                    metrics_sum[m]["final_energy"].append(res["final_energy"] if not np.isnan(res["final_energy"]) else 0)
                    
                    # Add comparable metrics for Mixed too
                    reg_path = os.path.join(res_dir, f"m{m}reg_pc.ply")
                    if os.path.exists(reg_path):
                        source_reg_xyz, _ = read_ply(reg_path)
                        source_reg_xyz_clean = source_reg_xyz[inlier_mask]
                        metrics_sum[m]["rms"].append(compute_rms(source_reg_xyz_clean, target_xyz[inlier_mask]))
                        metrics_sum[m]["energy"].append(compute_welsch_energy(source_reg_xyz_clean, target_xyz[inlier_mask]))
                    else:
                        metrics_sum[m]["rms"].append(np.nan)
                        metrics_sum[m]["energy"].append(np.nan)

                    if n_trials == 1 and config.VERBOSE:
                         print(f"      [DEBUG] {method_name}: rot_err={rot_err_fricp:.2f}, trans_err={trans_err_fricp:.4f}")
                         if T_fricp is not None and np.allclose(T_fricp, np.eye(4), atol=1e-5):
                              print(f"      [WARNING] {method_name} returned IDENTITY matrix.")

            # Average and append
            for m in config.methods:
                method_name = config.method_names.get(m, f"Method_{m}")
                
                # Calculate mean and std for all metrics
                stats = {}
                for k, v in metrics_sum[m].items():
                    stats[k] = np.mean(v)
                    stats[f"{k}_std"] = np.std(v)
                
                res_item = {
                    "noise_sigma": sigma,
                    "outlier_ratio": ratio,
                    "method": method_name
                }
                res_item.update(stats)
                results.append(res_item)
                
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(config.BASE_CONFIG["output_root"], "mixed_results.csv"), index=False)
    
    plot_mixed_results(
        df, 
        title='Resistance to Mixed Noise & Outliers (Rotation Error)', 
        save_path=os.path.join(config.BASE_CONFIG["output_root"], "mixed_heatmap.png")
    )
    
    plot_iteration_heatmap(
        df,
        title='Convergence Efficiency (Total Iterations)',
        save_path=os.path.join(config.BASE_CONFIG["output_root"], "mixed_convergence_heatmap.png")
    )
    return df

def run_experiment_noise(n_trials=1):
    print(f"--- Starting Noise Resistance Experiment ({n_trials} trials) ---")
    target_xyz, target_nor = read_ply(config.BASE_CONFIG["data_path"])
    runner = FRICPRunner(config.BASE_CONFIG["exe_path"])
    results = []

    T_gt_path = os.path.join(config.BASE_CONFIG["output_root"], "noise_gt_transform.txt")
    if config.SKIP_EXISTING and os.path.exists(T_gt_path):
        T_gt = np.loadtxt(T_gt_path)
        T_gt_inv = np.linalg.inv(T_gt)
        R_gt = T_gt_inv[:3, :3]
        t_gt = T_gt_inv[:3, 3]
    else:
        # Ground truth transform
        R_gt = get_random_rotation(max_angle_deg=15)
        t_gt = get_random_translation(max_dist=0.1)
        
        # Ground truth matrix (maps target to source_init)
        T_gt_inv = np.eye(4)
        T_gt_inv[:3, :3] = R_gt
        T_gt_inv[:3, 3] = t_gt
        
        # T_gt is what we want to estimate (maps source_init to target)
        T_gt = np.linalg.inv(T_gt_inv)
        np.savetxt(T_gt_path, T_gt)

    for sigma in config.noise_grid:
        print(f"  Testing noise sigma: {sigma}")
        
        metrics_sum = {m: {"time": [], "iters": [], "final_energy": [], "rot_err_fricp": [], "trans_err_fricp": [], "rot_err_kabsch": [], "trans_err_kabsch": [], "matrix_verify_err": [], "rms": [], "energy": []} for m in config.methods}
        
        for trial in range(n_trials):
            if n_trials > 1:
                seed = 42 + int(sigma*10000) + trial
                np.random.seed(seed)
                R_trial = get_random_rotation(max_angle_deg=15)
                t_trial = get_random_translation(max_dist=0.1)
                T_trial_inv = np.eye(4)
                T_trial_inv[:3, :3], T_trial_inv[:3, 3] = R_trial, t_trial
                T_curr = np.linalg.inv(T_trial_inv)
                R_curr, t_curr = R_trial, t_trial
            else:
                R_curr, t_curr, T_curr = R_gt, t_gt, T_gt
            
            source_init_xyz, source_init_nor = apply_transform(target_xyz, R_curr, t_curr, target_nor)
            source_perturbed_xyz, source_perturbed_nor = add_gaussian_noise(source_init_xyz, source_init_nor, sigma=sigma)
            
            work_dir = os.path.join(config.BASE_CONFIG["output_root"], f"noise_{sigma}_t{trial}")
            if not os.path.exists(work_dir): os.makedirs(work_dir)
            
            target_path = os.path.join(work_dir, "target.ply")
            source_path = os.path.join(work_dir, "source.ply")
            write_ply(target_path, target_xyz, target_nor)
            write_ply(source_path, source_perturbed_xyz, source_perturbed_nor)
            
            skip_if_exists = config.SKIP_EXISTING if n_trials == 1 else False
            
            for m in config.methods:
                if n_trials == 1:
                     method_name = config.method_names.get(m, f"Method_{m}")
                     # Print status only for single trial to avoid spam
                     pass

                res_dir = os.path.join(work_dir, f"method_{m}")
                res = runner.run(target_path, source_path, res_dir, method=m, skip_if_exists=skip_if_exists)
                
                if n_trials == 1:
                     if "Skipped" in res["stdout"]:
                          print(f"    - {config.method_names.get(m)}: [LOADED]")
                     else:
                          print(f"    - {config.method_names.get(m)}: [DONE] ({res['time']:.2f}s)")

                reg_path = os.path.join(res_dir, f"m{m}reg_pc.ply")
                T_fricp = res["transformation"]
                T_kabsch = None
                
                if os.path.exists(reg_path):
                    source_reg_xyz, _ = read_ply(reg_path)
                    T_kabsch = estimate_kabsch(source_perturbed_xyz, target_xyz)
                    T_fricp_verify = estimate_kabsch(source_perturbed_xyz, source_reg_xyz)
                else:
                    source_reg_xyz = None
                    T_fricp_verify = None

                rot_err_fricp, trans_err_fricp = compute_transform_error(T_fricp, T_curr)
                rot_err_kabsch, trans_err_kabsch = compute_transform_error(T_kabsch, T_curr)
                rot_err_verify, _ = compute_transform_error(T_fricp, T_fricp_verify)
                
                if source_reg_xyz is not None:
                     rms = compute_rms(source_reg_xyz, target_xyz)
                     energy = compute_welsch_energy(source_reg_xyz, target_xyz)
                else:
                     rms, energy = np.nan, np.nan
                
                metrics_sum[m]["time"].append(res["time"])
                metrics_sum[m]["iters"].append(res["iters"] if not np.isnan(res["iters"]) else 0)
                metrics_sum[m]["final_energy"].append(res["final_energy"] if not np.isnan(res["final_energy"]) else 0)
                metrics_sum[m]["rot_err_fricp"].append(rot_err_fricp)
                metrics_sum[m]["trans_err_fricp"].append(trans_err_fricp)
                metrics_sum[m]["rot_err_kabsch"].append(rot_err_kabsch if T_kabsch is not None else 0)
                metrics_sum[m]["trans_err_kabsch"].append(trans_err_kabsch if T_kabsch is not None else 0)
                metrics_sum[m]["matrix_verify_err"].append(rot_err_verify if source_reg_xyz is not None else 0)
                metrics_sum[m]["rms"].append(rms if not np.isnan(rms) else 0)
                metrics_sum[m]["energy"].append(energy if not np.isnan(energy) else 0)

                if n_trials == 1 and config.VERBOSE:
                     print(f"      [DEBUG] {method_name}: rot_err={rot_err_fricp:.2f} (Kabsch={rot_err_kabsch:.2f}), verify_err={rot_err_verify:.4f}")
                     if T_fricp is not None and np.allclose(T_fricp, np.eye(4), atol=1e-5):
                          print(f"      [WARNING] {method_name} returned IDENTITY matrix.")

            # Average and append
            for m in config.methods:
                method_name = config.method_names.get(m, f"Method_{m}")
                
                 # Calculate mean and std for all metrics
                stats = {}
                for k, v in metrics_sum[m].items():
                    stats[k] = np.mean(v)
                    stats[f"{k}_std"] = np.std(v)
                
                res_item = {
                    "noise_sigma": sigma,
                    "method": method_name
                }
                res_item.update(stats)
                results.append(res_item)


    df = pd.DataFrame(results)
    df.to_csv(os.path.join(config.BASE_CONFIG["output_root"], "noise_results.csv"), index=False)
    
    plot_experiment_results(
        df, 
        x_axis='noise_sigma', 
        title='Resistance to Noise', 
        save_path=os.path.join(config.BASE_CONFIG["output_root"], "noise_plot.png")
    )

    plot_convergence_metrics(
        df,
        x_axis='noise_sigma',
        title='Convergence Performance vs Noise',
        save_path=os.path.join(config.BASE_CONFIG["output_root"], "noise_convergence.png")
    )
    return df

def run_experiment_outliers(n_trials=1):
    print(f"\n--- Starting Outlier Resistance Experiment ({n_trials} trials) ---")
    target_xyz, target_nor = read_ply(config.BASE_CONFIG["data_path"])
    runner = FRICPRunner(config.BASE_CONFIG["exe_path"])
    results = []

    T_gt_path = os.path.join(config.BASE_CONFIG["output_root"], "outlier_gt_transform.txt")
    if config.SKIP_EXISTING and os.path.exists(T_gt_path):
        T_gt = np.loadtxt(T_gt_path)
        T_gt_inv = np.linalg.inv(T_gt)
        R_gt = T_gt_inv[:3, :3]
        t_gt = T_gt_inv[:3, 3]
    else:
        # Use same max angle/dist as in report for consistency
        R_gt = get_random_rotation(max_angle_deg=15)
        t_gt = get_random_translation(max_dist=0.1)
        T_gt_inv = np.eye(4)
        T_gt_inv[:3, :3], T_gt_inv[:3, 3] = R_gt, t_gt
        T_gt = np.linalg.inv(T_gt_inv)
        if not os.path.exists(config.BASE_CONFIG["output_root"]): os.makedirs(config.BASE_CONFIG["output_root"])
        np.savetxt(T_gt_path, T_gt)

    for ratio in config.outlier_grid:
        print(f"  Testing outlier ratio: {ratio*100}%")
        
        metrics_sum = {m: {"time": [], "iters": [], "final_energy": [], "rot_err_fricp": [], "trans_err_fricp": [], "rot_err_kabsch": [], "trans_err_kabsch": [], "rms": [], "energy": []} for m in config.methods}
        
        for trial in range(n_trials):
            if n_trials > 1:
                seed = 42 + int(ratio*1000) + trial
                np.random.seed(seed)
                R_trial = get_random_rotation(max_angle_deg=15)
                t_trial = get_random_translation(max_dist=0.1)
                T_trial_inv = np.eye(4)
                T_trial_inv[:3, :3], T_trial_inv[:3, 3] = R_trial, t_trial
                T_curr = np.linalg.inv(T_trial_inv)
                R_curr, t_curr = R_trial, t_trial
            else:
                R_curr, t_curr, T_curr = R_gt, t_gt, T_gt
            
            source_init_xyz, source_init_nor = apply_transform(target_xyz, R_curr, t_curr, target_nor)
            source_perturbed_xyz, source_perturbed_nor, inlier_mask = substitute_outliers(source_init_xyz, source_init_nor, ratio=ratio)
            
            work_dir = os.path.join(config.BASE_CONFIG["output_root"], f"outliers_{ratio}_t{trial}")
            if not os.path.exists(work_dir): os.makedirs(work_dir)
            
            target_path = os.path.join(work_dir, "target.ply")
            source_path = os.path.join(work_dir, "source.ply")
            write_ply(target_path, target_xyz, target_nor)
            write_ply(source_path, source_perturbed_xyz, source_perturbed_nor)
            
            skip_if_exists = config.SKIP_EXISTING if n_trials == 1 else False
            
            for m in config.methods:
                if n_trials == 1:
                     method_name = config.method_names.get(m, f"Method_{m}")

                res_dir = os.path.join(work_dir, f"method_{m}")
                res = runner.run(target_path, source_path, res_dir, method=m, skip_if_exists=skip_if_exists)
                
                if n_trials == 1:
                     if "Skipped" in res["stdout"]:
                          print(f"    - {config.method_names.get(m)}: [LOADED]")
                     else:
                          print(f"    - {config.method_names.get(m)}: [DONE] ({res['time']:.2f}s)")

                reg_path = os.path.join(res_dir, f"m{m}reg_pc.ply")
                T_fricp = res["transformation"]
                T_kabsch = None
                
                if os.path.exists(reg_path):
                    source_reg_xyz, _ = read_ply(reg_path)
                    # For outliers, use only the original inlier points for Kabsch/Metrics
                    T_kabsch = estimate_kabsch(source_perturbed_xyz[inlier_mask], target_xyz[inlier_mask])
                    source_reg_xyz_clean = source_reg_xyz[inlier_mask]
                    rms = compute_rms(source_reg_xyz_clean, target_xyz[inlier_mask])
                    energy = compute_welsch_energy(source_reg_xyz_clean, target_xyz[inlier_mask])
                else:
                    source_reg_xyz = None
                    rms, energy = np.nan, np.nan
                
                rot_err_fricp, trans_err_fricp = compute_transform_error(T_fricp, T_curr)
                rot_err_kabsch, trans_err_kabsch = compute_transform_error(T_kabsch, T_curr)
                
                metrics_sum[m]["time"].append(res["time"])
                metrics_sum[m]["iters"].append(res["iters"] if not np.isnan(res["iters"]) else 0)
                metrics_sum[m]["final_energy"].append(res["final_energy"] if not np.isnan(res["final_energy"]) else 0)
                metrics_sum[m]["rot_err_fricp"].append(rot_err_fricp)
                metrics_sum[m]["trans_err_fricp"].append(trans_err_fricp)
                metrics_sum[m]["rot_err_kabsch"].append(rot_err_kabsch if T_kabsch is not None else 0)
                metrics_sum[m]["trans_err_kabsch"].append(trans_err_kabsch if T_kabsch is not None else 0)
                metrics_sum[m]["rms"].append(rms if not np.isnan(rms) else 0)
                metrics_sum[m]["energy"].append(energy if not np.isnan(energy) else 0)

                if n_trials == 1 and config.VERBOSE:
                     print(f"      [DEBUG] {method_name}: rot_err={rot_err_fricp:.2f} (Kabsch={rot_err_kabsch:.2f})")
                     if T_fricp is not None and np.allclose(T_fricp, np.eye(4), atol=1e-5):
                          print(f"      [WARNING] {method_name} returned IDENTITY matrix.")

            # Average and append
            for m in config.methods:
                method_name = config.method_names.get(m, f"Method_{m}")
                
                 # Calculate mean and std for all metrics
                stats = {}
                for k, v in metrics_sum[m].items():
                    stats[k] = np.mean(v)
                    stats[f"{k}_std"] = np.std(v)
                
                res_item = {
                    "outlier_ratio": ratio,
                    "method": method_name
                }
                res_item.update(stats)
                results.append(res_item)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(config.BASE_CONFIG["output_root"], "outlier_results.csv"), index=False)
    
    plot_experiment_results(
        df, 
        x_axis='outlier_ratio', 
        title='Resistance to Outliers', 
        save_path=os.path.join(config.BASE_CONFIG["output_root"], "outlier_plot.png")
    )

    plot_convergence_metrics(
        df,
        x_axis='outlier_ratio',
        title='Convergence Performance vs Outliers',
        save_path=os.path.join(config.BASE_CONFIG["output_root"], "outlier_convergence.png")
    )
    return df

if __name__ == "__main__":
    args = get_args()
    if not os.path.exists(config.BASE_CONFIG["output_root"]):
        os.makedirs(config.BASE_CONFIG["output_root"])
    
    df_noise = run_experiment_noise(n_trials=args.trials)
    df_outliers = run_experiment_outliers(n_trials=args.trials)

    if config.RUN_MIXED_EXPERIMENT:
        df_mixed = run_experiment_mixed(n_trials=args.trials)
        
