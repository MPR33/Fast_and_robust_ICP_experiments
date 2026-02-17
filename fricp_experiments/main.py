import os
import numpy as np
import pandas as pd
from io_pc.ply import read_ply, write_ply
from perturb.noise import add_gaussian_noise
from perturb.outliers import substitute_outliers
from perturb.transform import apply_transform, get_random_rotation, get_random_translation
from fricp.runner import FRICPRunner
from metrics.transform_error import compute_transform_error, estimate_kabsch
from metrics.reconstruction import compute_rms, compute_welsch_energy
from viz.plots import plot_experiment_results, plot_mixed_results
import config

# Seed for reproducibility
np.random.seed(42)

def run_experiment_mixed():
    print("\n--- Starting Mixed (Noise + Outliers) Experiment ---")
    target_xyz, target_nor = read_ply(config.BASE_CONFIG["data_path"])
    runner = FRICPRunner(config.BASE_CONFIG["exe_path"])
    results = []

    T_gt_path = os.path.join(config.BASE_CONFIG["output_root"], "mixed_gt_transform.txt")
    if config.SKIP_EXISTING and os.path.exists(T_gt_path):
        T_gt = np.loadtxt(T_gt_path)
        R_gt = T_gt[:3, :3] # This is actually T_gt_inv in the logic below, let's be careful
        # Wait, the logic below uses R_gt, t_gt as the perturbation (target -> source)
        # and T_gt as source -> target.
        T_gt_inv = np.linalg.inv(T_gt)
        R_gt = T_gt_inv[:3, :3]
        t_gt = T_gt_inv[:3, 3]
    else:
        # Use a single random transform for all combinations to ensure comparability
        R_gt = get_random_rotation(max_angle_deg=12)
        t_gt = get_random_translation(max_dist=0.08)
        T_gt_inv = np.eye(4)
        T_gt_inv[:3, :3] = R_gt
        T_gt_inv[:3, 3] = t_gt
        T_gt = np.linalg.inv(T_gt_inv)
        np.savetxt(T_gt_path, T_gt)

    total_iters = len(config.noise_grid_mixed) * len(config.outlier_grid_mixed)
    current_iter = 0

    for sigma in config.noise_grid_mixed:
        for ratio in config.outlier_grid_mixed:
            current_iter += 1
            print(f"  Combination {current_iter}/{total_iters}: sigma={sigma}, outliers={ratio}")
            
            # 1. Perturb
            source_init_xyz, source_init_nor = apply_transform(target_xyz, R_gt, t_gt, target_nor)
            # Add outliers first then noise? Or vice versa. Order matters but usually outliers replace, then noise on top.
            source_perturbed_xyz, source_perturbed_nor = substitute_outliers(source_init_xyz, source_init_nor, ratio=ratio)
            source_perturbed_xyz, source_perturbed_nor = add_gaussian_noise(source_perturbed_xyz, source_perturbed_nor, sigma=sigma)
            
            # 2. Save
            work_dir = os.path.join(config.BASE_CONFIG["output_root"], f"mixed_s{sigma}_r{ratio}")
            if not os.path.exists(work_dir): os.makedirs(work_dir)
            
            target_path = os.path.join(work_dir, "target.ply")
            source_path = os.path.join(work_dir, "source.ply")
            write_ply(target_path, target_xyz, target_nor)
            write_ply(source_path, source_perturbed_xyz, source_perturbed_nor)
            
            for m in config.methods:
                method_name = config.method_names.get(m, f"Method_{m}")
                res_dir = os.path.join(work_dir, f"method_{m}")
                res = runner.run(target_path, source_path, res_dir, method=m, skip_if_exists=config.SKIP_EXISTING)
                
                T_fricp = res["transformation"]
                rot_err_fricp, trans_err_fricp = compute_transform_error(T_fricp, T_gt)
                
                results.append({
                    "noise_sigma": sigma,
                    "outlier_ratio": ratio,
                    "method": method_name,
                    "time": res["time"],
                    "rot_err_fricp": rot_err_fricp,
                    "trans_err_fricp": trans_err_fricp
                })

                if config.VERBOSE:
                    print(f"      [DEBUG] {method_name}: rot_err={rot_err_fricp:.2f}, trans_err={trans_err_fricp:.4f}")
                    if T_fricp is not None:
                        # Print if it's potentially Identity
                        is_identity = np.allclose(T_fricp, np.eye(4), atol=1e-5)
                        if is_identity:
                            print(f"      [WARNING] {method_name} returned IDENTITY matrix.")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(config.BASE_CONFIG["output_root"], "mixed_results.csv"), index=False)
    
    plot_mixed_results(
        df, 
        title='Resistance to Mixed Noise & Outliers (Rotation Error)', 
        save_path=os.path.join(config.BASE_CONFIG["output_root"], "mixed_heatmap.png")
    )
    return df

def run_experiment_noise():
    print("--- Starting Noise Resistance Experiment ---")
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
        
        # 1. Perturb
        source_init_xyz, source_init_nor = apply_transform(target_xyz, R_gt, t_gt, target_nor)
        source_perturbed_xyz, source_perturbed_nor = add_gaussian_noise(source_init_xyz, source_init_nor, sigma=sigma)
        
        # 2. Save
        work_dir = os.path.join(config.BASE_CONFIG["output_root"], f"noise_{sigma}")
        if not os.path.exists(work_dir): os.makedirs(work_dir)
        
        target_path = os.path.join(work_dir, "target.ply")
        source_path = os.path.join(work_dir, "source.ply")
        write_ply(target_path, target_xyz, target_nor)
        write_ply(source_path, source_perturbed_xyz, source_perturbed_nor)
        
        for m in config.methods:
            method_name = config.method_names.get(m, f"Method_{m}")
            
            res_dir = os.path.join(work_dir, f"method_{m}")
            res = runner.run(target_path, source_path, res_dir, method=m, skip_if_exists=config.SKIP_EXISTING)
            
            if "Skipped" in res["stdout"]:
                print(f"    - {method_name}: [LOADED]")
            else:
                print(f"    - {method_name}: [DONE] ({res['time']:.2f}s)")
            
            # Read registered points to compute Kabsch
            reg_path = os.path.join(res_dir, "m3reg_pc.ply") # FRICP always uses m3 prefix or method?
            # Actually runner.py should probably handle the prefix. 
            # In runner.py I used f"m{method}reg_pc.ply"? Let's check runner.
            reg_path = os.path.join(res_dir, f"m{m}reg_pc.ply")
            
            T_fricp = res["transformation"]
            T_kabsch = None
            
            if os.path.exists(reg_path):
                source_reg_xyz, _ = read_ply(reg_path)
                # (i) & (ii) Kabsch between perturbed source and target using GT correspondences
                # Since we know source_perturbed[i] corresponds to target_xyz[i]
                T_kabsch = estimate_kabsch(source_perturbed_xyz, target_xyz)
                
                # Also Kabsch between source_perturbed and source_reg to verify the file matrix
                T_fricp_verify = estimate_kabsch(source_perturbed_xyz, source_reg_xyz)
            else:
                source_reg_xyz = None
                T_fricp_verify = None

            # 3. Metrics
            rot_err_fricp, trans_err_fricp = compute_transform_error(T_fricp, T_gt)
            rot_err_kabsch, trans_err_kabsch = compute_transform_error(T_kabsch, T_gt)
            
            # Check if matrix in file matches actual points
            rot_err_verify, _ = compute_transform_error(T_fricp, T_fricp_verify)
            
            if source_reg_xyz is not None:
                rms = compute_rms(source_reg_xyz, target_xyz) # Brute RMS (no correspondence)
                # Energy with correspondence (more precise for validation)
                energy = compute_welsch_energy(source_reg_xyz, target_xyz)
            else:
                rms, energy = np.nan, np.nan
            
            results.append({
                "noise_sigma": sigma,
                "method": method_name,
                "time": res["time"],
                "rot_err_fricp": rot_err_fricp,
                "trans_err_fricp": trans_err_fricp,
                "rot_err_kabsch": rot_err_kabsch,
                "trans_err_kabsch": trans_err_kabsch,
                "matrix_verify_err": rot_err_verify,
                "rms": rms,
                "energy": energy
            })

            if config.VERBOSE:
                print(f"      [DEBUG] {method_name}: rot_err={rot_err_fricp:.2f} (Kabsch={rot_err_kabsch:.2f}), verify_err={rot_err_verify:.4f}")
                if T_fricp is not None and np.allclose(T_fricp, np.eye(4), atol=1e-5):
                    print(f"      [WARNING] {method_name} returned IDENTITY matrix.")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(config.BASE_CONFIG["output_root"], "noise_results.csv"), index=False)
    
    plot_experiment_results(
        df, 
        x_axis='noise_sigma', 
        title='Resistance to Noise', 
        save_path=os.path.join(config.BASE_CONFIG["output_root"], "noise_plot.png")
    )
    return df

def run_experiment_outliers():
    print("\n--- Starting Outlier Resistance Experiment ---")
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
        R_gt = get_random_rotation(max_angle_deg=10)
        t_gt = get_random_translation(max_dist=0.05)
        T_gt_inv = np.eye(4)
        T_gt_inv[:3, :3] = R_gt
        T_gt_inv[:3, 3] = t_gt
        T_gt = np.linalg.inv(T_gt_inv)
        np.savetxt(T_gt_path, T_gt)

    for ratio in config.outlier_grid:
        print(f"  Testing outlier ratio: {ratio}")
        
        # 1. Perturb
        source_init_xyz, source_init_nor = apply_transform(target_xyz, R_gt, t_gt, target_nor)
        source_perturbed_xyz, source_perturbed_nor = substitute_outliers(source_init_xyz, source_init_nor, ratio=ratio)
        
        # 2. Save
        work_dir = os.path.join(config.BASE_CONFIG["output_root"], f"outliers_{ratio}")
        if not os.path.exists(work_dir): os.makedirs(work_dir)
        
        target_path = os.path.join(work_dir, "target.ply")
        source_path = os.path.join(work_dir, "source.ply")
        write_ply(target_path, target_xyz, target_nor)
        write_ply(source_path, source_perturbed_xyz, source_perturbed_nor)
        
        for m in config.methods:
            method_name = config.method_names.get(m, f"Method_{m}")
            
            res_dir = os.path.join(work_dir, f"method_{m}")
            res = runner.run(target_path, source_path, res_dir, method=m, skip_if_exists=config.SKIP_EXISTING)
            
            if "Skipped" in res["stdout"]:
                print(f"    - {method_name}: [LOADED]")
            else:
                print(f"    - {method_name}: [DONE] ({res['time']:.2f}s)")
            
            reg_path = os.path.join(res_dir, f"m{m}reg_pc.ply")
            T_fricp = res["transformation"]
            T_kabsch = None
            
            if os.path.exists(reg_path):
                source_reg_xyz, _ = read_ply(reg_path)
                # For outliers, use only the original (non-outlier) points for Kabsch/Metrics
                n_original = len(target_xyz)
                T_kabsch = estimate_kabsch(source_perturbed_xyz[:n_original], target_xyz)
                source_reg_xyz_clean = source_reg_xyz[:n_original]
                rms = compute_rms(source_reg_xyz_clean, target_xyz)
                energy = compute_welsch_energy(source_reg_xyz_clean, target_xyz)
            else:
                source_reg_xyz = None
                rms, energy = np.nan, np.nan
            
            rot_err_fricp, trans_err_fricp = compute_transform_error(T_fricp, T_gt)
            rot_err_kabsch, trans_err_kabsch = compute_transform_error(T_kabsch, T_gt)
            
            results.append({
                "outlier_ratio": ratio,
                "method": method_name,
                "time": res["time"],
                "rot_err_fricp": rot_err_fricp,
                "trans_err_fricp": trans_err_fricp,
                "rot_err_kabsch": rot_err_kabsch,
                "trans_err_kabsch": trans_err_kabsch,
                "rms": rms,
                "energy": energy
            })

            if config.VERBOSE:
                print(f"      [DEBUG] {method_name}: rot_err={rot_err_fricp:.2f} (Kabsch={rot_err_kabsch:.2f})")
                if T_fricp is not None and np.allclose(T_fricp, np.eye(4), atol=1e-5):
                    print(f"      [WARNING] {method_name} returned IDENTITY matrix.")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(config.BASE_CONFIG["output_root"], "outlier_results.csv"), index=False)
    
    plot_experiment_results(
        df, 
        x_axis='outlier_ratio', 
        title='Resistance to Outliers', 
        save_path=os.path.join(config.BASE_CONFIG["output_root"], "outlier_plot.png")
    )
    return df

if __name__ == "__main__":
    if not os.path.exists(config.BASE_CONFIG["output_root"]):
        os.makedirs(config.BASE_CONFIG["output_root"])
    
    df_noise = run_experiment_noise()
    df_outliers = run_experiment_outliers()

    if config.RUN_MIXED_EXPERIMENT:
        df_mixed = run_experiment_mixed()
        
