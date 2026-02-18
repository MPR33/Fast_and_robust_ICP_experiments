import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def plot_convergence_metrics(df, x_axis, title, save_path):
    """
    Plots Iterations and Final Energy for different methods.
    """
    methods = df['method'].unique()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    print(f"\n📊 Generating convergence plots for: {title}")
    jitter = np.linspace(-0.01, 0.01, len(methods))
    
    for i, method in enumerate(methods):
        m_data = df[df['method'] == method]
        color = colors[i % 10]
        marker = markers[i % len(markers)]
        
        # Calculate jittered x values
        x_range = (m_data[x_axis].max() - m_data[x_axis].min()) if len(m_data) > 1 else 1.0
        x_values = m_data[x_axis] + jitter[i] * x_range

        # Plot 1: Iterations
        if 'iters_std' in m_data.columns:
            axes[0].errorbar(x_values, m_data['iters'], yerr=m_data['iters_std'],
                             fmt='none', ecolor=color, alpha=0.5, capsize=3)
        axes[0].plot(x_values, m_data['iters'], 
                     marker=marker, color=color, label=method)
        
        # Plot 2: Final Energy (using log scale if values vary widely)
        # We clarify that this is the internal objective function energy
        energy_vals = m_data['final_energy'].replace(0, np.nan)
        if 'final_energy_std' in m_data.columns:
             axes[1].errorbar(x_values, energy_vals, yerr=m_data['final_energy_std'],
                              fmt='none', ecolor=color, alpha=0.4, capsize=3)
        axes[1].plot(x_values, energy_vals, 
                     marker=marker, color=color, label=method)
        
    axes[0].set_title('Number of Iterations to Converge')
    axes[0].set_xlabel(x_axis)
    axes[0].set_ylabel('Iterations')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title('Internal Objective Energy (Method-Specific)')
    axes[1].set_xlabel(x_axis)
    axes[1].set_ylabel('Energy')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3, which="both")
    axes[1].legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Convergence plot saved to: {save_path}")
    plt.close()

def plot_iteration_heatmap(df, title, save_path):
    """
    Creates a heatmap for Iterations in mixed experiments (Noise vs Outliers).
    """
    methods = df['method'].unique()
    n_methods = len(methods)
    
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5), squeeze=False)
    
    # Global min/max for consistent colorbar across methods
    vmin = df['iters'].min()
    vmax = df['iters'].max()

    for i, method in enumerate(methods):
        m_data = df[df['method'] == method]
        pivot = m_data.pivot(index='outlier_ratio', columns='noise_sigma', values='iters')
        
        ax = axes[0, i]
        im = ax.imshow(pivot, origin='lower', extent=[
            pivot.columns.min(), pivot.columns.max(), 
            pivot.index.min(), pivot.index.max()
        ], aspect='auto', vmin=vmin, vmax=vmax, cmap='viridis')
        
        ax.set_title(f"Method: {method}")
        ax.set_xlabel("Noise Sigma")
        ax.set_ylabel("Outlier Ratio")
        
        # Add values in cells
        for row in range(len(pivot.index)):
            for col in range(len(pivot.columns)):
                val = pivot.iloc[row, col]
                ax.text(pivot.columns[col], pivot.index[row], 
                        f"{int(val)}" if not np.isnan(val) else "N/A", 
                        ha="center", va="center", 
                        color="white" if val > (vmax+vmin)/2 else "black", 
                        fontsize=9, fontweight='bold')

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Total Iterations")
    
    plt.suptitle(title)
    plt.savefig(save_path)
    print(f"✅ Iteration heatmap saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    # Example usage / test
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    
    noise_csv = os.path.join(results_dir, "noise_results.csv")
    if os.path.exists(noise_csv):
        df = pd.read_csv(noise_csv)
        plot_convergence_metrics(df, "noise_sigma", "Convergence vs Noise", os.path.join(results_dir, "noise_convergence.png"))

    outlier_csv = os.path.join(results_dir, "outlier_results.csv")
    if os.path.exists(outlier_csv):
        df = pd.read_csv(outlier_csv)
        plot_convergence_metrics(df, "outlier_ratio", "Convergence vs Outliers", os.path.join(results_dir, "outlier_convergence.png"))

    mixed_csv = os.path.join(results_dir, "mixed_results.csv")
    if os.path.exists(mixed_csv):
        df = pd.read_csv(mixed_csv)
        plot_iteration_heatmap(df, "Iteration Efficiency (Mixed)", os.path.join(results_dir, "mixed_convergence_heatmap.png"))
