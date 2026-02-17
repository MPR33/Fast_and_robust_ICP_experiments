import matplotlib.pyplot as plt
import numpy as np

def plot_experiment_results(df, x_axis, title, save_path):
    """
    Plots Rotation Error (FRICP & Kabsch) and RMS for different methods.
    Saves the plot to save_path.
    """
    methods = df['method'].unique()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Use a standard color cycle + different markers to help with overlaps
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    print(f"\n📈 Generating plots for: {title}")
    jitter = np.linspace(-0.01, 0.01, len(methods)) # Small horizontal shift for visibility
    
    for i, method in enumerate(methods):
        m_data = df[df['method'] == method]
        color = colors[i % 10]
        marker = markers[i % len(markers)]
        
        # Calculate jittered x values
        x_range = (m_data[x_axis].max() - m_data[x_axis].min()) if len(m_data) > 1 else 1.0
        x_values = m_data[x_axis] + jitter[i] * x_range

        # Plot 1: Rotation Error
        axes[0].plot(x_values, m_data['rot_err_fricp'], 
                     marker=marker, color=color, label=f'{method} (FRICP)')
        if 'rot_err_kabsch' in m_data.columns:
            axes[0].plot(x_values, m_data['rot_err_kabsch'], 
                         linestyle='--', marker=marker, fillstyle='none', 
                         color=color, alpha=0.4, label=f'{method} (Kabsch)')
        
        # Plot 2: RMS
        axes[1].plot(x_values, m_data['rms'], 
                     marker=marker, color=color, label=method)
        
        print(f"  - Included in plot: {method}")

    axes[0].set_title('Rotation Error (degrees)')
    axes[0].set_xlabel(x_axis)
    axes[0].set_ylabel('Error')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title('RMS Error')
    axes[1].set_xlabel(x_axis)
    axes[1].set_ylabel('RMS')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Plot saved to: {save_path}")
    plt.close()

def plot_mixed_results(df, title, save_path):
    """
    Creates heatmaps for mixed experiments (Noise vs Outliers).
    One heatmap per method for Rotation Error.
    Uses log scaling for color intensity to handle wide error ranges.
    """
    methods = df['method'].unique()
    n_methods = len(methods)
    
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5), squeeze=False)
    
    # Apply log(1+x) to the error for better color scaling
    # We create a copy to avoid modifying the original dataframe
    plot_df = df.copy()
    plot_df['log_rot_err'] = np.log1p(plot_df['rot_err_fricp'])
    
    vmin = plot_df['log_rot_err'].min()
    vmax = plot_df['log_rot_err'].max()

    for i, method in enumerate(methods):
        m_data = plot_df[plot_df['method'] == method]
        # Pivot for heatmap color (logged)
        pivot_log = m_data.pivot(index='outlier_ratio', columns='noise_sigma', values='log_rot_err')
        # Pivot for text labels (raw values)
        pivot_raw = m_data.pivot(index='outlier_ratio', columns='noise_sigma', values='rot_err_fricp')
        
        ax = axes[0, i]
        im = ax.imshow(pivot_log, origin='lower', extent=[
            pivot_log.columns.min(), pivot_log.columns.max(), 
            pivot_log.index.min(), pivot_log.index.max()
        ], aspect='auto', vmin=vmin, vmax=vmax, cmap='YlOrRd')
        
        ax.set_title(f"Method: {method}")
        ax.set_xlabel("Noise Sigma")
        ax.set_ylabel("Outlier Ratio")
        
        # Add raw values in cells for readability
        for row in range(len(pivot_raw.index)):
            for col in range(len(pivot_raw.columns)):
                ax.text(pivot_raw.columns[col], pivot_raw.index[row], 
                        f"{pivot_raw.iloc[row, col]:.1f}", 
                        ha="center", va="center", color="black", fontsize=8)

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="log(1 + Rotation Error)")
    
    plt.suptitle(title)
    plt.savefig(save_path)
    print(f"✅ Mixed experiment heatmaps saved to: {save_path}")
    plt.close()

def plot_point_clouds(source, target, source_transformed=None):
    """
    Very basic matplotlib 3D scatter plot.
    Only suitable for small point clouds.
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Sample if too many points
    step_t = max(1, len(target) // 1000)
    step_s = max(1, len(source) // 1000)
    
    ax.scatter(target[::step_t, 0], target[::step_t, 1], target[::step_t, 2], c='b', s=2, label='Target', alpha=0.5)
    ax.scatter(source[::step_s, 0], source[::step_s, 1], source[::step_s, 2], c='r', s=2, label='Source (init)', alpha=0.5)
    
    if source_transformed is not None:
        ax.scatter(source_transformed[::step_s, 0], source_transformed[::step_s, 1], source_transformed[::step_s, 2], c='g', s=2, label='Source (reg)', alpha=0.8)
        
    ax.legend()
    plt.show()
