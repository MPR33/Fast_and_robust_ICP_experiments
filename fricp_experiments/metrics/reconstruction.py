import numpy as np
try:
    from scipy.spatial import KDTree
except ImportError:
    KDTree = None

def compute_rms(source_transformed, target):
    """
    Computes Root Mean Square distance between transformed source and target.
    Requires scipy for fast search.
    """
    if KDTree is None:
        # Fallback to brute force for small clouds if scipy is missing
        if len(source_transformed) * len(target) < 1e7:
            dists = []
            for p in source_transformed:
                dists.append(np.min(np.linalg.norm(target - p, axis=1)))
            return np.sqrt(np.mean(np.square(dists)))
        else:
            print("Warning: scipy.spatial.KDTree not available. Skipping RMS for large clouds.")
            return np.nan
            
    tree = KDTree(target)
    distances, _ = tree.query(source_transformed)
    rms = np.sqrt(np.mean(np.square(distances)))
    return rms

def compute_welsch_energy(source_transformed, target, nu=0.1):
    """
    Computes Welsch energy: sum( 1 - exp(-d_i^2 / (2 * nu^2)) )
    """
    if KDTree is None:
        return np.nan
        
    tree = KDTree(target)
    distances, _ = tree.query(source_transformed)
    
    energy = np.sum(1 - np.exp(-(distances**2) / (2 * nu**2)))
    return energy
