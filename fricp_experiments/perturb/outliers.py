import numpy as np

def add_outliers(vertices, normals, ratio=0.1, box_scale=1.5):
    """
    Adds random points (outliers) to the point cloud.
    """
    n_outliers = int(len(vertices) * ratio)
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    center = (min_coords + max_coords) / 2
    size = (max_coords - min_coords) * box_scale
    
    outliers = np.random.uniform(center - size/2, center + size/2, (n_outliers, 3))
    # Random normals for outliers
    outlier_normals = np.random.normal(0, 1, (n_outliers, 3))
    l2 = np.linalg.norm(outlier_normals, axis=1, keepdims=True)
    outlier_normals /= np.where(l2 > 1e-6, l2, 1.0)
    
    return np.vstack([vertices, outliers]), np.vstack([normals, outlier_normals])

def substitute_outliers(vertices, normals, ratio=0.1, box_scale=1.5):
    """
    Replaces random points with outliers.
    """
    n_outliers = int(len(vertices) * ratio)
    indices = np.random.choice(len(vertices), n_outliers, replace=False)
    
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    center = (min_coords + max_coords) / 2
    size = (max_coords - min_coords) * box_scale
    
    outliers = np.random.uniform(center - size/2, center + size/2, (n_outliers, 3))
    # Random normals for outliers
    outlier_normals = np.random.normal(0, 1, (n_outliers, 3))
    l2 = np.linalg.norm(outlier_normals, axis=1, keepdims=True)
    outlier_normals /= np.where(l2 > 1e-6, l2, 1.0)
    
    new_vertices = vertices.copy()
    new_normals = normals.copy()
    new_vertices[indices] = outliers
    new_normals[indices] = outlier_normals
    
    inlier_mask = np.ones(len(vertices), dtype=bool)
    inlier_mask[indices] = False
    
    return new_vertices, new_normals, inlier_mask
