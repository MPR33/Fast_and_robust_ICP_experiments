import numpy as np

def compute_transform_error(T_est, T_gt):
    """
    Computes rotation error (degrees) and translation error.
    T_est, T_gt are 4x4 matrices.
    """
    if T_est is None or T_gt is None:
        return np.nan, np.nan
        
    R_est = T_est[:3, :3]
    t_est = T_est[:3, 3]
    
    R_gt = T_gt[:3, :3]
    t_gt = T_gt[:3, 3]
    
    # Rotation error
    # R_diff = R_est.T @ R_gt  (if source was moved by T_gt and we estimated T_est)
    # The error is the rotation needed to go from T_est to T_gt
    R_diff = R_est @ R_gt.T
    trace = np.trace(R_diff)
    angle_rad = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    angle_deg = np.rad2deg(angle_rad)
    
    # Translation error
    t_err = np.linalg.norm(t_est - t_gt)
    
    return angle_deg, t_err

def estimate_kabsch(A, B):
    """
    Estimates the rigid transform T = [R | t] such that B ~ R*A + t
    using the Kabsch algorithm (SVD).
    A, B: (N, 3) arrays of corresponding points.
    Returns: 4x4 matrix.
    """
    assert A.shape == B.shape
    
    # 1. Centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    
    # 2. Center the points
    AA = A - centroid_A
    BB = B - centroid_B
    
    # 3. Covariance matrix
    H = AA.T @ BB
    
    # 4. SVD
    U, S, Vt = np.linalg.svd(H)
    
    # 5. Rotation
    R = Vt.T @ U.T
    
    # Special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
        
    # 6. Translation
    t = centroid_B - R @ centroid_A
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T
