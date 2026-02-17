import numpy as np

def apply_transform(vertices, rotation_matrix, translation_vector, normals=None):
    """
    Applies R*V + t to vertices and R*N to normals.
    """
    new_vertices = (rotation_matrix @ vertices.T).T + translation_vector
    if normals is not None:
        new_normals = (rotation_matrix @ normals.T).T
        return new_vertices, new_normals
    return new_vertices

def get_random_rotation(max_angle_deg=30):
    """
    Generates a random rotation matrix with a maximum angle.
    """
    angle = np.deg2rad(np.random.uniform(0, max_angle_deg))
    axis = np.random.normal(0, 1, 3)
    axis /= np.linalg.norm(axis)
    
    # Rodrigues' rotation formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R

def get_random_translation(max_dist=0.5):
    """
    Generates a random translation vector.
    """
    t = np.random.uniform(-max_dist, max_dist, 3)
    return t
