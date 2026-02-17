import numpy as np

def read_ply(filename):
    """
    Reads an ASCII PLY file.
    Handles vertex x, y, z and nx, ny, nz.
    """
    with open(filename, 'r') as f:
        line = f.readline()
        if not line or not line.strip().startswith('ply'):
            raise ValueError("Not a PLY file")
        
        num_vertices = 0
        header_ended = False
        properties = []
        while not header_ended:
            line = f.readline()
            if not line: break
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            if line.startswith('property'):
                properties.append(line.split()[-1])
            if line.startswith('end_header'):
                header_ended = True
        
        data = []
        for _ in range(num_vertices):
            line = f.readline()
            if not line: break
            data.append([float(x) for x in line.split()])
            
    data = np.array(data)
    vertices = data[:, :3]
    normals = data[:, 3:6] if data.shape[1] >= 6 else np.zeros_like(vertices)
    return vertices, normals

def write_ply(filename, vertices, normals=None):
    """
    Writes an ASCII PLY file with normals.
    """
    if normals is None:
        normals = np.zeros_like(vertices)
        
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "end_header"
    ]
    
    with open(filename, 'w') as f:
        f.write("\n".join(header) + "\n")
        for i in range(len(vertices)):
            v = vertices[i]
            n = normals[i]
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
