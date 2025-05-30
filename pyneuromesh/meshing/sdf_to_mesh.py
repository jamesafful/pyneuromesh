import numpy as np
import torch
from skimage import measure
import trimesh

def sample_sdf_on_grid(sdf_model, grid_res=64, bounds=((-1, 1), (-1, 1), (-1, 1))):
    """
    Evaluate SDF on a 3D grid.
    """
    x = torch.linspace(*bounds[0], grid_res)
    y = torch.linspace(*bounds[1], grid_res)
    z = torch.linspace(*bounds[2], grid_res)
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)

    with torch.no_grad():
        sdf_vals = sdf_model(points).cpu().numpy().reshape((grid_res, grid_res, grid_res))
    return sdf_vals, (x, y, z)

def sdf_to_mesh(sdf_vals, grid_axes, level=0.0):
    """
    Run marching cubes and return a Trimesh mesh.
    """
    verts, faces, _, _ = measure.marching_cubes(sdf_vals, level=level)
    # Scale vertices to original axes
    x, y, z = [a.numpy() for a in grid_axes]
    scale = [x[1] - x[0], y[1] - y[0], z[1] - z[0]]
    offset = [x[0], y[0], z[0]]
    verts = verts * scale + offset
    return trimesh.Trimesh(vertices=verts, faces=faces)

def export_mesh(mesh, filename="output.obj"):
    mesh.export(filename)

