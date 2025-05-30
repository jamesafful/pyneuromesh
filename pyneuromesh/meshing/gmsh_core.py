import numpy as np
import torch
import gmsh
import trimesh
from skimage.measure import marching_cubes


class GmshMesher:
    def __init__(self, sdf_function, domain_bounds, resolution=0.05, level=0.0):
        """
        sdf_function: callable R^3 -> R (signed distance)
        domain_bounds: ((xmin, xmax), (ymin, ymax), (zmin, zmax))
        """
        self.sdf = sdf_function
        self.bounds = domain_bounds
        self.res = resolution
        self.level = level
        self._init_gmsh()

    def _init_gmsh(self):
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.add("PyNeuroMesh")

    def define_geometry(self):
        """
        Sample the SDF, extract isosurface, export to STL, import to GMSH.
        """
        xmin, xmax = self.bounds[0]
        ymin, ymax = self.bounds[1]
        zmin, zmax = self.bounds[2]
        res = self.res

        x = np.arange(xmin, xmax, res)
        y = np.arange(ymin, ymax, res)
        z = np.arange(zmin, zmax, res)
        grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
        pts = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)

        pts_tensor = torch.from_numpy(pts).float()
        with torch.no_grad():
            sdf_vals = self.sdf(pts_tensor).squeeze().cpu().numpy()

        sdf_grid = sdf_vals.reshape(len(x), len(y), len(z))
        verts, faces, _, _ = marching_cubes(sdf_grid, level=self.level)

        dx = x[1] - x[0]
        verts[:, 0] = verts[:, 0] * dx + xmin
        verts[:, 1] = verts[:, 1] * dx + ymin
        verts[:, 2] = verts[:, 2] * dx + zmin

        mesh = trimesh.Trimesh(verts, faces)
        mesh.export("temp_sdf_geom.stl")

        gmsh.model.occ.importShapes("temp_sdf_geom.stl")
        gmsh.model.occ.synchronize()

    def define_size_field(self):
        """
        Basic GMSH threshold-based distance field (fallback/default).
        """
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "SurfacesList", [1])

        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", self.res / 2)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", self.res * 2)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 0.1)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 0.5)
        gmsh.model.mesh.field.setAsBackgroundMesh(2)

    def inject_size_field_from_function(self, density_fn, sample_res=0.1):
        """
        Sample a neural density_fn over a grid and use it as GMSH background mesh field.
        """
        xmin, xmax = self.bounds[0]
        ymin, ymax = self.bounds[1]
        zmin, zmax = self.bounds[2]

        xs = np.arange(xmin, xmax, sample_res)
        ys = np.arange(ymin, ymax, sample_res)
        zs = np.arange(zmin, zmax, sample_res)

        coords = []
        values = []

        for x in xs:
            for y in ys:
                for z in zs:
                    size = density_fn([x, y, z])
                    coords.append([x, y, z])
                    values.append(size)

        field_tag = 1000
        gmsh.view.add("adaptive_density")
        gmsh.view.addModelData(field_tag, 0, "adaptive_density",
                               "NodeData", coords, [values])
        gmsh.model.mesh.field.add("PostView", field_tag)
        gmsh.model.mesh.field.setAsBackgroundMesh(field_tag)

    def generate(self, filename="output.msh", dim=3):
        gmsh.model.mesh.generate(dim)
        gmsh.write(filename)
        gmsh.finalize()
        print(f"[GMSH] Mesh written to {filename}")
