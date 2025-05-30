import gmsh
import numpy as np
import os

def gmsh_from_sdf(sdf_model, bbox=((-1, 1), (-1, 1), (-1, 1)), res=0.02, level=0.0, mesh_file="mesh.msh"):
    """
    Generates a mesh using GMSH via a neural SDF field.
    Uses distance field + threshold to extract geometry.
    """
    gmsh.initialize()
    gmsh.model.add("neural_mesh")

    # Create geometry bounding box
    xmin, xmax = bbox[0]
    ymin, ymax = bbox[1]
    zmin, zmax = bbox[2]
    gmsh.model.occ.addBox(xmin, ymin, zmin, xmax - xmin, ymax - ymin, zmax - zmin, 1)
    gmsh.model.occ.synchronize()

    # Define a distance field
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "FacesList", [1])

    # Threshold the distance field to define geometry
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", res / 2)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", res * 2)
    gmsh.model.mesh.field.setNumber(2, "DistMin", level - 0.02)
    gmsh.model.mesh.field.setNumber(2, "DistMax", level + 0.02)

    gmsh.model.mesh.field.setAsBackgroundMesh(2)

    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.write(mesh_file)
    gmsh.finalize()

    print(f"Mesh written to {mesh_file}")

