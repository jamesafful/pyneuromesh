from pyneuromesh.geometry.sdf_networks import SDFNetwork
from pyneuromesh.meshing.sdf_to_mesh import sample_sdf_on_grid, sdf_to_mesh, export_mesh

net = SDFNetwork()
sdf_vals, grid_axes = sample_sdf_on_grid(net, grid_res=64)
mesh = sdf_to_mesh(sdf_vals, grid_axes)
export_mesh(mesh, "neural_mesh.obj")

#to visualize
import matplotlib.pyplot as plt
mesh.show()



