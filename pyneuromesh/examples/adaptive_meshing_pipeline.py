from pyneuromesh.geometry.sdf_networks import SDFNetwork
from pyneuromesh.meshing.gmsh_core import GmshMesher
from pyneuromesh.refinement.apply_refine_net import load_refine_net, make_density_function

sdf_model = SDFNetwork()
mesher = GmshMesher(sdf_function=sdf_model, domain_bounds=((-1, 1), (-1, 1), (-1, 1)), resolution=0.1)
mesher.define_geometry()

refine_model = load_refine_net("refine_net.pt")
density_fn = make_density_function(refine_model)
mesher.inject_size_field_from_function(density_fn)

mesher.generate("adaptive_mesh.msh")

