from pyneuromesh.geometry.sdf_networks import SDFNetwork
import torch

def test_sdf_output_shape():
    net = SDFNetwork()
    pts = torch.rand(16, 3)
    sdf = net(pts)
    assert sdf.shape == (16, 1)

