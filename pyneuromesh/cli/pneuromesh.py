import torch
from pyneuromesh.geometry.sdf_networks import SDFNetwork

def main():
    net = SDFNetwork()
    pts = torch.rand(100, 3) * 2 - 1
    sdf = net(pts)
    print("SDF Range:", sdf.min().item(), sdf.max().item())

if __name__ == "__main__":
    main()

