import torch

def simulate_residuals(mesh_points):
    """
    Mock simulation that returns a fake residual based on location.
    In practice, you’d parse SU2, OpenFOAM, or FEniCS output.
    """
    # Assume highest "error" near center (0,0,0)
    center = torch.tensor([0.0, 0.0, 0.0])
    pts = torch.tensor(mesh_points).float()
    distances = torch.norm(pts - center, dim=1, keepdim=True)
    residuals = torch.exp(-5.0 * distances**2)
    return residuals

