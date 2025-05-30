import torch

def generate_mock_data(n_points=10000):
    x = torch.rand(n_points, 3) * 2 - 1
    residual = torch.norm(x, dim=1, keepdim=True)
    input_features = torch.cat([x, residual], dim=1)
    target_mesh_size = 0.01 + 0.1 * (1 - torch.exp(-10 * residual))
    return input_features, target_mesh_size

