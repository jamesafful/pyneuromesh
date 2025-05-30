import torch
from pyneuromesh.refinement.refine_net import RefineNet
from pyneuromesh.simulation.proxy_simulation import simulate_residuals

def train_refine_net_from_mesh(mesh_points):
    """
    mesh_points: list of [x, y, z] coordinates from GMSH surface/volume mesh
    """
    pts = torch.tensor(mesh_points).float()
    residuals = simulate_residuals(mesh_points)

    X = torch.cat([pts, residuals], dim=1)
    y = 0.01 + 0.1 * (1 - torch.exp(-10 * residuals))  # desired mesh size

    model = RefineNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(1000):
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"[Train] Epoch {epoch} - Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), "refine_net.pt")
    print("[✓] Saved retrained RefineNet from simulation data.")

