from pyneuromesh.refinement.refine_net import RefineNet
from pyneuromesh.refinement.mock_data import generate_mock_data
import torch
import torch.nn as nn

def train_model():
    X, y = generate_mock_data()
    net = RefineNet()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    for epoch in range(1000):
        pred = net(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

    torch.save(net.state_dict(), "refine_net.pt")
    print("[✓] Saved refine_net.pt")

if __name__ == "__main__":
    train_model()

