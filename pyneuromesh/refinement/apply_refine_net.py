import torch
from pyneuromesh.refinement.refine_net import RefineNet

def load_refine_net(path="refine_net.pt"):
    model = RefineNet()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def make_density_function(model):
    def density_fn(xyz_tensor):
        if isinstance(xyz_tensor, list) or isinstance(xyz_tensor, tuple):
            xyz_tensor = torch.tensor([xyz_tensor], dtype=torch.float32)
        if xyz_tensor.shape[-1] == 3:
            residual = torch.norm(xyz_tensor, dim=-1, keepdim=True)
            x_aug = torch.cat([xyz_tensor, residual], dim=-1)
            with torch.no_grad():
                return model(x_aug).squeeze().item()
        else:
            raise ValueError("Expected tensor of shape (3,)")
    return density_fn

