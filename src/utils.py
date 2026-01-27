import torch
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def gradient_penalty(critic, real, fake, device="cpu"):
    batch_size = real.size(0)
    epsilon = torch.rand(batch_size, 1, 1, device=device)
    if real.dim() == 4: # Handling cases with extra channel dims
        epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
    
    interpolated = epsilon * real + (1 - epsilon) * fake
    interpolated.requires_grad_(True)

    prob_interpolated = critic(interpolated)
    gradients = torch.autograd.grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(prob_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    # Use reshape instead of view to handle non-contiguous tensors from Critic permutations
    gradients = gradients.reshape(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp
