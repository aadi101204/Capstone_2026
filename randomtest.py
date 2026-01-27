import torch
x = torch.randn(10000,10000).cuda()
y = x @ x
print("Computed on:", y.device)
