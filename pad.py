import torch

a = torch.zeros(2, 3, 4)
b = torch.arange(12).view(3, 4) + 1

print(a + b[None, :])
