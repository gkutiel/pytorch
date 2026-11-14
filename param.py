

import torch
from torch import nn, optim
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

torch.manual_seed(0)


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.freeze = nn.Linear(1, 1)
        self.freeze.requires_grad_(False)
        self.freeze.weight.fill_(4.0)
        self.freeze.bias.fill_(1.0)

        self.net = nn.Linear(1, 1)

    def forward(self, x):
        out = self.freeze(x)
        out = self.net(out)
        return out


x = torch.rand(100, 1)
y = 2*x + 3


model = MyModule()


LR = 0.02
EPOCHS = 200

opt = optim.SGD(
    model.parameters(),
    LR,
    momentum=0.9)

bar = tqdm(range(EPOCHS))
for _ in bar:
    opt.zero_grad()
    y_hat = model(x)
    loss = mse_loss(y_hat, y)
    loss.backward()
    grad_norm = clip_grad_norm_(model.parameters(), max_norm=1)

    bar.set_postfix(
        loss=loss.item(),
        grad_norm=grad_norm.item())

    opt.step()

torch.save(model.state_dict(), 'model.pt')


# sd = torch.load('model.pt')
# print('SD:', sd)
# model.load_state_dict(sd)
# model.eval()
# with torch.no_grad():
#     x = torch.tensor([[5.0]])
#     print(x, model(x))
