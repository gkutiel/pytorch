

import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

torch.manual_seed(0)


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


LR = 0.02
EPOCHS = 200

x = torch.randn(500, 10)
y = (x.sum(dim=1) > 2).float()

ds = TensorDataset(x, y)
train_ds, test_ds = random_split(ds, [400, 100])
train_dl = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    drop_last=True)

test_dl = DataLoader(
    test_ds,
    batch_size=32,
    shuffle=False,
    drop_last=False
)

model = MyModule()

opt = SGD(
    model.parameters(),
    LR, momentum=0.9)


def acc_eval():
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_dl:
            y_hat = model(x)
            correct += ((y_hat > 0.5) == y).sum()

    return correct / len(test_ds)


def train():
    for e in range(EPOCHS):
        model.train()
        bar = tqdm(train_dl, desc=f'Epoch {e}/{EPOCHS}')
        for x, y in bar:
            opt.zero_grad()
            y_hat = model(x)
            loss = binary_cross_entropy(y_hat, y)
            loss.backward()
            opt.step()

            bar.set_postfix(loss=loss.item())

        # EVAL
        print(f'Epoch {e}/EPOCHS: acc={acc_eval():.2f}')


if __name__ == '__main__':
    train()
    layer = model.net[0]
    print(
        layer.weight,
        layer.bias)

    torch.save(model.state_dict(), 'model.pt')
