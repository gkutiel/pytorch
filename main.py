
import torch
from torch import nn
from torch.nn.functional import cross_entropy
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm

LR = 0.01
EPOCHS = 2

train = MNIST(
    root='./data',
    train=True,
    transform=ToTensor(),
    download=True)

train_dl = DataLoader(
    train,
    batch_size=512,
    shuffle=True,
    drop_last=True)

test = MNIST(
    root='./data',
    train=False,
    transform=ToTensor(),
    download=True)

test_dl = DataLoader(
    test,
    batch_size=512,
    shuffle=False,
    drop_last=True)


if __name__ == "__main__":
    torch.manual_seed(0)

    def blk(i: int, o: int):
        return [
            nn.Conv2d(i, o, kernel_size=3, padding=1),
            nn.BatchNorm2d(o),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]

    net = nn.Sequential(
        # 28 -> 14
        *blk(1, 32),
        # 14 -> 7
        *blk(32, 64),
        # 7 -> 64x1x7x7
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 10)
    )

    opt = SGD(net.parameters(), lr=LR, momentum=0.9)

    bar = tqdm(range(EPOCHS))
    net.train()
    for _ in bar:
        for x, y in train_dl:
            opt.zero_grad()
            y_hat = net(x)
            loss = cross_entropy(y_hat, y)
            bar.set_postfix(loss=loss.item())
            loss.backward()
            opt.step()

        # with torch.no_grad():
        #     for x, y in test_dl:
        #         ...

    torch.save(net.state_dict(), "mnist_cnn.pt")
