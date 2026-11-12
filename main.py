

from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

LR = 0.1
EPOCHS = 50

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

    x, y = next(iter(train_dl))

    y_hat = net(x)

    print(y_hat.shape)
