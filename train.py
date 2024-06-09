import torch
import torch.nn as nn
import torch.nn.functional as F
from model import LeNet
from optims.sgd import SGD
from optims.adam import Adam
from optims.adam_w import AdamW
from optims.adagrad import Adagrad
from torch.utils.data import DataLoader
from torchvision import transforms as trans
from torchvision.datasets.mnist import MNIST
import argparse
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot(train_losses, test_accuracies):

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(train_losses, label="Train Loss")
    ax[0].set_xlabel("Iterations")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    ax[1].plot(test_accuracies, label="Test Accuracy")
    ax[1].set_xlabel("Iterations")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()

    plt.savefig("results.png")

def train_and_evaluate(model, optim):
    transforms = trans.Compose(
        [
            trans.ToTensor(),
            trans.Normalize((0.1307,), (0.3081,)),
            trans.Resize((28, 28)),
        ]
    )
    train_dataset = MNIST(
        "../datasets/mnist", train=True, download=True, transform=transforms
    )
    test_dataset = MNIST(
        "../datasets/mnist", train=False, download=True, transform=transforms
    )
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    model = model.to(DEVICE)
    model.train()
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    train_losses = []

    for epoch in range(1):
        for i, (x, y) in enumerate(train_dataloader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            loss = criterion(logits, y)
            train_losses.append(loss.item())

            optim.zero_grad()
            loss.backward()
            optim.step()
            print(f"Epoch: {epoch} | Batch: {i} | Loss: {loss.item():.3f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        accuracies = []
        for i, (x, y) in enumerate(test_dataloader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            predictions = F.softmax(logits, dim=-1)
            predictions = predictions.argmax(dim=-1)
            correct += (predictions == y).sum().item()
            total += len(y)
            acc = correct / total
            accuracies.append(acc)
            print(f"Batch: {i} | Accuracy: {acc}")

    print(f"Final accuracy: {sum(accuracies) / len(accuracies):.3f}")
    plot(train_losses, accuracies)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--optim", type=str, default="sgd", choices=["sgd", "adam", "adamw", "adagrad"])
    args = parser.parse_args()

    model = LeNet().to(DEVICE)
    match args.optim:
        case "sgd":
            optim = torch.optim.SGD(model.parameters(), lr=args.lr)
        case "adam":
            raise NotImplementedError("Adam is not implemented yet")
        case "adamw":
            raise NotImplementedError("AdamW is not implemented yet")
        case "adagrad":
            raise NotImplementedError("Adagrad is not implemented yet")


    train_and_evaluate(model, optim)