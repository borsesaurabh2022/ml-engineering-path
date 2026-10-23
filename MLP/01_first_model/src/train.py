import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data import make_moons, train_val_test_split
from src.model import MLP


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Fraction of rows where the highest logit is the correct class."""
    return (logits.argmax(dim=1) == y).float().mean().item()


def train_one_epoch(
    model: nn.Module, loader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer
) -> float:
    """One pass over the training data. Returns mean loss per sample."""
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        logits = model(xb)
        loss = loss_fn(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)  # loss is a batch mean; weight by batch size
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module, X: torch.Tensor, y: torch.Tensor, loss_fn: nn.Module
) -> tuple[float, float]:
    """Loss and accuracy on a full split, without tracking gradients."""
    model.eval()
    logits = model(X)
    return loss_fn(logits, y).item(), accuracy(logits, y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an MLP on two moons.")
    parser.add_argument(
        "--hidden",
        type=int,
        nargs="*",
        default=[16, 16],
        help="hidden layer sizes, e.g. --hidden 16 16; empty = no hidden layer",
    )
    args = parser.parse_args()
    name = "mlp_" + ("-".join(map(str, args.hidden)) or "linear")

    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    X, y = make_moons(1000, noise=0.2, generator=generator)
    (X_train, y_train), (X_val, y_val), _ = train_val_test_split(
        X, y, val_frac=0.15, test_frac=0.15, generator=generator
    )  # test set untouched

    loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(0),
    )
    model = MLP(2, args.hidden, 2)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    for epoch in range(1, 101):
        train_loss = train_one_epoch(model, loader, loss_fn, optimizer)
        val_loss, val_acc = evaluate(model, X_val, y_val, loss_fn)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch {epoch:3d} | train loss {train_loss:.4f} | "
                f"val loss {val_loss:.4f} | val acc {val_acc:.3f}"
            )

    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/{name}.pt")

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    ax_loss.plot(history["train_loss"], label="train")
    ax_loss.plot(history["val_loss"], label="validation")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("cross-entropy loss")
    ax_loss.legend()
    ax_acc.plot(history["val_acc"])
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("validation accuracy")
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/{name}_training_curves.png", dpi=150, bbox_inches="tight")
    plt.show()
