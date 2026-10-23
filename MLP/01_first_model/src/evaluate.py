import torch
from torch import nn

from src.data import make_moons, train_val_test_split
from src.model import MLP
from src.train import evaluate

if __name__ == "__main__":
    generator = torch.Generator().manual_seed(0)
    X, y = make_moons(1000, noise=0.2, generator=generator)
    _, _, (X_test, y_test) = train_val_test_split(
        X, y, val_frac=0.15, test_frac=0.15, generator=generator
    )

    model = MLP(2, [16, 16], 2)  # chosen on validation accuracy, before seeing test data
    model.load_state_dict(torch.load("checkpoints/mlp_16-16.pt", weights_only=True))

    test_loss, test_acc = evaluate(model, X_test, y_test, nn.CrossEntropyLoss())
    n_errors = round((1 - test_acc) * len(X_test))
    print(
        f"TEST (used once) | loss {test_loss:.4f} | accuracy {test_acc:.3f} "
        f"| errors {n_errors}/{len(X_test)}"
    )
