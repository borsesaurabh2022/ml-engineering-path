# MLP 01: First neural network

A multi-layer perceptron (2 → 16 → 16 → 2, ReLU) trained to classify the
two-moons dataset, built from scratch in PyTorch.

## Data
1000 points from two interleaved half-circles with Gaussian noise (σ = 0.2),
generated in `src/data.py`. Split 70 / 15 / 15 into train / validation / test
with a fixed seed. Tests guarantee the splits don't overlap.

## Run
From this folder, with the venv active:

    python -m src.train --hidden 16 16   # train MLP, save checkpoint + loss curves
    python -m src.train --hidden         # train linear baseline (no hidden layer)
    python -m src.plot_boundary          # decision boundaries side by side
    python -m src.visualize              # network diagram with initial weights
    python -m src.evaluate               # final test-set evaluation (run once)
    python -m pytest -v                  # 9 tests: model shapes, gradients, data split

## Results
Adam (lr 0.01), batch size 32, 100 epochs, cross-entropy loss.

| Model | Parameters | Train acc | Val acc | Test acc |
|---|---|---|---|---|
| Linear (no hidden layer) | 6 | 0.860 | 0.840 | not evaluated |
| MLP [16, 16] | 354 | 0.977 | 0.967 | **0.947** (test loss: FILL IN) |

- The linear model can only draw a straight boundary and underfits (train ≈ val ≈ 0.85).
- The MLP's boundary is piecewise linear and follows the moons. The remaining errors are
  noise points lying inside the other class.
- Validation loss levels off after about 20 epochs while training loss keeps falling: mild overfitting.
- Test vs validation (0.947 vs 0.967) is within one standard error (≈ 0.018 for n = 150).

## What I learned

