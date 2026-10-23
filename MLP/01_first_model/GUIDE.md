# MLP 01 Guide: Building a First Neural Network from Scratch

This guide explains every concept used in `MLP/01_first_model`: the environment and
tools, the deep learning theory behind the code, and the engineering lessons learned
while building it. Read it alongside the source files in `src/` and `tests/`.

**What the stage builds:** a multi-layer perceptron (MLP) with the architecture
2 → 16 → 16 → 2 that classifies 2-D points from the "two moons" dataset into two
classes. Final result: **test accuracy 0.947**, compared with 0.840 validation
accuracy for a linear model with no hidden layer.

---

## Contents

1. [The ML pipeline at a glance](#1-the-ml-pipeline-at-a-glance)
2. [Project setup](#2-project-setup)
3. [Data](#3-data)
4. [The model: neurons, weights and biases](#4-the-model-neurons-weights-and-biases)
5. [Forward propagation](#5-forward-propagation)
6. [Loss and cost functions](#6-loss-and-cost-functions)
7. [Gradients and backpropagation](#7-gradients-and-backpropagation)
8. [Gradient descent and optimizers](#8-gradient-descent-and-optimizers)
9. [The training loop](#9-the-training-loop)
10. [Evaluation and diagnosis](#10-evaluation-and-diagnosis)
11. [Visualisation](#11-visualisation)
12. [Saving and loading models](#12-saving-and-loading-models)
13. [Testing with pytest](#13-testing-with-pytest)
14. [Code quality: ruff and pre-commit](#14-code-quality-ruff-and-pre-commit)
15. [Lessons learned](#15-lessons-learned)
16. [Glossary](#16-glossary)
17. [Command reference](#17-command-reference)

---

## 1. The ML pipeline at a glance

Every supervised learning project follows the same sequence. Each step has its own
file in `src/`:

```
data.py        generate data, split into train / validation / test
   │
model.py       define the network (layers, weights, biases, activations)
   │
train.py       repeat: forward pass → loss → gradients → update weights
   │           measure validation loss/accuracy after every epoch
   │           save the trained weights (checkpoint)
   │
plot_boundary.py / visualize.py   inspect what the model learned
   │
evaluate.py    measure test accuracy ONCE, on data never used before
```

Keeping the steps in separate files means each one can be tested and changed
independently. When something breaks, you know which file to open.

---

## 2. Project setup

### 2.1 Virtual environment (venv)

A **virtual environment** is a project-specific Python installation with its own
`site-packages` folder. Packages installed into it don't affect system Python or other
projects.

```bash
python3 -m venv .venv            # create it (once)
source .venv/bin/activate        # activate it (every new terminal)
which python                     # must point into .venv/
```

Why it matters here: ROS 2 Jazzy uses the system Python 3.12. Installing ML packages
into system Python with pip can break `rclpy` and other ROS packages.

**The ROS leak.** Sourcing `/opt/ros/jazzy/setup.bash` (done in `~/.bashrc`) sets
`PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages`. Python searches `PYTHONPATH`
**before** the venv's own packages, so ROS packages leak into the venv. Real
consequence seen in this project: pytest auto-loaded a ROS pytest plugin
(`launch_testing`), which imported `yaml`, which isn't in the venv, so pytest crashed.

Fix: the `activate_ml` alias in `~/.bashrc` activates the venv **and** runs
`unset PYTHONPATH`. Check with `echo "[$PYTHONPATH]"`, which should print `[]`.

### 2.2 Dependencies: `requirements.txt`

`requirements.txt` lists every package the project depends on, with **exact versions**
(`==`), so anyone can rebuild the same environment with
`python -m pip install -r requirements.txt`.

- List only the packages you chose directly (torch, numpy, matplotlib, pytest, ruff,
  pre-commit). Their own dependencies are resolved by pip.
- **CPU build of PyTorch:** `torch==2.14.1+cpu` from `download.pytorch.org/whl/cpu`.
  It's ~200 MB instead of ~2.5 GB with CUDA. Small MLPs train as fast on CPU, because
  for tiny matrix multiplications the GPU's overhead (copying data, launching kernels)
  costs more than the computation. The GPU comes in with CNNs, where each operation is
  large.
- The `.venv/` folder itself is never committed: it can always be rebuilt from
  `requirements.txt`.

### 2.3 `python -m`

Run scripts as modules, from the stage folder:

```bash
python -m src.train      # correct: the current folder goes on sys.path, so "src" is importable
python src/train.py      # fails: ModuleNotFoundError: No module named 'src'
python -m pytest         # runs the pytest of the ACTIVE Python, not /usr/bin/pytest
```

### 2.4 Git essentials

| Concept | Command | Notes |
|---|---|---|
| Create a repo | `git init -b main` | `-b main` names the first branch |
| See state | `git status` | Run before every commit |
| Stage files | `git add <files>` | List files explicitly; avoid `git add .` |
| Commit | `git commit -m "message"` | A snapshot of the staged files |
| History | `git log --oneline --decorate` | Shows commits, branches, tags |
| Mark a milestone | `git tag mlp-01-done` | A permanent label on one commit |
| Fix the last commit | `git commit --amend --no-edit` | **Only before it's pushed** |
| Move a tag | `git tag -f mlp-01-done` | Needed after an amend |

**`.gitignore`** keeps generated and large files out of history: `.venv/`,
`__pycache__/`, `data/`, `figures/`, `checkpoints/`, `*.pt`. Anything committed stays
in history forever, even if deleted later. So the rule is: commit the **code that
produces** an output, not the output itself.

**Amend rule:** `--amend` replaces a commit with a new one (new hash). That's harmless
locally, but if the old commit is already on GitHub, the histories diverge and you'd
need a force-push. So amend only unpushed commits.

---

## 3. Data

### 3.1 The two-moons dataset (`make_moons`)

1000 points in 2-D from two interleaved half-circles, one per class:

- class 0: `(cos θ, sin θ)` for θ in [0, π], the upper arc
- class 1: `(1 − cos θ, 0.5 − sin θ)`, the lower arc, shifted to interlock
- Gaussian noise with standard deviation 0.2 is added to every coordinate

**Features** `X` have shape `(1000, 2)` and dtype `float32`. **Labels** `y` have shape
`(1000,)` and dtype `int64` (`torch.long`), because `CrossEntropyLoss` expects class
indices as integers.

The classes **cannot be separated by a straight line**, and the noise pushes some
points into the other class. So no model can reach 100% accuracy, which is realistic.

### 3.2 Train, validation and test splits

| Split | Share | Points | Role |
|---|---|---|---|
| Training | 70% | 700 | The **only** data that changes the weights and biases |
| Validation | 15% | 150 | Measured after each epoch: monitor overfitting, compare models, choose hyperparameters |
| Test | 15% | 150 | Measured **once**, at the very end: the unbiased final estimate |

`train_val_test_split` shuffles the indices **once** with `torch.randperm` and cuts the
permutation into three slices, so every point lands in exactly one split.

Why shuffle: `make_moons` returns all class-0 points first. Without shuffling, the test
set would contain only class 0.

**Why a separate test set:** every decision (architecture, epochs, learning rate) is
made by looking at validation results. Validation therefore becomes slightly
optimistic: you picked the model that happened to do best on those exact points. The
test set has influenced nothing, so it's the only unbiased estimate of performance on
new data. As soon as a test result influences a decision, it stops being unbiased.

**Data leakage** happens when information from the validation or test set reaches
training. Training still runs without errors, and results just look better than they
really are. The test `test_splits_do_not_overlap` exists to catch exactly this.

### 3.3 Reproducibility: seeds and generators

Random numbers in PyTorch come from a **pseudo-random generator**: given the same seed,
it produces the same sequence. This project fixes every source of randomness:

- `torch.Generator().manual_seed(0)` is passed explicitly to data generation and
  splitting, so the data doesn't depend on any other random calls in the program.
- `torch.manual_seed(0)` before creating the model fixes the initial weights.
- The `DataLoader` gets its own generator, which fixes the shuffle order.

Result: every run produces exactly the same numbers, so differences between
experiments come from your changes, not from chance.

---

## 4. The model: neurons, weights and biases

### 4.1 A single neuron

A neuron computes a weighted sum of its inputs plus a bias, then applies an
activation function:

```
z = w₁x₁ + w₂x₂ + … + wₙxₙ + b        (weighted sum + bias)
a = f(z)                             (activation)
```

- **Weights** `w` scale each input: how strongly each input affects the output.
- **Bias** `b` shifts the result, independently of the inputs.
- Weights and biases are the **parameters**: the numbers that training adjusts.

### 4.2 A layer: `nn.Linear`

A **fully connected (linear) layer** computes many neurons at once as a matrix
operation:

```
Y = X Wᵀ + b

X: (batch, in_features)      input, one row per sample
W: (out_features, in_features)   weight matrix, one row per neuron
b: (out_features,)           one bias per neuron
Y: (batch, out_features)
```

`nn.Linear(2, 16)` holds a 16×2 weight matrix and 16 biases.

### 4.3 Activation function: ReLU

```
ReLU(z) = max(0, z)
```

**Why an activation function is necessary:** without one, stacking layers gains
nothing. Two linear layers collapse into one:

```
W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂) = W'x + b'
```

ReLU breaks this: it's **non-linear**, so a stack of Linear → ReLU layers can
represent curved decision boundaries. This was demonstrated directly: the model with
no hidden layer (and so no ReLU) reached only 0.84 validation accuracy and could only
draw a straight boundary.

### 4.4 The MLP architecture (`model.py`)

```
input (2) → Linear(2,16) → ReLU → Linear(16,16) → ReLU → Linear(16,2) → logits (2)
```

- **Hidden layers:** the layers between input and output (here 2 layers of 16 neurons).
- **No activation after the last layer.** The output values are **logits**: raw,
  unnormalised scores, one per class. `CrossEntropyLoss` applies softmax internally,
  so adding softmax in the model would apply it twice.
- Layer sizes are constructor arguments (`MLP(in_features, hidden_sizes, out_features)`),
  so the same class builds a linear model (`hidden_sizes=[]`) or a regression model
  (`out_features=1`).
- `super().__init__()` must be called first. It sets up the machinery that registers
  each `nn.Linear` as a submodule, so `model.parameters()` can find its weights and
  biases.
- `forward(x)` defines the computation. Call `model(x)`, not `model.forward(x)`.

### 4.5 Counting parameters

Each linear layer has `in × out` weights plus `out` biases:

| Layer | Weights | Biases | Total |
|---|---|---|---|
| Linear(2, 16) | 32 | 16 | 48 |
| Linear(16, 16) | 256 | 16 | 272 |
| Linear(16, 2) | 32 | 2 | 34 |
| **MLP [16, 16]** | | | **354** |
| Linear(2, 2), the linear model | 4 | 2 | **6** |

### 4.6 Weight initialisation

Before training, PyTorch fills each `nn.Linear` with random values drawn uniformly from
**±1/√in_features**:

| Layer | in_features | Initial range |
|---|---|---|
| 2 → 16 | 2 | ±0.71 |
| 16 → 16 | 16 | ±0.25 |
| 16 → 2 | 16 | ±0.25 |

The range shrinks with the number of inputs, so each neuron's weighted sum starts at a
similar scale whatever the layer width. Without this, activations grow or shrink layer
by layer, and training becomes unstable. Random values (not zeros) are needed so
neurons in the same layer start differently and learn different features.

---

## 5. Forward propagation

**Forward propagation** (the forward pass) computes the output from the input, layer by
layer, using the current weights and biases. For a batch of 32 points:

```
X        (32, 2)
h₁ = ReLU(X W₁ᵀ + b₁)      (32, 16)
h₂ = ReLU(h₁ W₂ᵀ + b₂)     (32, 16)
logits = h₂ W₃ᵀ + b₃       (32, 2)
```

Checking shapes at each step is the fastest way to find bugs: the second dimension of
one layer's output must equal the next layer's `in_features`.

**Why the decision boundary is made of straight segments:** each ReLU neuron is either
"off" (output 0) or "on" (output = its weighted sum), and the switch happens along a
straight line in input space. Within each region where the on/off pattern is fixed,
the whole network is a single affine function. So an MLP with ReLU computes a
**piecewise linear** function, and its decision boundary is straight segments joined at
corners. More neurons mean more regions and more corners to follow curved data.

---

## 6. Loss and cost functions

### 6.1 Terminology

- **Loss function:** measures the error for **one** sample (or one prediction).
- **Cost function:** the average loss over a set of samples (a batch, or the whole
  dataset). This is the number the optimizer minimises.
- In practice, and in PyTorch, both are called "loss". `nn.CrossEntropyLoss()` returns
  the **mean** over the batch by default, which is the cost.

The loss turns "how wrong is the model" into a single number that can be
differentiated, so gradients can show which direction to change each weight.

### 6.2 Softmax

Softmax converts logits `z` into probabilities that are positive and sum to 1:

```
p_k = exp(z_k) / Σⱼ exp(z_j)
```

### 6.3 Cross-entropy loss (classification)

For a sample whose true class is `y`:

```
L = −log(p_y)
```

The loss is the negative log-probability that the model assigns to the **correct**
class.

**Worked example:** logits `[2.0, 0.5]`, true class 0:

```
p₀ = e²·⁰ / (e²·⁰ + e⁰·⁵) = 7.389 / (7.389 + 1.649) = 0.818
L  = −log(0.818) = 0.201
```

If the model were confident in the wrong class (p₀ = 0.01), the loss would be
−log(0.01) = 4.6. Cross-entropy punishes confident mistakes heavily.

`nn.CrossEntropyLoss` takes **logits** (not probabilities) and **integer class
labels**. It combines log-softmax and the negative log-likelihood in one numerically
stable step.

### 6.4 Mean squared error (regression, used in MLP 02)

```
MSE = (1/N) Σ (ŷᵢ − yᵢ)²
RMSE = √MSE          (same units as y, easier to interpret)
```

---

## 7. Gradients and backpropagation

### 7.1 Gradient

The **gradient** of the loss with respect to a parameter, ∂L/∂w, says how much the loss
changes when that parameter changes slightly:

- positive gradient: increasing w increases the loss, so decrease w
- negative gradient: increasing w decreases the loss, so increase w
- magnitude: how sensitive the loss is to that parameter

The network has 354 parameters, so the gradient is a vector of 354 numbers.

### 7.2 Backpropagation

**Backpropagation** computes all these gradients efficiently by applying the **chain
rule** of calculus backwards through the network, from the loss to the first layer.
Each layer receives the gradient of the loss with respect to its output and computes:

1. the gradients for its own weights and biases (stored for the update), and
2. the gradient with respect to its input (passed back to the previous layer).

**Worked example with one neuron** (no activation), squared-error loss:

```
x = 2.0, w = 0.5, b = 0.1, target t = 1.5

Forward:   y = w·x + b = 0.5·2.0 + 0.1 = 1.1
           L = (y − t)² = (1.1 − 1.5)² = 0.16

Backward (chain rule):
           ∂L/∂y = 2(y − t)     = −0.8
           ∂L/∂w = ∂L/∂y · x    = −0.8 · 2.0 = −1.6
           ∂L/∂b = ∂L/∂y · 1    = −0.8
```

Both gradients are negative, so increasing w and b reduces the loss (section 8
continues this example).

A useful result for classification: for softmax + cross-entropy, the gradient with
respect to the logits is simply `p − one_hot(y)`: predicted probabilities minus the
true distribution.

### 7.3 Autograd in PyTorch

PyTorch does backpropagation automatically (**autograd**):

- During the forward pass, every operation on tensors is recorded in a
  **computational graph**.
- `loss.backward()` walks that graph backwards and stores ∂loss/∂parameter in each
  parameter's `.grad` attribute.
- **Gradients accumulate:** `.backward()` **adds** to `.grad`. That's why
  `optimizer.zero_grad()` must reset them before every new backward pass. Otherwise
  each step would include the gradients from all previous steps.
- `@torch.no_grad()` turns graph recording off. Use it for evaluation and plotting,
  where no gradients are needed: it saves memory and time.
- `.detach()` returns a tensor without its graph history (used when plotting weights).

---

## 8. Gradient descent and optimizers

### 8.1 Gradient descent

**Gradient descent** updates every parameter a small step in the direction that
reduces the loss:

```
w ← w − η · ∂L/∂w
```

`η` (eta) is the **learning rate**: the step size.

Continuing the one-neuron example with η = 0.1:

```
w ← 0.5 − 0.1 · (−1.6) = 0.66
b ← 0.1 − 0.1 · (−0.8) = 0.18
new y = 0.66 · 2.0 + 0.18 = 1.50    → loss 0.0
```

- Learning rate too large: steps overshoot the minimum, and the loss jumps or diverges.
- Learning rate too small: training is very slow.

### 8.2 Batch, mini-batch and stochastic

| Variant | Gradient computed from | Updates per epoch (700 samples) |
|---|---|---|
| Batch gradient descent | all 700 samples | 1 |
| **Mini-batch** (used here, batch size 32) | 32 samples | 22 |
| Stochastic (SGD) | 1 sample | 700 |

A mini-batch gradient is a noisy estimate of the full gradient, but you get many more
updates per epoch, and the noise can help escape poor regions.

- **Epoch:** one full pass over the training data.
- **Iteration (step):** one parameter update, i.e. one batch.
- 700 / 32 = 21.9, so 22 batches per epoch (the last one has 28 samples). 100 epochs
  means 2,200 updates.

### 8.3 Adam

This project uses `torch.optim.Adam(model.parameters(), lr=0.01)`. Adam improves plain
gradient descent by keeping two running averages per parameter:

```
m ← β₁·m + (1 − β₁)·g          (average gradient: momentum)
v ← β₂·v + (1 − β₂)·g²         (average squared gradient: scale)
w ← w − η · m̂ / (√v̂ + ε)        (m̂, v̂ = bias-corrected m, v)
```

The momentum term smooths noisy mini-batch gradients. Dividing by √v gives each
parameter its own effective step size. Adam works well with default settings, which
is why it's the usual first choice.

---

## 9. The training loop

### 9.1 One training step, in order (`train_one_epoch`)

```python
logits = model(xb)  # 1. forward pass
loss = loss_fn(logits, yb)  # 2. loss for this batch
optimizer.zero_grad()  # 3. reset accumulated gradients
loss.backward()  # 4. backpropagation: fill .grad
optimizer.step()  # 5. update weights and biases
```

The order matters: `zero_grad` must happen before `backward`, and `step` after it.

### 9.2 One epoch: train first, then validate

```python
for epoch in range(1, 101):
    train_loss = train_one_epoch(...)  # 22 batches → 22 weight updates (training data only)
    val_loss, val_acc = evaluate(...)  # weights frozen, one forward pass over validation data
```

Validation **never changes a weight**: `evaluate` has no `backward()` and no `step()`.
It's measured every epoch to produce a curve over time, which shows when the model
stops improving or starts overfitting.

### 9.3 `model.train()` and `model.eval()`

These switch layers like Dropout and BatchNorm between training and inference
behaviour. This MLP has neither, so here they change nothing, but they're written
anyway because forgetting them in later models (CNNs) is a silent bug.

### 9.4 Reading the loss numbers

- **Training loss** is the **average over the epoch**, while the weights are still
  changing. Early batches use worse weights.
- **Validation loss** is measured **after** the epoch, on the updated weights.
- That's why epoch 1 showed train 0.52 but validation 0.37: validation was measured
  later, not on easier data.
- `total_loss += loss.item() * len(xb)`: each batch loss is a mean, so it's weighted by
  batch size to get a correct per-sample average (the last batch is smaller).

---

## 10. Evaluation and diagnosis

### 10.1 Results

Adam (lr 0.01), batch size 32, 100 epochs:

| Model | Parameters | Train acc | Val acc | Test acc |
|---|---|---|---|---|
| Linear (no hidden layer) | 6 | 0.860 | 0.840 | not evaluated |
| MLP [16, 16] | 354 | 0.977 | 0.967 | **0.947** |

**Accuracy** = fraction of samples where the highest logit is the correct class
(`logits.argmax(dim=1) == y`).

### 10.2 Underfitting and overfitting

Compare training and validation performance:

| Pattern | Diagnosis | Meaning | Here |
|---|---|---|---|
| Train low, val low, close together | **Underfitting** | Model too simple for the pattern | Linear: 0.86 / 0.84 |
| Train high, val high, small gap | **Good fit** | | MLP: 0.977 / 0.967 |
| Train high, val clearly lower | **Overfitting** | Model memorises training details | Mild sign: val loss stops falling after ~20 epochs while train loss keeps falling |

The **generalisation gap** is the difference between training and validation
performance.

### 10.3 Measurement uncertainty

With a small evaluation set, accuracy itself is noisy. For accuracy p measured on n
samples:

```
standard error ≈ √(p · (1 − p) / n) = √(0.95 · 0.05 / 150) ≈ 0.018
```

With 150 points, one point is 0.67%, and a different but equally valid test set could
give anything from about 0.93 to 0.98. So test 0.947 against validation 0.967 is **not
a real difference** (it's about one standard error). Linear 0.84 against MLP 0.967 is.

Rule: before claiming one model is better, check that the gap is larger than the
measurement uncertainty.

### 10.4 Using the test set once

1. Choose the final model using validation results only.
2. Write down the expected test result.
3. Run the evaluation once (`python -m src.evaluate`).
4. Report the number, and don't change anything afterwards to improve it.

---

## 11. Visualisation

### 11.1 Network diagram (`draw_mlp` in `visualize.py`)

Circles are neurons, and lines are weights: blue positive, red negative, thickness
proportional to magnitude. Layer sizes are read from the model itself
(`model.modules()`, `nn.Linear.weight`). The input-layer lines look thickest because
of the ±1/√in_features initialisation (section 4.6). Because `draw_mlp` takes an `ax`,
"before" and "after" training can be drawn side by side.

### 11.2 Decision boundary (`plot_decision_boundary`)

1. Build a 300×300 grid covering the plot area (90,000 points).
2. Run all grid points through the model in one forward pass.
3. Colour each point by the predicted probability of class 1 (softmax of the logits).
4. Draw the contour where that probability is 0.5: the **decision boundary**.

What the plots showed:

- **Linear model:** one straight line, necessarily, since the model is a single affine
  map.
- **MLP:** straight segments joined at corners (piecewise linear), following the moons.
- **Remaining errors** are noise points that lie inside the other moon.
- **Outside the data,** the boundary just continues as straight lines. The model has
  no information there, so predictions in those regions are meaningless
  (extrapolation).

---

## 12. Saving and loading models

```python
torch.save(model.state_dict(), "checkpoints/mlp_16-16.pt")  # save

model = MLP(2, [16, 16], 2)  # rebuild the architecture
model.load_state_dict(torch.load("checkpoints/mlp_16-16.pt", weights_only=True))
```

- `state_dict()` is an ordered dictionary of all weights and biases. It contains no
  code, so the architecture must be rebuilt identically before loading.
- `weights_only=True` makes `torch.load` accept only tensors, not arbitrary Python
  objects. It's a safety habit, because loading pickled objects can execute code.
- Checkpoints are git-ignored: they're reproducible by re-running training with the
  same seed.

---

## 13. Testing with pytest

### 13.1 What to test

Test properties that must **always** hold, not results that depend on training
(like "accuracy > 0.9"):

| Test | Protects against |
|---|---|
| `test_output_shape` | A layer with the wrong size |
| `test_parameter_count` (parametrized ×3) | Architecture changes that go unnoticed |
| `test_gradients_reach_every_parameter` | A layer disconnected from the loss, which would never train |
| `test_shapes_and_dtypes` | Wrong label dtype for the loss function |
| `test_split_sizes` | Wrong split fractions |
| `test_splits_do_not_overlap` | **Data leakage** between splits |
| `test_same_seed_gives_same_data` | Broken reproducibility |

### 13.2 How pytest works

- Finds files named `test_*.py` and functions named `test_*` automatically.
- A plain `assert` is enough. On failure, pytest shows the actual values.
- `@pytest.mark.parametrize("hidden, expected", [...])` runs one test function with
  several inputs.
- `pytest.ini` with `pythonpath = .` puts the stage folder on `sys.path`, so tests can
  `from src.model import MLP`. With `testpaths = tests`, only `tests/` is searched.

### 13.3 Break it on purpose

A test you've never seen fail might be testing nothing. Changing the validation slice
to `perm[n_test - 10 : n_test + n_val]` made `test_split_sizes` and
`test_splits_do_not_overlap` fail, which proved they detect a real leak. Then the change
was undone.

---

## 14. Code quality: ruff and pre-commit

### 14.1 ruff check (linter)

A **linter** reads code without running it and reports likely bugs and style problems.
Configuration is in the root `pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "UP"]

[tool.ruff.lint.isort]
known-first-party = ["src"]
```

| Rule group | Checks | Example found in this project |
|---|---|---|
| E, W (pycodestyle) | Style errors and warnings | Trailing whitespace, line too long, no final newline |
| F (pyflakes) | Real bugs | Unused imports, undefined names |
| I (isort) | Import order | `import argparse` placed after local imports |
| B (bugbear) | Likely bugs | `zip()` without `strict=True` silently drops elements |
| UP (pyupgrade) | Modern syntax | Python 3.12 idioms |

Import order convention (PEP 8): standard library → third-party → your own code,
separated by blank lines. `known-first-party = ["src"]` tells ruff that `src` is your
code. Without it, ruff groups `src` with `torch` and gives wrong advice.

`ruff check . --fix` fixes the safe findings automatically. Always read the result with
`git diff`.

### 14.2 ruff format (formatter)

A **formatter** rewrites layout (line breaks, spacing, quotes) to one consistent style.
Nobody spends time on layout, and diffs show only real changes. A common result:
long argument lists split one per line, with a trailing comma.

### 14.3 pre-commit

**pre-commit** runs checks automatically on every `git commit`, and stops the commit if
one fails. The configuration is `.pre-commit-config.yaml`:

| Hook | Purpose |
|---|---|
| `ruff-check --fix` | Lint and auto-fix |
| `ruff-format` | Format |
| `trailing-whitespace`, `end-of-file-fixer` | Whitespace in all file types |
| `check-yaml` | Catch broken YAML |
| `check-added-large-files --maxkb=500` | Safety net for the "no binaries in git" rule |

- `pre-commit install` writes `.git/hooks/pre-commit` (once per clone).
- `pre-commit run --all-files` checks the whole repo.
- "Failed, files were modified" means a hook fixed something: `git add` those files and
  commit again.
- The hook versions (`rev:`) match the pinned versions in `requirements.txt`, so the
  terminal and the hook apply identical rules.

---

## 15. Lessons learned

Mistakes made while building this stage, and what they teach:

| What happened | Cause | Lesson |
|---|---|---|
| `AttributeError: module 'torch' has no attribute 'Module'` | `import torch as nn` instead of `from torch import nn` | `import X as Y` renames a package; `from X import Y` imports a submodule |
| pytest crashed with `No module named 'yaml'` in `/opt/ros/...` | `PYTHONPATH` from ROS leaked into the venv | Activate with `activate_ml`; use `python -m pytest` |
| Edits "missing" from commits | Files edited in VS Code but not saved (● on the tab) | Save, then check `git status` before committing |
| `git add` staged nothing | `project.toml` instead of `pyproject.toml`: one wrong path aborts the whole `git add` | Read error messages; run `git status` before every commit |
| ruff gave wrong import advice | It never read the misnamed config file | Tools look for specific file names |
| Test accuracy lower than validation | Small test set plus validation selection bias | Check differences against the measurement uncertainty |
| Linear model stuck at 0.84 | No hidden layer means no non-linearity | Underfitting is diagnosed by train ≈ val, both low |

Engineering principles that came up repeatedly:

- **Reproducibility:** fixed seeds, pinned versions, data and checkpoints regenerated
  from code.
- **Separation:** one job per file; train/validation/test kept apart.
- **Verification:** predict a result before running, and compare. Tests turn manual
  checks into automatic ones.
- **Commit what produces outputs, not the outputs.**

---

## 16. Glossary

| Term | Meaning |
|---|---|
| Activation function | Non-linear function applied after a layer (here ReLU) |
| Autograd | PyTorch's automatic computation of gradients |
| Backpropagation | Computing all gradients by applying the chain rule backwards through the network |
| Batch / mini-batch | The subset of samples used for one parameter update |
| Bias | Learnable offset added to a neuron's weighted sum |
| Checkpoint | Saved model parameters (`state_dict`) |
| Cost function | Average loss over a batch or dataset |
| Cross-entropy | Classification loss: −log(probability of the correct class) |
| Data leakage | Evaluation data influencing training, which inflates results |
| Decision boundary | Where the model's predicted class changes |
| Epoch | One full pass over the training data |
| Forward propagation | Computing outputs from inputs with the current parameters |
| Generalisation | Performance on data not seen during training |
| Gradient | Derivative of the loss with respect to a parameter |
| Gradient descent | Updating parameters against the gradient: w ← w − η·∂L/∂w |
| Hidden layer | A layer between input and output |
| Hyperparameter | A setting chosen by you, not learned (learning rate, batch size, layer sizes, epochs) |
| Learning rate | Step size of each parameter update |
| Logits | Raw, unnormalised output scores of a classifier |
| Loss function | Error measure for one prediction |
| MLP | Multi-layer perceptron: stacked fully connected layers with activations |
| Overfitting | Fitting training data details that don't generalise |
| Parameter | A learnable value: weights and biases |
| ReLU | max(0, z) |
| Seed | Starting value that makes random numbers reproducible |
| Softmax | Converts logits into probabilities that sum to 1 |
| Standard error | Uncertainty of a measured value due to finite sample size |
| Underfitting | Model too simple to fit even the training data |
| Weight | Learnable factor scaling one input of a neuron |

---

## 17. Command reference

Run from `MLP/01_first_model/` after `activate_ml`:

```bash
python -m src.data                    # generate and plot the dataset
python -m src.visualize               # network diagram with initial weights
python -m src.train --hidden 16 16    # train the MLP, save checkpoint and loss curves
python -m src.train --hidden          # train the linear baseline
python -m src.plot_boundary           # decision boundaries side by side
python -m src.evaluate                # final test evaluation (once)
python -m pytest -v                   # run the 9 tests
```

From the repo root:

```bash
ruff check .                          # lint
ruff check . --fix                    # lint and auto-fix
ruff format .                         # format
pre-commit run --all-files            # all hooks on all files
git status                            # always before committing
git log --oneline --decorate          # history with tags
```
