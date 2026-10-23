# ML Engineering Path: Roadmap

Goal: build complete end-to-end ML workflows from scratch, for four architectures.
Started: 21 Oct 2026. Update the status column when a stage starts and finishes.

## Decisions
- Library: PyTorch (robotics and VLA stack is PyTorch; ONNX export covers Jetson and ROS 2)
- Repo: one repo, one top-level folder per architecture, four numbered stages per folder
- Environment: one shared venv + root requirements.txt; each `04_end_to_end` stage has its own requirements.txt and Dockerfile
- Hardware: CPU for MLP and RNN; RTX 4090 Laptop (16 GB VRAM) for CNN and Transformer

## Layout
```
MchineL_ws/
├── requirements.txt          # shared dependencies
├── data/                     # datasets, shared (not in git)
├── docs/ROADMAP.md
├── MLP/
│   ├── 01_first_model/       # build, visualise, train, test
│   ├── 02_regression_classification/
│   ├── 03_pipeline_features/ # feature engineering + transforms on a standard dataset
│   └── 04_end_to_end/        # database, deployment, versioning, API
├── CNN/          (same 01-04)
├── RNN/          (same 01-04)
└── Transformer/  (same 01-04)
```
Each stage folder contains `src/`, `tests/` and `README.md` (goal, how to run, results).

## Stages (same four for every architecture)
1. **01 First model**: build, visualise, train, test
2. **02 Regression and classification**: one model for each task
3. **03 Pipeline with feature engineering and transforms**: on a standard dataset
4. **04 End-to-end**: database integration, deployment, version control, API

## Plan
| # | Architecture | Stage | Key tools introduced | Started | Status |
|---|---|---|---|---|---|
| 1 | MLP | 01 First model | git CLI, venv, pytest | 21 Oct | done |
| 2 | MLP | 02 Regression + classification | TensorBoard, ruff, pre-commit | | not started |
| 3 | MLP | 03 Pipeline + features | scikit-learn transforms, Optuna | | not started |
| 4 | MLP | 04 End-to-end | PostgreSQL, FastAPI, Docker, DVC, MLflow | | not started |
| 5 | CNN | 01 First model | GPU training | | not started |
| 6 | CNN | 02 Regression + classification | | | not started |
| 7 | CNN | 03 Pipeline + features | torchvision transforms, data augmentation | | not started |
| 8 | CNN | 04 End-to-end | ONNX Runtime, MLflow registry, GitHub Actions | | not started |
| 9 | RNN | 01 First model (LSTM/GRU) | | | not started |
| 10 | RNN | 02 Regression + classification | | | not started |
| 11 | RNN | 03 Pipeline + features | | | not started |
| 12 | RNN | 04 End-to-end | MongoDB | | not started |
| 13 | Transformer | 01 First model | | | not started |
| 14 | Transformer | 02 Regression + classification | | | not started |
| 15 | Transformer | 03 Pipeline + features | | | not started |
| 16 | Transformer | 04 End-to-end | Ansible, Kubernetes | | not started |

## Rules
- I type all code. Claude coaches, reviews and explains.
- Commit on main. Tag each finished stage as `<arch>-<nn>-done` (e.g. `mlp-01-done`).
- Test set is used once per stage. Seeds are fixed.
- No data or model binaries in git.

## Log
| Date | Stage | Result | Next |
|---|---|---|---|
| 21 Oct | Setup | Tools checked, venv, requirements.txt, first commit | MLP 01 |
| 23 Oct | MLP 01 | MLP [16, 16]: test acc 0.947 (val 0.967); linear baseline val 0.84; 9 tests passing | MLP 02 |