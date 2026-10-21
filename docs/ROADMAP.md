# ML Engineering Path: Roadmap

Goal: build a complete end-to-end ML workflow from scratch, one project per week.
Started: 21 Oct 2026. Update the status column at the end of each week.

## Decisions
- Library: PyTorch (robotics and VLA stack is PyTorch; ONNX export covers Jetson and ROS 2)
- Repo: one monorepo for P1-P7, a separate repo for P8
- Environment: venv + requirements.txt, CPU until P6, RTX 4090 from P6

## Plan
| # | Project | Week | Key tools introduced | Status |
|---|---|---|---|---|
| P1 | First neural network (MLP): build, visualise, train, test | 21-27 Oct | git CLI, venv, pytest | in progress |
| P2 | Classification with a deep network | 28 Oct-3 Nov | TensorBoard/MLflow, ruff, pre-commit | not started |
| P3 | Regression with a deep network | 4-10 Nov | Optuna, GitHub Actions | not started |
| P4 | Pipeline from training to deployment | 11-17 Nov | scikit-learn + skorch, DVC, FastAPI, Docker intro | not started |
| P5 | RNN (LSTM/GRU) on sequence data | 18-24 Nov | | not started |
| P6 | CNN on images | 25 Nov-1 Dec | GPU training | not started |
| P7 | CNN pipeline and export | 2-8 Dec | MLflow registry, ONNX Runtime | not started |
| P8 | Production service (2 weeks) | 9-22 Dec | PostgreSQL, MongoDB, Docker, FastAPI, Ansible, Kubernetes | not started |

## Rules
- I type all code. Claude coaches, reviews and explains.
- One branch per project, merged with --no-ff, tagged p1-done when finished.
- Test set is used once per project. Seeds are fixed.
- No data or model binaries in git.

## Weekly log
| Week | Project | Result | Next |
|---|---|---|---|
| 21-27 Oct | P1 | | |