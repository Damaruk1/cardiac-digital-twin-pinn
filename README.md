# Cardiac Digital Twin using Physics-Informed Neural Networks

Research-grade pipeline: ECG → Signal Processing → AI Diagnosis →
Anatomical Mapping → PINN → 3D Personalized Heart Simulation →
Clinical Biomarkers.

## Status
🚧 Phase 1: Project scaffolding — in progress.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python -m src.main
```

## Test

```bash
pytest tests/ -v
```

## Project Structure

```
configs/    Project configuration (YAML)
src/        Source code
  utils/    Logging, config loading, shared helpers
tests/      Unit tests
data/       Raw and processed data (not committed to git)
logs/       Runtime logs (not committed to git)
```

## Roadmap

| Phase | Milestone |
|---|---|
| 1 | Project setup ✅ |
| 2 | Understanding ECG |
| 3 | Signal preprocessing |
| 4 | Visualization |
| 5 | Dataset loading |
| 6 | CNN |
| 7 | Transformer |
| 8 | Training |
| 9 | Evaluation |
| 10 | Explainability |
| 11 | Anatomical mapping |
| 12 | PINN theory |
| 13 | PINN implementation |
| 14 | Simulation |
| 15 | Visualization |
| 16 | Clinical metrics |
| 17 | Deployment |
