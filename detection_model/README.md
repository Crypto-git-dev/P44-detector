# Poker44 Detection Model

Updated detection model project using the first hierarchical model architecture with a final XGBoost head.

## What changed

- Restored the first-model backbone:
  - action categorical embeddings
  - compact numeric action projection
  - action Transformer with CLS token per hand
  - GRU over hands
  - attention pooling for chunk embedding
- Removed the extra `model/train_xgboost.py` step.
- `model/train_hierarchical.py` now trains the neural encoder and then trains/saves XGBoost as the final classifier inside the same `.pt` artifact.
- Simplified feature engineering to compact, essential chunk-level signals only.
- Removed wide per-seat hand feature expansion and hand-feature fusion/gating.
- Updated inference, benchmark, simulation, tools compatibility, and dashboard to use the embedded XGBoost head automatically.
- Cleaned dashboard run logs/history from the packaged project.

## Install

```bash
cd detection_model
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```bash
python -m model.train_hierarchical \
  --data data/public_miner_benchmark.json.gz \
  --out artifacts/p44_first_arch_xgb.pt \
  --epochs 60 \
  --batch-size 8 \
  --augment-windows \
  --augment-validation-windows \
  --window-hands 4 \
  --window-stride 1 \
  --overwrite
```

The output artifact contains both:

- the neural hierarchical encoder weights
- the final XGBoost classifier head

## Predict

```bash
python -m model.simulate_result \
  --data data/chunks.json \
  --model artifacts/p44_first_arch_xgb.pt \
  --out-csv outputs/predictions.csv
```

No separate `--xgb-model` is required for new artifacts because XGBoost is embedded in the `.pt` file.

## Benchmark

```bash
python -m model.evaluate_benchmark \
  --data data/public_miner_benchmark.json.gz \
  --model artifacts/p44_first_arch_xgb.pt \
  --split all \
  --out-csv outputs/benchmark_predictions.csv \
  --out-json outputs/benchmark_metrics.json
```

## Dashboard

```bash
streamlit run dashboard/training_dashboard.py
```

The dashboard Train tab now maps to the single integrated training command and exposes the final XGBoost parameters there.

## Compliance note

Train only on public benchmark data and your own legal synthetic simulations. Do not train on validator-only evaluation payloads, live `/internal/eval/current` batches, leaked data, or payload hashes.
