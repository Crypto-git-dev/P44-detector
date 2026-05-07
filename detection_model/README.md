# Poker44 Chunk Bot Detector Starter

This starter trains a chunk-level bot probability model for Poker44 miner inference.

It is designed around the current miner contract:

- input: `DetectionSynapse(chunks=...)`
- each chunk: a list of sanitized hands
- output: one `risk_score` per chunk
- score: `P(chunk is bot)`

## Install

```bash
cd p44_miner_starter
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Build public benchmark inside Poker44-subnet

From the Poker44 subnet repo:

```bash
python scripts/publish/publish_public_benchmark.py --skip-wandb
```

Expected default:

```text
data/public_miner_benchmark.json.gz
```

## Train

```bash
python -m model.train \
  --data ~/Poker44-subnet/data/public_miner_benchmark.json.gz \
  --out artifacts/p44_chunk_detector.pt \
  --epochs 8 \
  --batch-size 64
```

## Use in miner inference

```python
from p44_miner_model.inference import Poker44BotDetector
from p44_miner_model.manifest import build_model_manifest

DETECTOR = Poker44BotDetector.load("artifacts/p44_chunk_detector.pt")
MANIFEST = build_model_manifest(
    repo_url="https://github.com/YOUR_NAME/YOUR_REPO",
    repo_commit="YOUR_COMMIT_SHA",
    artifact_path="artifacts/p44_chunk_detector.pt",
)

def forward(self, synapse):
    scores = DETECTOR.predict_chunks(synapse.chunks)
    synapse.risk_scores = scores
    synapse.predictions = [s >= 0.5 for s in scores]
    synapse.model_manifest = MANIFEST
    return synapse
```

## Important safety / compliance note

Train only on public benchmark data and your own legal synthetic simulations. Do not train on validator-only evaluation payloads, live `/internal/eval/current` batches, leaked data, or payload hashes.
