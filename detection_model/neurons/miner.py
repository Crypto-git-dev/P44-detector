"""Poker44 miner using trained hierarchical chunk-level bot detection model."""

from __future__ import annotations

import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Tuple

import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse


MODEL_REPO_PATH = os.getenv("P44_MODEL_REPO", "/root/Poker-bot-detection-model")
if MODEL_REPO_PATH and MODEL_REPO_PATH not in sys.path:
    sys.path.append(MODEL_REPO_PATH)

try:
    from model.inference import Poker44BotDetector
except Exception as exc:  # keep miner alive if import path is not ready
    Poker44BotDetector = None
    MODEL_IMPORT_ERROR = exc
else:
    MODEL_IMPORT_ERROR = None


class Miner(BaseMinerNeuron):
    """Scores each DetectionSynapse chunk independently with the trained model."""

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        bt.logging.info("🤖 Poker44 hierarchical trained-model miner started")
        repo_root = Path(__file__).resolve().parents[1]

        self.model_path = os.getenv(
            "P44_MODEL_PATH",
            "/root/Poker-bot-detection-model/artifacts/p44_hierarchical_detector.pt",
        )
        self.model_device = os.getenv("P44_MODEL_DEVICE", "cpu")
        self.inference_batch_size = int(os.getenv("P44_INFERENCE_BATCH_SIZE", "64"))
        self.prediction_threshold = float(os.getenv("P44_PREDICTION_THRESHOLD", "0.5"))
        self.temperature = float(os.getenv("P44_LOGITS_TEMPERATURE", "1.0"))
        self.detector = None

        self.model_manifest = build_local_model_manifest(
            repo_root=repo_root,
            implementation_files=[Path(__file__).resolve()],
            defaults={
                "model_name": "p44-hierarchical-action-hand-chunk-detector",
                "model_version": "1.0.0",
                "framework": "pytorch-hierarchical-transformer",
                "license": "MIT",
                "repo_url": os.getenv("P44_MODEL_REPO_URL", "https://github.com/YOUR_NAME/YOUR_REPO"),
                "notes": "Hierarchical chunk-level detector: action tokens -> hand encoder -> chunk probability.",
                "open_source": True,
                "inference_mode": "remote",
                "training_data_statement": (
                    "Trained on the public Poker44 miner benchmark and local chunk-prefix/window augmentation only. "
                    "No validator-only live evaluation data was used."
                ),
                "training_data_sources": ["data/public_miner_benchmark.json.gz"],
                "data_attestation": (
                    "This miner does not train on validator-only evaluation data, live eval batches, "
                    "hidden validator labels, or leaked payloads."
                ),
                "private_data_attestation": (
                    "This miner does not train on validator-only evaluation data, live eval batches, "
                    "hidden validator labels, or leaked payloads."
                ),
                "artifact_path": self.model_path,
            },
        )
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        self._log_manifest_startup(repo_root)
        self._load_trained_model()
        bt.logging.info(f"Axon created: {self.axon}")

    def _load_trained_model(self) -> None:
        if Poker44BotDetector is None:
            bt.logging.error(f"Could not import Poker44BotDetector from P44_MODEL_REPO={MODEL_REPO_PATH}")
            bt.logging.error(f"Import error: {MODEL_IMPORT_ERROR}")
            bt.logging.error("Miner will use heuristic fallback.")
            return
        model_path = Path(self.model_path).expanduser()
        if not model_path.exists():
            bt.logging.error(f"Model artifact does not exist: {model_path}")
            bt.logging.error("Set P44_MODEL_PATH correctly. Miner will use heuristic fallback.")
            return
        try:
            bt.logging.info(f"Loading trained Poker44 model from: {model_path}")
            bt.logging.info(f"Model device: {self.model_device}")
            bt.logging.info(f"Inference batch size: {self.inference_batch_size}")
            bt.logging.info(f"Logit temperature: {self.temperature}")
            self.detector = Poker44BotDetector.load(model_path, device=self.model_device, temperature=self.temperature)
            if hasattr(self.detector, "threshold"):
                self.prediction_threshold = float(self.detector.threshold)
            bt.logging.info("✅ Trained Poker44 model loaded successfully")
            bt.logging.info(f"Prediction threshold: {self.prediction_threshold}")
        except Exception as exc:
            bt.logging.error(f"Failed to load trained model: {exc}")
            bt.logging.error("Miner will use heuristic fallback.")
            self.detector = None

    def _log_manifest_startup(self, repo_root: Path) -> None:
        bt.logging.info("Open-sourced miner manifest standard active for this miner.")
        bt.logging.info(
            f"Miner transparency status: {self.manifest_compliance['status']} "
            f"(missing_fields={self.manifest_compliance['missing_fields']})"
        )
        bt.logging.info(
            f"Manifest summary | model={self.model_manifest.get('model_name', '')} "
            f"version={self.model_manifest.get('model_version', '')} "
            f"repo={self.model_manifest.get('repo_url', '')} "
            f"commit={self.model_manifest.get('repo_commit', '')} "
            f"open_source={self.model_manifest.get('open_source')}"
        )
        bt.logging.info(f"Manifest digest={self.manifest_digest} inference_mode={self.model_manifest.get('inference_mode', '')}")
        bt.logging.info(
            "Miner prep tooling available | "
            f"benchmark_doc={repo_root / 'docs' / 'public-benchmark.md'} "
            f"miner_doc={repo_root / 'docs' / 'miner.md'} "
            f"anti_leakage_doc={repo_root / 'docs' / 'anti-leakage.md'}"
        )

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        chunks = synapse.chunks or []
        if not chunks:
            synapse.risk_scores = []
            synapse.predictions = []
            synapse.model_manifest = dict(self.model_manifest)
            return synapse
        try:
            if self.detector is None:
                bt.logging.warning("Trained model is not loaded. Using heuristic fallback.")
                scores = [self.score_chunk(chunk) for chunk in chunks]
            else:
                scores = self.detector.predict_chunks(chunks, batch_size=self.inference_batch_size)
            if len(scores) != len(chunks):
                raise ValueError(f"Wrong score count: chunks={len(chunks)}, scores={len(scores)}")
            clean_scores = [round(max(0.0, min(1.0, float(s))), 6) for s in scores]
            synapse.risk_scores = clean_scores
            synapse.predictions = [score >= self.prediction_threshold for score in clean_scores]
            synapse.model_manifest = dict(self.model_manifest)
            bt.logging.info(
                f"Scored {len(chunks)} chunks with {'trained model' if self.detector else 'heuristic fallback'} | "
                f"preview_scores={clean_scores[:5]} | preview_predictions={synapse.predictions[:5]}"
            )
            return synapse
        except Exception as exc:
            bt.logging.error(f"Trained model inference failed: {exc}")
            try:
                fallback_scores = [self.score_chunk(chunk) for chunk in chunks]
            except Exception as fallback_exc:
                bt.logging.error(f"Heuristic fallback also failed: {fallback_exc}")
                fallback_scores = [0.5 for _ in chunks]
            synapse.risk_scores = fallback_scores
            synapse.predictions = [score >= self.prediction_threshold for score in fallback_scores]
            synapse.model_manifest = dict(self.model_manifest)
            bt.logging.info(f"Returned fallback scores for {len(chunks)} chunks.")
            return synapse

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    @classmethod
    def _score_hand(cls, hand: dict) -> float:
        actions = hand.get("actions") or []
        players = hand.get("players") or []
        streets = hand.get("streets") or []
        outcome = hand.get("outcome") or {}
        action_counts = Counter(action.get("action_type") for action in actions)
        meaningful_actions = max(1, sum(action_counts.get(k, 0) for k in ("call", "check", "bet", "raise", "fold")))
        call_ratio = action_counts.get("call", 0) / meaningful_actions
        check_ratio = action_counts.get("check", 0) / meaningful_actions
        fold_ratio = action_counts.get("fold", 0) / meaningful_actions
        raise_ratio = action_counts.get("raise", 0) / meaningful_actions
        street_depth = len(streets) / 3.0
        showdown_flag = 1.0 if outcome.get("showdown") else 0.0
        player_count_signal = (6 - min(len(players), 6)) / 4.0 if players else 0.0
        score = 0.0
        score += 0.32 * street_depth
        score += 0.22 * showdown_flag
        score += 0.18 * cls._clamp01(call_ratio / 0.35)
        score += 0.12 * cls._clamp01(check_ratio / 0.30)
        score += 0.08 * cls._clamp01(player_count_signal)
        score -= 0.18 * cls._clamp01(fold_ratio / 0.55)
        score -= 0.10 * cls._clamp01(raise_ratio / 0.20)
        return cls._clamp01(score)

    @classmethod
    def score_chunk(cls, chunk: list[dict]) -> float:
        if not chunk:
            return 0.5
        hand_scores = [cls._score_hand(hand) for hand in chunk]
        return round(cls._clamp01(sum(hand_scores) / len(hand_scores)), 6)

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Poker44 hierarchical trained-model miner running...")
        while True:
            bt.logging.info(f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}")
            time.sleep(5 * 60)
