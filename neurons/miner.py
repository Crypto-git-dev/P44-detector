"""Poker44 miner using trained structured-action hierarchical bot detection model."""

import hashlib
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

MODEL_REPO_PATH = "detection_model"

from detection_model.model.inference import Poker44BotDetector

def _sha256_file(path: str | Path) -> str:
    """Return SHA256 hash for a file, or empty string if unavailable."""
    path = Path(path).expanduser()

    if not path.exists() or not path.is_file():
        return ""

    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _existing_paths(paths: list[str | Path]) -> list[Path]:
    """Return only paths that exist and are files."""
    existing: list[Path] = []

    for path in paths:
        p = Path(path).expanduser()

        if p.exists() and p.is_file():
            existing.append(p)

    return existing


class Miner(BaseMinerNeuron):
    """Scores each DetectionSynapse chunk with a trained Poker44 model."""

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)

        bt.logging.info("🤖 Poker44 structured-action trained-model miner started")

        repo_root = Path(__file__).resolve().parents[1]
        model_repo_root = Path(MODEL_REPO_PATH).expanduser()

        self.model_path = "detection_model/artifacts/p44_action_vector_gru_window_v4_05_07.pt"

        self.xgb_path = os.getenv("P44_XGB_PATH", "")

        # These must match how the model was trained.
        self.window_hands = int(os.getenv("P44_WINDOW_HANDS", "60"))
        self.window_stride = int(os.getenv("P44_WINDOW_STRIDE", "10"))
        self.window_agg = os.getenv("P44_WINDOW_AGG", "mean").lower()

        self.model_device = os.getenv("P44_MODEL_DEVICE", "cpu")
        self.inference_batch_size = int(os.getenv("P44_INFERENCE_BATCH_SIZE", "64"))
        self.prediction_threshold = float(os.getenv("P44_PREDICTION_THRESHOLD", "0.5"))
        self.temperature = float(os.getenv("P44_LOGITS_TEMPERATURE", "1.0"))

        self.detector = None

        self.model_manifest = self._build_model_manifest(
            repo_root=repo_root,
            model_repo_root=model_repo_root,
        )

        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)

        self._log_manifest_startup(repo_root)
        self._load_trained_model()

        bt.logging.info(f"Axon created: {self.axon}")

    def _build_model_manifest(
        self,
        repo_root: Path,
        model_repo_root: Path,
    ) -> dict:
        """Build miner transparency manifest for the deployed model."""

        model_artifact_path = Path(self.model_path).expanduser()

        implementation_files = _existing_paths(
            [
                Path(__file__).resolve(),

                # External model implementation files.
                model_repo_root / "model" / "inference.py",
                model_repo_root / "model" / "hierarchical_model.py",
                model_repo_root / "model" / "hierarchical_dataset.py",
                model_repo_root / "model" / "action_vectorizer.py",
                model_repo_root / "model" / "features.py",
                model_repo_root / "model" / "dataset.py",
            ]
        )

        artifact_sha256 = os.getenv(
            "POKER44_MODEL_ARTIFACT_SHA256",
            _sha256_file(model_artifact_path),
        )

        return build_local_model_manifest(
            repo_root=repo_root,
            implementation_files=implementation_files,
            defaults={
                "open_source": True,

                "model_name": os.getenv(
                    "P44_MANIFEST_MODEL_NAME",
                    "p44-structured-action-chunk-detector",
                ),

                "model_version": os.getenv(
                    "P44_MANIFEST_MODEL_VERSION",
                    "1.3.0",
                ),

                "framework": os.getenv(
                    "P44_MANIFEST_FRAMEWORK",
                    "pytorch-structured-action-transformer-gru",
                ),

                "license": os.getenv(
                    "P44_MANIFEST_LICENSE",
                    "MIT",
                ),

                "repo_url": os.getenv(
                    "P44_MANIFEST_REPO_URL",
                    "https://github.com/Crypto-git-dev/P44-detector",
                ),

                "repo_commit": os.getenv(
                    "P44_MANIFEST_REPO_COMMIT",
                    os.getenv(
                        "P44_MODEL_REPO_COMMIT",
                        "023c9ccde49cface05b44d4ae59d1621a4c8e6ac",
                    ),
                ),

                # Use this if you publish the .pt artifact somewhere.
                # Leave blank for local-only artifacts.
                "artifact_url": os.getenv(
                    "P44_MANIFEST_ARTIFACT_URL",
                    "",
                ),

                "artifact_sha256": artifact_sha256,

                "model_card_url": os.getenv(
                    "P44_MANIFEST_MODEL_CARD_URL",
                    "",
                ),

                "training_data_statement": (
                    "Trained only on the public Poker44 miner benchmark and local "
                    "sliding-window augmentation generated from that public benchmark. "
                    "No validator-only live evaluation data, hidden validator labels, "
                    "leaked validator payloads, or validator-private data were used."
                ),

                "training_data_sources": [
                    "data/public_miner_benchmark.json.gz",
                    "local sliding-window augmentation of public benchmark chunks",
                ],

                "private_data_attestation": (
                    "This miner does not train on validator-only evaluation data, "
                    "live eval batches, hidden validator labels, leaked validator "
                    "payloads, or any private validator data."
                ),

                "inference_mode": "remote",

                "notes": (
                    "Structured-action Poker44 bot detector. Raw poker action objects "
                    "are converted into categorical action fields plus numeric action "
                    "features using ActionVectorizer. A small Transformer encodes actions "
                    "inside each hand. A GRU encodes the hand sequence inside each window. "
                    "The classifier outputs P(bot), where higher risk_score means more "
                    "bot-like. Miner inference splits validator chunks into sliding windows "
                    f"with window_hands={self.window_hands}, "
                    f"window_stride={self.window_stride}, "
                    f"aggregation={self.window_agg}. "
                    f"Local artifact path: {model_artifact_path}"
                ),
            },
        )

    def _load_trained_model(self) -> None:
        """Load the trained detector. If loading fails, fallback heuristic remains available."""

        if Poker44BotDetector is None:
            bt.logging.error(
                f"Could not import Poker44BotDetector from P44_MODEL_REPO={MODEL_REPO_PATH}"
            )
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
            bt.logging.info(f"XGBoost path: {self.xgb_path or '[disabled]'}")

            self.detector = Poker44BotDetector.load(
                model_path,
                device=self.model_device,
                temperature=self.temperature,
                xgb_path=self.xgb_path or None,
            )

            # Environment threshold should override artifact threshold.
            env_threshold = os.getenv("P44_PREDICTION_THRESHOLD")

            if env_threshold is not None:
                self.prediction_threshold = float(env_threshold)
            elif hasattr(self.detector, "threshold"):
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

        bt.logging.info(
            f"Manifest digest={self.manifest_digest} "
            f"inference_mode={self.model_manifest.get('inference_mode', '')}"
        )

        bt.logging.info(
            f"Manifest artifact | url={self.model_manifest.get('artifact_url', '')} "
            f"sha256={self.model_manifest.get('artifact_sha256', '')}"
        )

        bt.logging.info(
            "Miner prep tooling available | "
            f"benchmark_doc={repo_root / 'docs' / 'public-benchmark.md'} "
            f"miner_doc={repo_root / 'docs' / 'miner.md'} "
            f"anti_leakage_doc={repo_root / 'docs' / 'anti-leakage.md'}"
        )

    def _make_windows_for_chunk(self, chunk: list[dict]) -> list[list[dict]]:
        """
        Convert one validator chunk into fixed-length consecutive windows.

        Example:
            chunk = [h1, h2, h3, h4, h5, h6]
            window_hands = 4
            stride = 1

            windows:
                [h1, h2, h3, h4]
                [h2, h3, h4, h5]
                [h3, h4, h5, h6]
        """

        if not chunk:
            return []

        n = len(chunk)

        if self.window_hands <= 0:
            return [chunk]

        if self.window_stride <= 0:
            self.window_stride = 1

        # If the chunk is shorter than the trained window length,
        # still score the original chunk rather than returning no score.
        if n < self.window_hands:
            return [chunk]

        windows: list[list[dict]] = []

        for start in range(0, n - self.window_hands + 1, self.window_stride):
            end = start + self.window_hands
            windows.append(chunk[start:end])

        return windows or [chunk]

    def _aggregate_window_scores(self, scores: list[float]) -> float:
        """
        Aggregate several window scores into one score for the original chunk.

        Higher score = more bot-like.
        """

        if not scores:
            return 0.5

        scores = [float(s) for s in scores]

        if self.window_agg == "max":
            # Sensitive: if any window looks bot-like, chunk becomes bot-like.
            value = max(scores)

        elif self.window_agg == "min":
            # Conservative: all windows must look bot-like for chunk to be bot-like.
            value = min(scores)

        elif self.window_agg == "mean":
            # Stable: average behavior across all windows.
            value = sum(scores) / len(scores)

        elif self.window_agg == "topk_mean":
            # Balanced: focus on the most suspicious windows without using only max.
            k = min(3, len(scores))
            top_scores = sorted(scores, reverse=True)[:k]
            value = sum(top_scores) / len(top_scores)

        else:
            # Safe default.
            value = sum(scores) / len(scores)

        return round(max(0.0, min(1.0, value)), 6)

    def _predict_original_chunks_with_windows(
        self,
        chunks: list[list[dict]],
    ) -> list[float]:
        """
        Score original validator chunks using sliding windows.

        Returns:
            one final score per original chunk.
        """

        if self.detector is None:
            return [self.score_chunk(chunk) for chunk in chunks]

        all_windows: list[list[dict]] = []
        window_ranges: list[tuple[int, int]] = []

        cursor = 0

        for chunk in chunks:
            windows = self._make_windows_for_chunk(chunk)

            start = cursor
            all_windows.extend(windows)
            cursor += len(windows)
            end = cursor

            window_ranges.append((start, end))

        if not all_windows:
            return [0.5 for _ in chunks]

        window_scores = self.detector.predict_chunks(
            all_windows,
            batch_size=self.inference_batch_size,
        )

        final_scores: list[float] = []

        for start, end in window_ranges:
            scores_for_chunk = window_scores[start:end]
            final_scores.append(
                self._aggregate_window_scores(scores_for_chunk)
            )

        return final_scores

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        """
        Score each validator chunk.

        Output rule:
            len(synapse.risk_scores) == len(synapse.chunks)

        Convention:
            risk_score close to 0 = human-like
            risk_score close to 1 = bot-like
        """

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
                scores = self._predict_original_chunks_with_windows(chunks)

            if len(scores) != len(chunks):
                raise ValueError(
                    f"Wrong score count: chunks={len(chunks)}, scores={len(scores)}"
                )

            clean_scores: list[float] = []

            for score in scores:
                score = float(score)
                score = max(0.0, min(1.0, score))
                clean_scores.append(round(score, 6))

            synapse.risk_scores = clean_scores

            synapse.predictions = [
                score >= self.prediction_threshold
                for score in clean_scores
            ]

            synapse.model_manifest = dict(self.model_manifest)

            bt.logging.info(
                f"Scored {len(chunks)} chunks with "
                f"{'trained model' if self.detector else 'heuristic fallback'} | "
                f"window_hands={self.window_hands} "
                f"stride={self.window_stride} "
                f"agg={self.window_agg} | "
                f"preview_scores={clean_scores} | "
                f"preview_predictions={synapse.predictions}"
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

            # Safer fallback: don't mark neutral fallback as bot.
            synapse.predictions = [False for _ in fallback_scores]

            synapse.model_manifest = dict(self.model_manifest)

            bt.logging.info(f"Returned fallback scores for {len(chunks)} chunks.")

            return synapse

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    @classmethod
    def _score_hand(cls, hand: dict) -> float:
        """Heuristic fallback only."""

        actions = hand.get("actions") or []
        players = hand.get("players") or []
        streets = hand.get("streets") or []
        outcome = hand.get("outcome") or {}

        action_counts = Counter(action.get("action_type") for action in actions)

        meaningful_actions = max(
            1,
            sum(
                action_counts.get(kind, 0)
                for kind in ("call", "check", "bet", "raise", "fold")
            ),
        )

        call_ratio = action_counts.get("call", 0) / meaningful_actions
        check_ratio = action_counts.get("check", 0) / meaningful_actions
        fold_ratio = action_counts.get("fold", 0) / meaningful_actions
        raise_ratio = action_counts.get("raise", 0) / meaningful_actions

        street_depth = len(streets) / 3.0
        showdown_flag = 1.0 if outcome.get("showdown") else 0.0

        player_count_signal = 0.0

        if players:
            player_count_signal = (6 - min(len(players), 6)) / 4.0

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
        """Heuristic fallback chunk score."""

        if not chunk:
            return 0.5

        hand_scores = [
            cls._score_hand(hand)
            for hand in chunk
        ]

        return round(cls._clamp01(sum(hand_scores) / len(hand_scores)), 6)

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Poker44 structured-action trained-model miner running...")

        while True:
            bt.logging.info(
                f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}"
            )
            time.sleep(5 * 60)