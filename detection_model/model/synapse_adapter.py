from __future__ import annotations

from typing import Any, Dict

from .inference import Poker44BotDetector


def score_detection_synapse(synapse: Any, detector: Poker44BotDetector, model_manifest: Dict[str, Any] | None = None) -> Any:
    """Attach model predictions to a Poker44 DetectionSynapse-like object.

    Use this from neurons/miner.py. The object is kept generic so this starter does not
    need to import Poker44's concrete DetectionSynapse class.
    """
    chunks = getattr(synapse, "chunks", None) or []
    scores = detector.predict_chunks(chunks)
    synapse.risk_scores = scores
    synapse.predictions = [score >= detector.threshold for score in scores]
    if model_manifest is not None:
        synapse.model_manifest = model_manifest
    return synapse
