from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .action_vectorizer import ActionVectorizer
from .features import FeatureVectorizer
from .hierarchical_model import HierarchicalChunkClassifier


class Poker44BotDetector:
    """
    Inference wrapper for structured-action hierarchical Poker44 model.

    Input:
        chunks: List[List[Dict[str, Any]]]

    Meaning:
        chunks = [
            chunk_0,  # List[hand_dict]
            chunk_1,
            ...
        ]

    Output:
        one bot-risk score per chunk.

    Score convention:
        0.0 = human-like
        1.0 = bot-like
    """

    def __init__(
        self,
        model: HierarchicalChunkClassifier,
        action_vectorizer: ActionVectorizer,
        vectorizer: FeatureVectorizer,
        threshold: float = 0.5,
        device: str | torch.device | None = None,
        temperature: float = 1.0,
        xgb_model: Any = None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = model.to(self.device).eval()
        self.action_vectorizer = action_vectorizer
        self.vectorizer = vectorizer

        self.threshold = float(threshold)
        self.temperature = float(temperature)

        if self.temperature <= 0:
            self.temperature = 1.0

        self.xgb_model = xgb_model

        model_config = getattr(self.model, "config", {}) or {}

        self.max_hands = int(
            model_config.get(
                "max_hands",
                getattr(self.model, "max_hands", 4),
            )
        )

        self.numeric_dim = int(self.action_vectorizer.numeric_dim)

    @classmethod
    def load(
        cls,
        artifact_path: str | Path,
        device: str | torch.device | None = None,
        temperature: float = 1.0,
        xgb_path: str | Path | None = None,
    ) -> "Poker44BotDetector":
        artifact_path = Path(artifact_path).expanduser()

        if not artifact_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {artifact_path}")

        artifact = torch.load(
            artifact_path,
            map_location="cpu",
        )

        if "action_vectorizer" not in artifact:
            raise KeyError(
                "Artifact does not contain 'action_vectorizer'. "
                "This inference.py is for the structured-action model. "
                f"Artifact keys: {list(artifact.keys())}"
            )

        if "vectorizer" not in artifact:
            raise KeyError(
                "Artifact does not contain 'vectorizer'. "
                f"Artifact keys: {list(artifact.keys())}"
            )

        if "model_config" not in artifact:
            raise KeyError(
                "Artifact does not contain 'model_config'. "
                f"Artifact keys: {list(artifact.keys())}"
            )

        action_vectorizer = ActionVectorizer.from_state_dict(
            artifact["action_vectorizer"]
        )

        vectorizer = FeatureVectorizer.from_state_dict(
            artifact["vectorizer"]
        )

        model_config = dict(artifact["model_config"])

        model = HierarchicalChunkClassifier(**model_config)

        model_state = (
            artifact.get("model_state_dict")
            or artifact.get("model")
            or artifact.get("state_dict")
        )

        if model_state is None:
            raise KeyError(
                "Artifact does not contain model weights. "
                f"Artifact keys: {list(artifact.keys())}"
            )

        model.load_state_dict(model_state)

        xgb_model = None
        threshold = float(artifact.get("threshold", 0.5))

        if xgb_path:
            try:
                import joblib
            except ImportError as exc:
                raise ImportError(
                    "joblib is required to load XGBoost model. "
                    "Install with: pip install joblib xgboost"
                ) from exc

            xgb_path = Path(xgb_path).expanduser()

            if not xgb_path.exists():
                raise FileNotFoundError(f"XGBoost model not found: {xgb_path}")

            xgb_payload = joblib.load(xgb_path)

            if isinstance(xgb_payload, dict):
                xgb_model = xgb_payload.get("xgb_model") or xgb_payload.get("model")

                if "threshold" in xgb_payload:
                    threshold = float(xgb_payload["threshold"])
            else:
                xgb_model = xgb_payload

            if xgb_model is None:
                raise KeyError(
                    f"Could not find XGBoost model inside {xgb_path}"
                )

        return cls(
            model=model,
            action_vectorizer=action_vectorizer,
            vectorizer=vectorizer,
            threshold=threshold,
            device=device,
            temperature=temperature,
            xgb_model=xgb_model,
        )

    def _make_inference_batch(
        self,
        chunks: List[List[Dict[str, Any]]],
    ) -> Dict[str, torch.Tensor]:
        """
        Convert raw chunks into structured model tensors.

        Output tensors:
            action_cat:  [batch, hands, actions, 3]
            action_num:  [batch, hands, actions, numeric_dim]
            action_mask: [batch, hands, actions]
            hand_mask:   [batch, hands]
            features:    [batch, feature_dim]
        """

        if not chunks:
            raise ValueError("No chunks provided for inference.")

        encoded_cat: List[List[List[List[int]]]] = []
        encoded_num: List[List[List[List[float]]]] = []

        for chunk in chunks:
            cat_hands, num_hands = self.action_vectorizer.encode_chunk(
                chunk,
                max_hands=self.max_hands,
            )

            encoded_cat.append(cat_hands)
            encoded_num.append(num_hands)

        batch_size = len(chunks)

        max_hands = max(len(cat_hands) for cat_hands in encoded_cat)
        max_hands = max(1, max_hands)

        max_actions = 1

        for cat_hands in encoded_cat:
            for cat_rows in cat_hands:
                max_actions = max(max_actions, len(cat_rows))

        action_cat = torch.zeros(
            size=(batch_size, max_hands, max_actions, 3),
            dtype=torch.long,
        )

        action_num = torch.zeros(
            size=(batch_size, max_hands, max_actions, self.numeric_dim),
            dtype=torch.float32,
        )

        action_mask = torch.zeros(
            size=(batch_size, max_hands, max_actions),
            dtype=torch.bool,
        )

        hand_mask = torch.zeros(
            size=(batch_size, max_hands),
            dtype=torch.bool,
        )

        for batch_idx, (cat_hands, num_hands) in enumerate(zip(encoded_cat, encoded_num)):
            for hand_idx, (cat_rows, num_rows) in enumerate(zip(cat_hands, num_hands)):
                if hand_idx >= max_hands:
                    break

                length = min(len(cat_rows), max_actions)

                if length <= 0:
                    continue

                cat_arr = np.asarray(cat_rows[:length], dtype=np.int64)

                if cat_arr.ndim != 2 or cat_arr.shape[1] != 3:
                    raise ValueError(
                        f"Invalid categorical action shape: {cat_arr.shape}. "
                        "Expected [actions, 3]."
                    )

                num_arr = np.asarray(num_rows[:length], dtype=np.float32)

                if num_arr.ndim != 2 or num_arr.shape[1] != self.numeric_dim:
                    raise ValueError(
                        f"Invalid numeric action shape: {num_arr.shape}. "
                        f"Expected [actions, {self.numeric_dim}]."
                    )

                action_cat[batch_idx, hand_idx, :length, :] = torch.tensor(
                    cat_arr,
                    dtype=torch.long,
                )

                action_num[batch_idx, hand_idx, :length, :] = torch.tensor(
                    num_arr,
                    dtype=torch.float32,
                )

                action_mask[batch_idx, hand_idx, :length] = True
                hand_mask[batch_idx, hand_idx] = True

        feature_matrix = self.vectorizer.transform(chunks)
        feature_matrix = np.asarray(feature_matrix, dtype=np.float32)

        features = torch.tensor(
            feature_matrix,
            dtype=torch.float32,
        )

        return {
            "action_cat": action_cat.to(self.device),
            "action_num": action_num.to(self.device),
            "action_mask": action_mask.to(self.device),
            "hand_mask": hand_mask.to(self.device),
            "features": features.to(self.device),
        }

    @torch.no_grad()
    def _predict_neural_batch(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> List[float]:
        logits = self.model(
            action_cat=batch["action_cat"],
            action_num=batch["action_num"],
            action_mask=batch["action_mask"],
            hand_mask=batch["hand_mask"],
            features=batch["features"],
        )

        logits = logits.view(-1)

        probs = torch.sigmoid(logits / self.temperature)
        return probs.detach().cpu().numpy().astype(float).tolist()

    @torch.no_grad()
    def _predict_xgb_batch(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> List[float]:
        if self.xgb_model is None:
            raise RuntimeError("XGBoost model is not loaded.")

        chunk_embedding = self.model.extract_chunk_embedding(
            action_cat=batch["action_cat"],
            action_num=batch["action_num"],
            action_mask=batch["action_mask"],
            hand_mask=batch["hand_mask"],
        )

        emb_np = chunk_embedding.detach().cpu().numpy().astype(np.float32)
        feat_np = batch["features"].detach().cpu().numpy().astype(np.float32)

        xgb_x = np.concatenate(
            [emb_np, feat_np],
            axis=1,
        )

        if hasattr(self.xgb_model, "predict_proba"):
            probs = self.xgb_model.predict_proba(xgb_x)[:, 1]
        else:
            probs = self.xgb_model.predict(xgb_x)

        return np.asarray(probs, dtype=float).tolist()

    @torch.no_grad()
    def predict_chunks(
        self,
        chunks: List[List[Dict[str, Any]]],
        batch_size: int = 64,
    ) -> List[float]:
        """
        Return one bot-risk score per input chunk.

        If len(chunks) == 80, this returns 80 scores.
        """

        if not chunks:
            return []

        all_scores: List[float] = []

        self.model.eval()

        for start in range(0, len(chunks), batch_size):
            end = start + batch_size
            batch_chunks = chunks[start:end]

            batch = self._make_inference_batch(batch_chunks)

            if self.xgb_model is not None:
                probs = self._predict_xgb_batch(batch)
            else:
                probs = self._predict_neural_batch(batch)

            for prob in probs:
                score = float(prob)
                score = max(0.0, min(1.0, score))
                all_scores.append(round(1.0 - score, 6))

        if len(all_scores) != len(chunks):
            raise RuntimeError(
                f"Wrong score count: chunks={len(chunks)}, scores={len(all_scores)}"
            )

        return all_scores

    @torch.no_grad()
    def predict_chunk(
        self,
        chunk: List[Dict[str, Any]],
    ) -> float:
        scores = self.predict_chunks([chunk], batch_size=1)
        return scores[0] if scores else 0.5

    def predict_labels(
        self,
        chunks: List[List[Dict[str, Any]]],
        batch_size: int = 64,
    ) -> List[bool]:
        scores = self.predict_chunks(
            chunks,
            batch_size=batch_size,
        )

        return [
            score >= self.threshold
            for score in scores
        ]