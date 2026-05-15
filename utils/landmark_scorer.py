"""
landmark_scorer.py
==================
Uses MediaPipe Face Landmarker blendshapes to compute geometry-based
signals for emotions CNNs commonly confuse:

  angry   → brow-down score     (browDownLeft/Right high, browInnerUp low)
  sad     → grief-brow score    (browInnerUp × mouthShrugLower product)
  surprise→ brow-raise+jaw-open (browOuterUp × jawOpen)

Scores are in [0, 1].  Applied after model inference to boost the
corresponding class probabilities, then the distribution is renormalised.

Validated on reference images (6-participant test set):
  angry_score   : Angry=0.62,  others≤0.08
  sad_score     : Sad=0.99,    Neutral=0.10, others≤0.01
  surprise_score: Surprise=0.32, others≤0.12
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

_MODEL_PATH = Path(__file__).resolve().parent.parent / "face_landmarker.task"

# Boost weights — how strongly landmark scores scale probabilities.
ANGRY_FURROW_WEIGHT   = 1.6
SAD_DROOP_WEIGHT      = 1.4
SURPRISE_RAISE_WEIGHT = 1.2

# Neutral confidence guard thresholds.
# When model confidence for neutral exceeds these, the corresponding boost
# is dampened — avoids over-correcting faces with naturally expressive brows.
_ANGRY_NEUTRAL_GUARD_THRESHOLD    = 0.55   # high — furrow is a reliable angry signal
_SURPRISE_NEUTRAL_GUARD_THRESHOLD = 0.40   # lower — brow-raise fires on many neutral faces
_NEUTRAL_GUARD_FACTOR             = 0.25


class LandmarkScorer:
    """Lazy-init wrapper around MediaPipe FaceLandmarker (Tasks API)."""

    def __init__(self) -> None:
        self._landmarker = None

    # ── Initialisation ───────────────────────────────────────────

    def _ensure_landmarker(self) -> bool:
        if self._landmarker is not None:
            return True
        if not _MODEL_PATH.exists():
            print(f"[LandmarkScorer] Model not found: {_MODEL_PATH}")
            return False
        try:
            from mediapipe.tasks.python import vision, BaseOptions
            options = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(_MODEL_PATH)),
                num_faces=1,
                min_face_detection_confidence=0.40,
                min_face_presence_confidence=0.40,
                min_tracking_confidence=0.40,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=False,
            )
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
            return True
        except Exception as exc:
            print(f"[LandmarkScorer] Init error: {exc}")
            return False

    # ── Public API ───────────────────────────────────────────────

    def score(self, face_rgb: np.ndarray) -> Dict[str, float]:
        """Return {'brow_furrow': float, 'mouth_droop': float, 'brow_raise': float} in [0, 1].

        Falls back to 0.0 on detection failure so no boost is applied.
        """
        if not self._ensure_landmarker():
            return {"brow_furrow": 0.0, "mouth_droop": 0.0, "brow_raise": 0.0}

        try:
            import mediapipe as mp
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=face_rgb.astype(np.uint8),
            )
            result = self._landmarker.detect(mp_image)
        except Exception as exc:
            print(f"[LandmarkScorer] Detection error: {exc}")
            return {"brow_furrow": 0.0, "mouth_droop": 0.0, "brow_raise": 0.0}

        if not result.face_blendshapes:
            return {"brow_furrow": 0.0, "mouth_droop": 0.0, "brow_raise": 0.0}

        s = {b.category_name: b.score for b in result.face_blendshapes[0]}
        return {
            "brow_furrow": self._angry_score(s),
            "mouth_droop": self._sad_score(s),
            "brow_raise":  self._surprise_score(s),
        }

    def apply(
        self,
        probs: Dict[str, float],
        scores: Dict[str, float],
        angry_weight:   float = ANGRY_FURROW_WEIGHT,
        sad_weight:     float = SAD_DROOP_WEIGHT,
        surprise_weight: float = SURPRISE_RAISE_WEIGHT,
    ) -> Dict[str, float]:
        """Boost angry/sad/surprise then renormalise the full distribution.

        Angry boost includes a neutral-confidence guard: when the model is
        already confident the face is neutral (>55%), the angry boost is
        dampened to avoid over-correcting people with naturally low brows.
        Sad boost has no such guard — droop is a reliable sad signal.
        """
        out = dict(probs)

        furrow   = scores["brow_furrow"]
        droop    = scores["mouth_droop"]
        raise_   = scores.get("brow_raise", 0.0)

        # ── Angry ────────────────────────────────────────────────
        if "angry" in out:
            neutral_conf = out.get("neutral", 0.0)
            effective_angry_weight = angry_weight
            effective_furrow = furrow
            if neutral_conf > _ANGRY_NEUTRAL_GUARD_THRESHOLD:
                effective_angry_weight *= _NEUTRAL_GUARD_FACTOR
                effective_furrow *= _NEUTRAL_GUARD_FACTOR

            out["angry"] *= 1.0 + effective_furrow * effective_angry_weight
            if furrow > 0.65 and neutral_conf <= _ANGRY_NEUTRAL_GUARD_THRESHOLD:
                floor = furrow * 0.28
                out["angry"] = max(out["angry"], floor)

        # ── Sad ──────────────────────────────────────────────────
        if "sad" in out:
            out["sad"] *= 1.0 + droop * sad_weight
            if droop > 0.65:
                floor = droop * 0.40
                out["sad"] = max(out["sad"], floor)

        # ── Surprise ─────────────────────────────────────────────
        # Neutral guard: people who naturally hold brows up at rest can
        # trigger high raise scores on non-surprise faces.
        if "surprise" in out:
            neutral_conf_s = out.get("neutral", 0.0)
            eff_surprise_weight = surprise_weight
            if neutral_conf_s > _SURPRISE_NEUTRAL_GUARD_THRESHOLD:
                eff_surprise_weight *= _NEUTRAL_GUARD_FACTOR
            out["surprise"] *= 1.0 + raise_ * eff_surprise_weight

        total = sum(out.values())
        if total > 0:
            out = {k: v / total for k, v in out.items()}

        return out

    # ── Blendshape formulas ──────────────────────────────────────

    @staticmethod
    def _angry_score(s: Dict[str, float]) -> float:
        """browDown high + browInnerUp low  →  angry."""
        brow_down  = (s.get("browDownLeft", 0.0) + s.get("browDownRight", 0.0)) / 2.0
        brow_inner = s.get("browInnerUp", 0.0)
        raw = brow_down * 8.0 + max(0.0, 0.25 - brow_inner) * 2.0
        return float(np.clip(raw, 0.0, 1.0))

    @staticmethod
    def _sad_score(s: Dict[str, float]) -> float:
        """browInnerUp × mouthShrugLower (grief brow + lip tension)  →  sad."""
        brow_inner  = s.get("browInnerUp", 0.0)
        mouth_shrug = s.get("mouthShrugLower", 0.0)
        raw = brow_inner * mouth_shrug
        return float(np.clip((raw - 0.015) / 0.065, 0.0, 1.0))

    @staticmethod
    def _surprise_score(s: Dict[str, float]) -> float:
        """browOuterUp × jawOpen (raised brows + open mouth)  →  surprise."""
        brow_outer = (s.get("browOuterUpLeft", 0.0) + s.get("browOuterUpRight", 0.0)) / 2.0
        brow_inner = s.get("browInnerUp", 0.0)
        jaw_open   = s.get("jawOpen", 0.0)
        brow_avg   = brow_outer * 0.5 + brow_inner * 0.5
        raw = brow_avg * jaw_open
        return float(np.clip((raw - 0.02) / 0.10, 0.0, 1.0))
