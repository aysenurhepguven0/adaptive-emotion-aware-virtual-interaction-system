"""
calibration.py - User-Specific Emotion Calibration
====================================================
Records per-user baseline probability distributions for each emotion,
builds a confusion matrix, and applies pseudo-inverse correction to
reduce individual / cultural bias in real-time predictions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROFILES_DIR = PROJECT_ROOT / "calibration_profiles"

CONDITION_NUMBER_THRESHOLD = 50.0


@dataclass
class CalibrationProfile:
    """Holds a single user's calibration data for one model."""

    user_name: str
    model_name: str
    created_at: str
    emotion_classes: Tuple[str, ...]
    frames_per_emotion: int
    baseline_distributions: Dict[str, Dict[str, float]]
    confusion_matrix: List[List[float]]
    correction_matrix: List[List[float]]
    correction_method: str  # "pinv" or "diagonal"


class CalibrationManager:
    """Manages calibration recording, correction, and profile persistence."""

    def __init__(
        self,
        emotion_classes: Tuple[str, ...],
        profiles_dir: Path = DEFAULT_PROFILES_DIR,
    ) -> None:
        self._emotion_classes = emotion_classes
        self._profiles_dir = profiles_dir
        self._n = len(emotion_classes)

        # Active profile for correction
        self._active_profile: Optional[CalibrationProfile] = None
        self._correction_matrix: Optional[np.ndarray] = None

        # Session state (recording)
        self._recording = False
        self._session_user: str = ""
        self._session_model: str = ""
        self._session_frames_per_emotion: int = 30
        self._session_emotion_index: int = 0
        self._session_frame_buffer: List[Dict[str, float]] = []
        self._session_baselines: Dict[str, Dict[str, float]] = {}

    # ── Recording ────────────────────────────────────────────────

    def start_session(
        self,
        user_name: str,
        model_name: str,
        frames_per_emotion: int = 30,
    ) -> None:
        self._recording = True
        self._session_user = user_name
        self._session_model = model_name
        self._session_frames_per_emotion = frames_per_emotion
        self._session_emotion_index = 0
        self._session_frame_buffer = []
        self._session_baselines = {}

    def get_current_emotion(self) -> Optional[str]:
        if not self._recording:
            return None
        if self._session_emotion_index >= self._n:
            return None
        return self._emotion_classes[self._session_emotion_index]

    def record_frame(self, probs: Dict[str, float]) -> Tuple[str, int, int]:
        """Record one frame's raw probabilities for the current emotion.

        Returns (current_emotion, frames_recorded, frames_needed).
        """
        emotion = self.get_current_emotion()
        if emotion is None:
            return ("", 0, 0)

        self._session_frame_buffer.append(dict(probs))
        recorded = len(self._session_frame_buffer)
        needed = self._session_frames_per_emotion

        if recorded >= needed:
            self._finalize_current_emotion()

        return (emotion, min(recorded, needed), needed)

    def _finalize_current_emotion(self) -> None:
        """Average the frame buffer into a baseline for the current emotion."""
        emotion = self._emotion_classes[self._session_emotion_index]
        avg: Dict[str, float] = {}
        n_frames = len(self._session_frame_buffer)
        for cls in self._emotion_classes:
            total = sum(f.get(cls, 0.0) for f in self._session_frame_buffer)
            avg[cls] = total / max(n_frames, 1)
        self._session_baselines[emotion] = avg
        self._session_frame_buffer = []
        self._session_emotion_index += 1

    def is_recording(self) -> bool:
        return self._recording

    def is_session_complete(self) -> bool:
        return (
            self._recording
            and self._session_emotion_index >= self._n
        )

    def finish_session(self) -> CalibrationProfile:
        """Build confusion matrix and correction matrix from baselines."""
        if not self.is_session_complete():
            raise RuntimeError("Calibration session is not complete.")

        self._recording = False

        # Build confusion matrix C[i][j] = avg P(pred=j | true=i)
        C = np.zeros((self._n, self._n), dtype=np.float64)
        for i, true_emotion in enumerate(self._emotion_classes):
            baseline = self._session_baselines[true_emotion]
            for j, pred_emotion in enumerate(self._emotion_classes):
                C[i, j] = baseline.get(pred_emotion, 0.0)

        # Compute correction matrix
        cond = np.linalg.cond(C)
        if cond < CONDITION_NUMBER_THRESHOLD:
            correction = np.linalg.pinv(C)
            method = "pinv"
        else:
            # Fallback: diagonal scaling
            diag = np.diag(C).copy()
            diag[diag < 1e-6] = 1e-6  # avoid division by zero
            correction = np.diag(1.0 / diag)
            method = "diagonal"

        profile = CalibrationProfile(
            user_name=self._session_user,
            model_name=self._session_model,
            created_at=datetime.now().isoformat(timespec="seconds"),
            emotion_classes=self._emotion_classes,
            frames_per_emotion=self._session_frames_per_emotion,
            baseline_distributions=dict(self._session_baselines),
            confusion_matrix=C.tolist(),
            correction_matrix=correction.tolist(),
            correction_method=method,
        )
        return profile

    def cancel_session(self) -> None:
        self._recording = False
        self._session_frame_buffer = []
        self._session_baselines = {}
        self._session_emotion_index = 0

    # ── Correction ───────────────────────────────────────────────

    def apply_correction(self, probs: Dict[str, float]) -> Dict[str, float]:
        """Apply the active profile's correction to raw probabilities."""
        if self._correction_matrix is None:
            return probs

        raw = np.array(
            [probs.get(e, 0.0) for e in self._emotion_classes],
            dtype=np.float64,
        )
        corrected = self._correction_matrix @ raw
        corrected = np.clip(corrected, 0.0, None)
        total = corrected.sum()
        if total > 0:
            corrected /= total
        else:
            return probs  # fallback to raw if correction collapses

        return {
            e: round(float(corrected[i]), 4)
            for i, e in enumerate(self._emotion_classes)
        }

    def set_active_profile(
        self, profile: Optional[CalibrationProfile]
    ) -> None:
        self._active_profile = profile
        if profile is not None:
            self._correction_matrix = np.array(
                profile.correction_matrix, dtype=np.float64,
            )
        else:
            self._correction_matrix = None

    def has_active_profile(self) -> bool:
        return self._active_profile is not None

    def get_active_profile(self) -> Optional[CalibrationProfile]:
        return self._active_profile

    # ── Persistence ──────────────────────────────────────────────

    def save_profile(self, profile: CalibrationProfile) -> Path:
        self._profiles_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_user = "".join(
            c if c.isalnum() or c in "_-" else "_"
            for c in profile.user_name
        )
        safe_model = profile.model_name.replace(" ", "_").replace("-", "_")
        filename = f"{safe_user}_{safe_model}_{ts}.json"
        path = self._profiles_dir / filename

        data = asdict(profile)
        # Convert tuple to list for JSON serialization
        data["emotion_classes"] = list(profile.emotion_classes)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return path

    def load_profile(self, path: Path) -> CalibrationProfile:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["emotion_classes"] = tuple(data["emotion_classes"])
        return CalibrationProfile(**data)

    def list_profiles(
        self, model_name: Optional[str] = None,
    ) -> List[Path]:
        if not self._profiles_dir.exists():
            return []
        profiles = sorted(self._profiles_dir.glob("*.json"), reverse=True)
        if model_name is not None:
            safe_model = model_name.replace(" ", "_").replace("-", "_")
            profiles = [
                p for p in profiles if safe_model.lower() in p.stem.lower()
            ]
        return profiles

    def delete_profile(self, path: Path) -> None:
        if path.exists():
            path.unlink()

    def get_diagonal_scores(
        self, profile: CalibrationProfile,
    ) -> Dict[str, float]:
        """Return the diagonal values of the confusion matrix.

        High values (close to 1.0) mean the model recognized that
        emotion well during calibration.  Values below 0.2 are a
        warning that calibration quality is poor for that emotion.
        """
        scores: Dict[str, float] = {}
        for i, emotion in enumerate(profile.emotion_classes):
            scores[emotion] = profile.confusion_matrix[i][i]
        return scores
