"""
ensemble.py - Confidence-Based Adaptive Ensemble Model Selection
=================================================================
Loads multiple emotion recognition models and combines their
predictions using either weighted-average or max-confidence
strategies.  Designed for real-time use with frame-skipping to
maintain acceptable FPS on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torchvision import transforms
from PIL import Image

from data.dataset import build_transforms
from models import load_model_from_checkpoint
from utils.grad_cam import GradCAM, get_target_layer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
BEST_DIR = PROJECT_ROOT / "best_models_output"


@dataclass
class _LoadedModel:
    """Internal container for a loaded model and its preprocessing."""

    model: nn.Module
    transform: transforms.Compose
    display_name: str   # e.g. "ResNet-18"
    arch_name: str      # e.g. "resnet18"
    grad_cam: Optional[GradCAM] = None


class EnsembleManager:
    """Manages multiple models for ensemble prediction."""

    def __init__(
        self,
        device: torch.device,
        class_names: Tuple[str, ...],
    ) -> None:
        self._device = device
        self._class_names = class_names
        self._models: Dict[str, _LoadedModel] = {}
        self._active_names: List[str] = []

    # ── Model loading ────────────────────────────────────────────

    def load_models(
        self, model_configs: Dict[str, Dict],
    ) -> List[str]:
        """Load all available model checkpoints.

        Args:
            model_configs: MODEL_OPTIONS dict from gui_app.py.
                Each value must have keys: name, checkpoint, input_size,
                grayscale.

        Returns:
            List of display names that were loaded successfully.
        """
        loaded: List[str] = []
        for display_name, cfg in model_configs.items():
            checkpoint = Path(cfg["checkpoint"])
            if not checkpoint.exists():
                alt = BEST_DIR / f"best_{cfg['name']}.pth"
                if alt.exists():
                    checkpoint = alt
                else:
                    print(
                        f"[ENSEMBLE] Checkpoint not found for "
                        f"{display_name}, skipping."
                    )
                    continue

            try:
                model, _ = load_model_from_checkpoint(
                    cfg["name"], checkpoint, self._device,
                    class_names=self._class_names,
                )
                tf = build_transforms(
                    cfg["input_size"], grayscale=cfg["grayscale"],
                )["eval"]

                # Init Grad-CAM
                grad_cam = None
                try:
                    target_layer = get_target_layer(model, cfg["name"])
                    grad_cam = GradCAM(model, target_layer)
                except Exception:
                    pass

                self._models[display_name] = _LoadedModel(
                    model=model,
                    transform=tf,
                    display_name=display_name,
                    arch_name=cfg["name"],
                    grad_cam=grad_cam,
                )
                loaded.append(display_name)
                print(f"[ENSEMBLE] {display_name} loaded.")
            except Exception as e:
                print(f"[ENSEMBLE] Failed to load {display_name}: {e}")

        # Activate all loaded models by default
        self._active_names = list(loaded)
        return loaded

    def get_loaded_models(self) -> List[str]:
        return list(self._models.keys())

    def set_active_models(self, names: List[str]) -> None:
        self._active_names = [
            n for n in names if n in self._models
        ]

    def get_active_models(self) -> List[str]:
        return list(self._active_names)

    # ── Prediction ───────────────────────────────────────────────

    def predict(
        self,
        face_pil: Image.Image,
        strategy: str = "weighted_avg",
    ) -> Tuple[Dict[str, float], str, Optional[_LoadedModel]]:
        """Run ensemble inference on a cropped face image.

        Args:
            face_pil: RGB PIL Image of the detected face.
            strategy: "weighted_avg" or "max_confidence".

        Returns:
            (merged_probs, winner_display_name, winner_loaded_model)
        """
        if not self._active_names:
            empty = {e: 0.0 for e in self._class_names}
            return empty, "", None

        all_probs: List[Tuple[str, Dict[str, float], float]] = []

        for name in self._active_names:
            lm = self._models[name]
            tensor = lm.transform(face_pil).unsqueeze(0).to(self._device)

            with torch.no_grad():
                output = lm.model(tensor)
                prob_tensor = torch.softmax(output, dim=1)[0]

            probs = {
                cls: float(prob_tensor[i])
                for i, cls in enumerate(self._class_names)
            }
            confidence = float(prob_tensor.max())
            all_probs.append((name, probs, confidence))

        if strategy == "max_confidence":
            return self._max_confidence(all_probs)
        else:
            return self._weighted_average(all_probs)

    def _weighted_average(
        self,
        all_probs: List[Tuple[str, Dict[str, float], float]],
    ) -> Tuple[Dict[str, float], str, Optional[_LoadedModel]]:
        """Confidence-weighted average of all model outputs."""
        total_weight = sum(conf for _, _, conf in all_probs)
        if total_weight <= 0:
            return (
                {e: 0.0 for e in self._class_names},
                all_probs[0][0],
                self._models.get(all_probs[0][0]),
            )

        merged: Dict[str, float] = {e: 0.0 for e in self._class_names}
        for _, probs, conf in all_probs:
            for emotion in self._class_names:
                merged[emotion] += conf * probs[emotion]

        for emotion in self._class_names:
            merged[emotion] = round(merged[emotion] / total_weight, 4)

        # Winner = the model with highest individual confidence
        best_name = max(all_probs, key=lambda x: x[2])[0]
        return merged, best_name, self._models.get(best_name)

    def _max_confidence(
        self,
        all_probs: List[Tuple[str, Dict[str, float], float]],
    ) -> Tuple[Dict[str, float], str, Optional[_LoadedModel]]:
        """Select the prediction from the most confident model."""
        best_name, best_probs, _ = max(all_probs, key=lambda x: x[2])
        return best_probs, best_name, self._models.get(best_name)

    # ── Grad-CAM access ─────────────────────────────────────────

    def get_grad_cam_for_model(
        self, display_name: str,
    ) -> Optional[GradCAM]:
        lm = self._models.get(display_name)
        if lm is not None:
            return lm.grad_cam
        return None

    def get_model_and_transform(
        self, display_name: str,
    ) -> Optional[Tuple[nn.Module, transforms.Compose]]:
        lm = self._models.get(display_name)
        if lm is not None:
            return lm.model, lm.transform
        return None

    # ── Cleanup ──────────────────────────────────────────────────

    def release(self) -> None:
        for lm in self._models.values():
            if lm.grad_cam is not None:
                lm.grad_cam.release()
        self._models.clear()
        self._active_names.clear()
