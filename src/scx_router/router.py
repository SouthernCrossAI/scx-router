from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

import torch
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer

MODEL_ID = "scx-admin/scx-router-v0.1"

DEFAULT_MODEL_LABELS = (
    "coder",
    "DeepSeek-V3.1",
    "gemma-4-31B-it",
    "gpt-oss-120b",
    "Llama-4-Maverick-17B-128E-Instruct",
    "MAGPiE",
    "Meta-Llama-3.3-70B-Instruct",
    "Qwen3-32B",
)


class RouterScore(TypedDict):
    label: str
    score: float


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SCXRouter:
    """Load SCX Router once and rank model labels for each request."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str | torch.device | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = torch.device(device) if device is not None else default_device()
        self.model = GLiClassModel.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.pipeline = ZeroShotClassificationPipeline(
            self.model,
            self.tokenizer,
            classification_type="multi-label",
            device=self.device,
            progress_bar=False,
        )

    def rank(
        self,
        text: str,
        labels: Sequence[str] = DEFAULT_MODEL_LABELS,
    ) -> list[RouterScore]:
        """Return every candidate sorted from highest to lowest score."""
        if not text.strip():
            raise ValueError("text must not be empty")
        if not labels:
            raise ValueError("labels must not be empty")

        predictions = self.pipeline(text, list(labels), threshold=0.0)[0]
        scores = [
            RouterScore(label=str(item["label"]), score=float(item["score"]))
            for item in predictions
        ]
        return sorted(scores, key=lambda item: item["score"], reverse=True)
