from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import EvalPrediction, Trainer


def classification_metrics(prediction: EvalPrediction) -> dict[str, float]:
    logits = prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    labels = np.asarray(prediction.label_ids)
    predicted_labels = np.argmax(np.asarray(logits), axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, predicted_labels)),
        "precision": float(
            precision_score(labels, predicted_labels, zero_division=0)
        ),
        "recall": float(recall_score(labels, predicted_labels, zero_division=0)),
        "f1": float(f1_score(labels, predicted_labels, zero_division=0)),
    }


def evaluate_test_set(
    trainer: Trainer,
    test_dataset: Any,
    report_dir: Path,
) -> dict[str, float]:
    metrics = trainer.evaluate(
        eval_dataset=test_dataset,
        metric_key_prefix="test",
    )
    serializable = {
        key: float(value) for key, value in metrics.items() if np.isscalar(value)
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "test_metrics.json").write_text(
        json.dumps(serializable, indent=2) + "\n"
    )
    return serializable
