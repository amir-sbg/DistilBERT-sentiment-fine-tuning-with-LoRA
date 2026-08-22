from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import EvalPrediction, Trainer


def _logits_array(predictions) -> np.ndarray:
    logits = predictions[0] if isinstance(predictions, tuple) else predictions
    values = np.asarray(logits)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("classification logits must have shape (n_examples, n_labels)")
    return values


def _positive_probabilities(predictions) -> np.ndarray:
    logits = _logits_array(predictions)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities = probabilities / np.sum(probabilities, axis=-1, keepdims=True)
    return probabilities[:, 1]


def classification_metrics(prediction: EvalPrediction) -> dict[str, float]:
    logits = _logits_array(prediction.predictions)
    labels = np.asarray(prediction.label_ids)
    predicted_labels = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, predicted_labels)),
        "precision": float(
            precision_score(labels, predicted_labels, zero_division=0)
        ),
        "recall": float(recall_score(labels, predicted_labels, zero_division=0)),
        "f1": float(f1_score(labels, predicted_labels, zero_division=0)),
    }


def threshold_metrics(
    labels,
    positive_probabilities,
    threshold: float,
) -> dict[str, float]:
    if threshold <= 0 or threshold >= 1:
        raise ValueError("threshold must be between 0 and 1")
    labels = np.asarray(labels)
    probabilities = np.asarray(positive_probabilities)
    if labels.shape[0] != probabilities.shape[0]:
        raise ValueError("labels and probabilities must have the same length")

    predicted_labels = (probabilities >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predicted_labels)),
        "precision": float(precision_score(labels, predicted_labels, zero_division=0)),
        "recall": float(recall_score(labels, predicted_labels, zero_division=0)),
        "f1": float(f1_score(labels, predicted_labels, zero_division=0)),
        "positive_rate": float(np.mean(predicted_labels)),
    }


def threshold_sweep(
    labels,
    positive_probabilities,
    thresholds: tuple[float, ...],
) -> list[dict[str, float]]:
    return [
        threshold_metrics(labels, positive_probabilities, threshold)
        for threshold in thresholds
    ]


def prediction_records(labels, positive_probabilities) -> list[dict[str, float | int]]:
    labels = np.asarray(labels)
    probabilities = np.asarray(positive_probabilities)
    predicted_labels = (probabilities >= 0.5).astype(int)
    confidence = np.where(predicted_labels == 1, probabilities, 1.0 - probabilities)
    return [
        {
            "index": int(index),
            "label": int(label),
            "predicted_label": int(predicted),
            "positive_probability": float(probability),
            "confidence": float(score),
        }
        for index, (label, predicted, probability, score) in enumerate(
            zip(labels, predicted_labels, probabilities, confidence)
        )
    ]


def evaluate_test_set(
    trainer: Trainer,
    test_dataset: Any,
    report_dir: Path,
    thresholds: tuple[float, ...] = (0.5,),
) -> dict[str, float]:
    output = trainer.predict(
        test_dataset=test_dataset,
        metric_key_prefix="test",
    )
    metrics = output.metrics
    serializable = {
        key: float(value) for key, value in metrics.items() if np.isscalar(value)
    }
    probabilities = _positive_probabilities(output.predictions)
    sweep = threshold_sweep(output.label_ids, probabilities, thresholds)
    best_threshold = max(sweep, key=lambda row: (row["f1"], row["accuracy"]))
    serializable.update(
        {
            "test_positive_rate_at_0_5": float(np.mean(probabilities >= 0.5)),
            "test_mean_positive_probability": float(np.mean(probabilities)),
            "test_best_threshold": best_threshold["threshold"],
            "test_best_threshold_f1": best_threshold["f1"],
        }
    )

    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "test_metrics.json").write_text(
        json.dumps(serializable, indent=2) + "\n"
    )
    (report_dir / "threshold_sweep.json").write_text(
        json.dumps(sweep, indent=2) + "\n"
    )
    (report_dir / "test_predictions.json").write_text(
        json.dumps(prediction_records(output.label_ids, probabilities), indent=2) + "\n"
    )
    return serializable
