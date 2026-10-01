from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from transformers import EvalPrediction, Trainer


def _logits_array(predictions) -> np.ndarray:
    logits = predictions[0] if isinstance(predictions, tuple) else predictions
    values = np.asarray(logits)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("binary classification logits must have shape (n_examples, 2)")
    if not np.all(np.isfinite(values)):
        raise ValueError("classification logits must be finite")
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
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("labels must be one-dimensional and match the logits")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("labels must be binary")
    predicted_labels = np.argmax(logits, axis=-1)
    positive_probabilities = _positive_probabilities(logits)
    return {
        "accuracy": float(accuracy_score(labels, predicted_labels)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predicted_labels)
        ),
        "precision": float(
            precision_score(labels, predicted_labels, zero_division=0)
        ),
        "recall": float(recall_score(labels, predicted_labels, zero_division=0)),
        "f1": float(f1_score(labels, predicted_labels, zero_division=0)),
        "matthews_correlation": float(matthews_corrcoef(labels, predicted_labels)),
        "log_loss": float(
            log_loss(labels, positive_probabilities, labels=[0, 1])
        ),
    }


def threshold_metrics(
    labels,
    positive_probabilities,
    threshold: float,
) -> dict[str, float | int]:
    if threshold <= 0 or threshold >= 1:
        raise ValueError("threshold must be between 0 and 1")
    labels, probabilities = _validated_binary_arrays(labels, positive_probabilities)

    predicted_labels = (probabilities >= threshold).astype(int)
    true_negative, false_positive, false_negative, true_positive = _confusion_counts(
        labels,
        predicted_labels,
    )
    return {
        "threshold": float(threshold),
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
        "accuracy": float(accuracy_score(labels, predicted_labels)),
        "precision": float(precision_score(labels, predicted_labels, zero_division=0)),
        "recall": float(recall_score(labels, predicted_labels, zero_division=0)),
        "specificity": _safe_rate(true_negative, true_negative + false_positive),
        "false_positive_rate": _safe_rate(false_positive, false_positive + true_negative),
        "false_negative_rate": _safe_rate(false_negative, false_negative + true_positive),
        "f1": float(f1_score(labels, predicted_labels, zero_division=0)),
        "positive_rate": float(np.mean(predicted_labels)),
    }


def calibration_report(
    labels,
    positive_probabilities,
    bins: int = 10,
) -> dict[str, float | int | list[dict[str, float | int]]]:
    if bins < 2:
        raise ValueError("bins must be at least 2")
    labels, probabilities = _validated_binary_arrays(labels, positive_probabilities)

    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(probabilities, edges[1:-1]), bins - 1)
    rows = []
    expected_calibration_error = 0.0
    maximum_calibration_error = 0.0
    for index in range(bins):
        mask = assignments == index
        count = int(mask.sum())
        if count == 0:
            continue
        mean_probability = float(probabilities[mask].mean())
        observed_positive_rate = float(labels[mask].mean())
        gap = abs(observed_positive_rate - mean_probability)
        expected_calibration_error += count / len(labels) * gap
        maximum_calibration_error = max(maximum_calibration_error, gap)
        rows.append(
            {
                "bin": index,
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": count,
                "mean_positive_probability": mean_probability,
                "observed_positive_rate": observed_positive_rate,
                "absolute_gap": float(gap),
            }
        )

    return {
        "bins": bins,
        "n_examples": int(len(labels)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "expected_calibration_error": float(expected_calibration_error),
        "maximum_calibration_error": float(maximum_calibration_error),
        "table": rows,
    }


def confidence_slices(
    labels,
    positive_probabilities,
    threshold: float = 0.5,
) -> list[dict[str, float | int | str]]:
    if threshold <= 0 or threshold >= 1:
        raise ValueError("threshold must be between 0 and 1")
    labels, probabilities = _validated_binary_arrays(labels, positive_probabilities)
    predicted_labels = (probabilities >= threshold).astype(int)
    confidence = np.where(predicted_labels == 1, probabilities, 1.0 - probabilities)
    correct = predicted_labels == labels
    bands = (
        ("low", 0.50, 0.70),
        ("medium", 0.70, 0.85),
        ("high", 0.85, 1.000001),
    )
    rows = []
    for name, lower, upper in bands:
        mask = (confidence >= lower) & (confidence < upper)
        count = int(mask.sum())
        if count == 0:
            rows.append(
                {
                    "slice": name,
                    "count": 0,
                    "accuracy": 0.0,
                    "error_rate": 0.0,
                    "mean_confidence": 0.0,
                }
            )
            continue
        rows.append(
            {
                "slice": name,
                "count": count,
                "accuracy": float(correct[mask].mean()),
                "error_rate": float(1.0 - correct[mask].mean()),
                "mean_confidence": float(confidence[mask].mean()),
            }
        )
    return rows


def selective_risk_report(
    labels,
    positive_probabilities,
    coverages: tuple[float, ...] = (1.0, 0.9, 0.75, 0.5),
) -> dict[str, float | int | list[dict[str, float | int]]]:
    labels, probabilities = _validated_binary_arrays(labels, positive_probabilities)
    if not coverages or any(not 0.0 < coverage <= 1.0 for coverage in coverages):
        raise ValueError("coverages must be in (0, 1]")

    predicted_labels = (probabilities >= 0.5).astype(int)
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    order = np.argsort(-confidence, kind="stable")
    ordered_errors = (predicted_labels[order] != labels[order]).astype(float)
    prefix_risk = np.cumsum(ordered_errors) / np.arange(1, len(labels) + 1)

    points = []
    for target_coverage in coverages:
        retained = min(
            len(labels),
            max(1, int(np.ceil(target_coverage * len(labels)))),
        )
        selected = order[:retained]
        errors = int((predicted_labels[selected] != labels[selected]).sum())
        risk = errors / retained
        points.append(
            {
                "target_coverage": float(target_coverage),
                "coverage": float(retained / len(labels)),
                "retained": retained,
                "errors": errors,
                "selective_accuracy": float(1.0 - risk),
                "selective_risk": float(risk),
                "minimum_confidence": float(confidence[selected].min()),
            }
        )

    return {
        "n_examples": int(len(labels)),
        "area_under_risk_coverage": float(prefix_risk.mean()),
        "points": points,
    }


def threshold_sweep(
    labels,
    positive_probabilities,
    thresholds: tuple[float, ...],
) -> list[dict[str, float | int]]:
    return [
        threshold_metrics(labels, positive_probabilities, threshold)
        for threshold in thresholds
    ]


def select_threshold_with_recall_floor(
    labels,
    positive_probabilities,
    thresholds: tuple[float, ...],
    minimum_recall: float,
) -> dict[str, float | int | str]:
    if not 0 < minimum_recall <= 1:
        raise ValueError("minimum_recall must be in (0, 1]")
    rows = threshold_sweep(labels, positive_probabilities, thresholds)
    candidates = [row for row in rows if row["recall"] >= minimum_recall]
    if candidates:
        selected = max(candidates, key=lambda row: (row["f1"], row["precision"]))
        reason = "selected best F1 among thresholds that satisfy recall floor"
    else:
        selected = max(rows, key=lambda row: (row["recall"], row["f1"]))
        reason = "no threshold satisfied recall floor; selected highest recall"
    return {
        **selected,
        "minimum_recall": float(minimum_recall),
        "selection_reason": reason,
    }


def prediction_records(labels, positive_probabilities) -> list[dict[str, float | int]]:
    labels, probabilities = _validated_binary_arrays(labels, positive_probabilities)
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


def confident_error_records(
    labels,
    positive_probabilities,
    threshold: float = 0.5,
    top_k: int = 20,
) -> list[dict[str, float | int | str]]:
    if threshold <= 0 or threshold >= 1:
        raise ValueError("threshold must be between 0 and 1")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    labels, probabilities = _validated_binary_arrays(labels, positive_probabilities)

    predicted_labels = (probabilities >= threshold).astype(int)
    confidence = np.where(predicted_labels == 1, probabilities, 1.0 - probabilities)
    error_indices = np.flatnonzero(predicted_labels != labels)
    ranked_indices = sorted(
        error_indices,
        key=lambda index: confidence[index],
        reverse=True,
    )[:top_k]
    return [
        {
            "index": int(index),
            "label": int(labels[index]),
            "predicted_label": int(predicted_labels[index]),
            "positive_probability": float(probabilities[index]),
            "confidence": float(confidence[index]),
            "error_type": (
                "false_positive"
                if predicted_labels[index] == 1
                else "false_negative"
            ),
        }
        for index in ranked_indices
    ]


def _validated_binary_arrays(labels, positive_probabilities) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(positive_probabilities, dtype=float)
    if labels.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("labels and probabilities must be one-dimensional")
    if labels.shape[0] != probabilities.shape[0]:
        raise ValueError("labels and probabilities must have the same length")
    if labels.size == 0:
        raise ValueError("labels and probabilities must not be empty")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("labels must be binary")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities must be finite")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("probabilities must be between 0 and 1")
    return labels, probabilities


def _confusion_counts(
    labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> tuple[int, int, int, int]:
    true_negative = int(((labels == 0) & (predicted_labels == 0)).sum())
    false_positive = int(((labels == 0) & (predicted_labels == 1)).sum())
    false_negative = int(((labels == 1) & (predicted_labels == 0)).sum())
    true_positive = int(((labels == 1) & (predicted_labels == 1)).sum())
    return true_negative, false_positive, false_negative, true_positive


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def evaluate_test_set(
    trainer: Trainer,
    test_dataset: Any,
    report_dir: Path,
    thresholds: tuple[float, ...] = (0.5,),
    minimum_recall: float | None = None,
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
    calibration = calibration_report(output.label_ids, probabilities)
    selective_risk = selective_risk_report(output.label_ids, probabilities)
    recall_floor_report = None
    if minimum_recall is not None:
        recall_floor_report = select_threshold_with_recall_floor(
            output.label_ids,
            probabilities,
            thresholds,
            minimum_recall,
        )
    confidence = confidence_slices(
        output.label_ids,
        probabilities,
        threshold=best_threshold["threshold"],
    )
    confident_errors = confident_error_records(
        output.label_ids,
        probabilities,
        threshold=best_threshold["threshold"],
    )
    high_confidence = next(row for row in confidence if row["slice"] == "high")
    serializable.update(
        {
            "test_positive_rate_at_0_5": float(np.mean(probabilities >= 0.5)),
            "test_mean_positive_probability": float(np.mean(probabilities)),
            "test_best_threshold": best_threshold["threshold"],
            "test_best_threshold_f1": best_threshold["f1"],
            "test_brier_score": calibration["brier_score"],
            "test_expected_calibration_error": calibration[
                "expected_calibration_error"
            ],
            "test_high_confidence_error_rate": high_confidence["error_rate"],
            "test_confident_error_count": len(confident_errors),
            "test_area_under_risk_coverage": selective_risk[
                "area_under_risk_coverage"
            ],
            "test_risk_at_half_coverage": selective_risk["points"][-1][
                "selective_risk"
            ],
        }
    )
    if recall_floor_report is not None:
        serializable["test_recall_floor_threshold"] = recall_floor_report["threshold"]
        serializable["test_recall_floor_f1"] = recall_floor_report["f1"]
        serializable["test_recall_floor_recall"] = recall_floor_report["recall"]

    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "test_metrics.json").write_text(
        json.dumps(serializable, indent=2) + "\n"
    )
    (report_dir / "threshold_sweep.json").write_text(
        json.dumps(sweep, indent=2) + "\n"
    )
    if recall_floor_report is not None:
        (report_dir / "threshold_recall_floor.json").write_text(
            json.dumps(recall_floor_report, indent=2) + "\n"
        )
    (report_dir / "calibration_report.json").write_text(
        json.dumps(calibration, indent=2) + "\n"
    )
    (report_dir / "confidence_slices.json").write_text(
        json.dumps(confidence, indent=2) + "\n"
    )
    (report_dir / "selective_risk.json").write_text(
        json.dumps(selective_risk, indent=2) + "\n"
    )
    (report_dir / "confident_errors.json").write_text(
        json.dumps(confident_errors, indent=2) + "\n"
    )
    (report_dir / "test_predictions.json").write_text(
        json.dumps(prediction_records(output.label_ids, probabilities), indent=2) + "\n"
    )
    return serializable
