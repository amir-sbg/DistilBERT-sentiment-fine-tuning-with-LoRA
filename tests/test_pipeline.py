from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from datasets import Dataset, DatasetDict
from transformers import EvalPrediction

from fine_tuning_llms.config import FineTuneConfig
from fine_tuning_llms.data import (
    _limit_split,
    label_distribution,
    tokenize_dataset,
    validate_dataset,
)
from fine_tuning_llms.evaluate import (
    brier_decomposition,
    bootstrap_metric_intervals,
    calibration_report,
    classification_metrics,
    confident_error_records,
    confidence_slices,
    evaluate_test_set,
    selective_risk_report,
    select_threshold_by_cost,
    select_threshold_with_recall_floor,
    threshold_sweep,
    uncertainty_summary,
)
from fine_tuning_llms.inference import predict
from fine_tuning_llms.model import parameter_report
from fine_tuning_llms.train import optimization_plan


class TinyTokenizer:
    def __call__(self, texts, max_length, truncation):
        input_ids = [list(range(min(len(text.split()), max_length))) for text in texts]
        return {
            "input_ids": [tokens or [0] for tokens in input_ids],
            "attention_mask": [[1] * max(len(tokens), 1) for tokens in input_ids],
        }


class TinyBatch(dict):
    def to(self, device):
        return self


class InferenceTokenizer:
    def __call__(self, texts, **kwargs):
        return TinyBatch(
            input_ids=torch.ones((len(texts), 2), dtype=torch.long),
            attention_mask=torch.ones((len(texts), 2), dtype=torch.long),
        )


class InferenceModel:
    def __call__(self, **kwargs):
        return SimpleNamespace(logits=torch.tensor([[2.0, 1.0], [1.0, 3.0]]))


class BatchAwareInferenceModel:
    def __call__(self, **kwargs):
        rows = kwargs["input_ids"].shape[0]
        negative_logit = torch.zeros(rows)
        positive_logit = torch.linspace(-1.0, 1.0, rows)
        return SimpleNamespace(
            logits=torch.stack([negative_logit, positive_logit], dim=1)
        )


class TinyTrainer:
    def predict(self, test_dataset, metric_key_prefix):
        return SimpleNamespace(
            metrics={f"{metric_key_prefix}_loss": 0.2},
            predictions=np.array(
                [
                    [2.0, 0.2],
                    [0.1, 2.5],
                    [0.6, 1.0],
                ]
            ),
            label_ids=np.array([0, 1, 0]),
        )


class TinyAdapterModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = torch.nn.Linear(2, 2)
        self.lora_A = torch.nn.Linear(2, 1, bias=False)
        self.classifier = torch.nn.Linear(1, 2)
        for parameter in self.base.parameters():
            parameter.requires_grad = False


def tiny_dataset() -> DatasetDict:
    split = Dataset.from_dict(
        {
            "text": ["short review", "another review"],
            "label": [0, 1],
        }
    )
    return DatasetDict({"train": split, "validation": split, "test": split})


def test_dataset_validation_requires_all_splits() -> None:
    with pytest.raises(ValueError, match="missing splits"):
        validate_dataset({"train": tiny_dataset()["train"]})


def test_tokenization_removes_raw_text() -> None:
    tokenized = tokenize_dataset(tiny_dataset(), TinyTokenizer(), max_length=4)
    assert "text" not in tokenized["train"].column_names
    assert "input_ids" in tokenized["train"].column_names
    assert tokenized["train"][0]["input_ids"] == [0, 1]


def test_label_distribution_reports_each_split() -> None:
    report = label_distribution(tiny_dataset())

    assert report["train"]["rows"] == 2
    assert report["validation"]["positive"] == 1
    assert report["test"]["positive_rate"] == 0.5


def test_sample_limit_preserves_label_balance() -> None:
    dataset = Dataset.from_dict(
        {
            "text": [f"review {index}" for index in range(10)],
            "label": [0] * 8 + [1] * 2,
        }
    )

    limited = _limit_split(dataset, limit=5, seed=7)

    assert len(limited) == 5
    assert limited["label"].count(0) == 4
    assert limited["label"].count(1) == 1
    assert _limit_split(dataset, limit=5, seed=7)["text"] == limited["text"]


def test_classification_metrics_are_consistent() -> None:
    prediction = EvalPrediction(
        predictions=np.array([[3.0, 1.0], [0.5, 2.0], [2.0, 1.0], [0.0, 2.5]]),
        label_ids=np.array([0, 1, 0, 1]),
    )
    metrics = classification_metrics(prediction)
    assert metrics == {
        "accuracy": 1.0,
        "balanced_accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "matthews_correlation": 1.0,
        "log_loss": pytest.approx(0.180123, abs=1e-6),
    }


def test_classification_metrics_reject_non_binary_logits() -> None:
    with pytest.raises(ValueError, match="shape"):
        classification_metrics(
            EvalPrediction(
                predictions=np.zeros((2, 3)),
                label_ids=np.array([0, 1]),
            )
        )
    with pytest.raises(ValueError, match="finite"):
        classification_metrics(
            EvalPrediction(
                predictions=np.array([[np.nan, 0.0], [0.0, 1.0]]),
                label_ids=np.array([0, 1]),
            )
        )


def test_threshold_sweep_reports_decision_tradeoffs() -> None:
    rows = threshold_sweep(
        labels=np.array([0, 1, 1, 0]),
        positive_probabilities=np.array([0.10, 0.55, 0.80, 0.60]),
        thresholds=(0.50, 0.70),
    )

    assert rows[0]["threshold"] == 0.5
    assert rows[0]["positive_rate"] == 0.75
    assert rows[0]["true_negative"] == 1
    assert rows[0]["false_positive"] == 1
    assert rows[0]["false_negative"] == 0
    assert rows[0]["specificity"] == 0.5
    assert rows[0]["false_positive_rate"] == 0.5
    assert rows[0]["false_negative_rate"] == 0.0
    assert rows[1]["recall"] == 0.5


def test_select_threshold_with_recall_floor_prefers_feasible_f1() -> None:
    result = select_threshold_with_recall_floor(
        labels=np.array([0, 1, 1, 0]),
        positive_probabilities=np.array([0.10, 0.45, 0.80, 0.55]),
        thresholds=(0.40, 0.60),
        minimum_recall=1.0,
    )

    assert result["threshold"] == 0.4
    assert result["recall"] == 1.0
    assert "recall floor" in result["selection_reason"]


def test_select_threshold_with_recall_floor_falls_back_to_highest_recall() -> None:
    result = select_threshold_with_recall_floor(
        labels=np.array([0, 1]),
        positive_probabilities=np.array([0.70, 0.20]),
        thresholds=(0.60, 0.80),
        minimum_recall=0.5,
    )

    assert result["recall"] == 0.0
    assert "no threshold satisfied" in result["selection_reason"]


def test_cost_sensitive_threshold_penalizes_false_negatives() -> None:
    result = select_threshold_by_cost(
        labels=np.array([0, 0, 1, 1]),
        positive_probabilities=np.array([0.20, 0.55, 0.45, 0.80]),
        thresholds=(0.40, 0.60),
        false_positive_cost=1.0,
        false_negative_cost=4.0,
    )

    assert result["threshold"] == 0.4
    assert result["false_negative"] == 0
    assert result["cost_per_example"] == 0.25


def test_threshold_sweep_rejects_probability_range_errors() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        threshold_sweep(
            labels=np.array([0, 1]),
            positive_probabilities=np.array([0.2, 1.2]),
            thresholds=(0.5,),
        )


def test_calibration_report_tracks_probability_reliability() -> None:
    report = calibration_report(
        labels=np.array([0, 0, 1, 1]),
        positive_probabilities=np.array([0.10, 0.30, 0.70, 0.90]),
        bins=2,
    )

    assert report["brier_score"] == pytest.approx(0.05)
    assert report["expected_calibration_error"] == pytest.approx(0.20)
    assert report["brier_decomposition"]["reliability"] == pytest.approx(0.04)
    assert report["brier_decomposition"]["resolution"] == pytest.approx(0.25)
    assert len(report["table"]) == 2


def test_brier_decomposition_rewards_resolved_probability_groups() -> None:
    report = brier_decomposition(
        labels=np.array([0, 0, 1, 1]),
        positive_probabilities=np.array([0.05, 0.15, 0.85, 0.95]),
        bins=2,
    )

    assert report["uncertainty"] == pytest.approx(0.25)
    assert report["resolution"] == pytest.approx(0.25)
    assert report["reliability"] == pytest.approx(0.01)
    assert report["decomposed_brier_score"] == pytest.approx(0.01)


def test_confidence_slices_separate_error_bands() -> None:
    rows = confidence_slices(
        labels=np.array([0, 1, 1, 0]),
        positive_probabilities=np.array([0.48, 0.72, 0.93, 0.88]),
        threshold=0.5,
    )
    by_slice = {row["slice"]: row for row in rows}

    assert by_slice["low"]["count"] == 1
    assert by_slice["medium"]["accuracy"] == 1.0
    assert by_slice["high"]["error_rate"] == 0.5


def test_uncertainty_summary_separates_entropy_from_decision_margin() -> None:
    report = uncertainty_summary(
        labels=np.array([0, 1, 1, 0]),
        positive_probabilities=np.array([0.05, 0.55, 0.95, 0.95]),
    )

    assert report["n_examples"] == 4
    assert report["low_margin_rate"] == 0.25
    assert report["mean_error_entropy_bits"] < report["mean_correct_entropy_bits"]
    assert 0.0 < report["mean_entropy_bits"] < 1.0


def test_selective_risk_removes_uncertain_errors() -> None:
    report = selective_risk_report(
        labels=np.array([0, 1, 1, 0]),
        positive_probabilities=np.array([0.10, 0.90, 0.49, 0.80]),
        coverages=(1.0, 0.5),
    )

    assert report["points"][0]["selective_risk"] == 0.5
    assert report["points"][1]["selective_risk"] == 0.0
    assert report["points"][1]["retained"] == 2


def test_bootstrap_intervals_capture_selected_threshold_uncertainty() -> None:
    report = bootstrap_metric_intervals(
        labels=np.array([0, 0, 1, 1, 0, 1]),
        positive_probabilities=np.array([0.1, 0.4, 0.6, 0.9, 0.2, 0.8]),
        threshold=0.5,
        samples=100,
        seed=9,
    )

    assert report["used_samples"] == 100
    assert report["threshold"] == 0.5
    assert report["metrics"]["f1"]["lower"] <= report["metrics"]["f1"]["upper"]


def test_confident_error_records_are_ranked_by_confidence() -> None:
    rows = confident_error_records(
        labels=np.array([0, 1, 1, 0]),
        positive_probabilities=np.array([0.91, 0.20, 0.65, 0.40]),
        threshold=0.5,
        top_k=2,
    )

    assert [row["error_type"] for row in rows] == [
        "false_positive",
        "false_negative",
    ]
    assert rows[0]["confidence"] > rows[1]["confidence"]


def test_calibration_report_rejects_invalid_probabilities() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        calibration_report(np.array([0, 1]), np.array([0.2, 1.4]))


def test_evaluate_test_set_writes_threshold_reports(tmp_path) -> None:
    metrics = evaluate_test_set(
        trainer=TinyTrainer(),
        test_dataset=tiny_dataset()["test"],
        report_dir=tmp_path,
        thresholds=(0.40, 0.50, 0.60),
        minimum_recall=0.5,
    )

    sweep = json.loads((tmp_path / "threshold_sweep.json").read_text())
    cost_selection = json.loads(
        (tmp_path / "threshold_cost_selection.json").read_text()
    )
    recall_floor = json.loads((tmp_path / "threshold_recall_floor.json").read_text())
    calibration = json.loads((tmp_path / "calibration_report.json").read_text())
    confidence = json.loads((tmp_path / "confidence_slices.json").read_text())
    uncertainty = json.loads((tmp_path / "uncertainty_summary.json").read_text())
    selective_risk = json.loads((tmp_path / "selective_risk.json").read_text())
    intervals = json.loads((tmp_path / "metric_intervals.json").read_text())
    confident_errors = json.loads((tmp_path / "confident_errors.json").read_text())
    predictions = json.loads((tmp_path / "test_predictions.json").read_text())
    assert metrics["test_loss"] == 0.2
    assert metrics["test_best_threshold"] in {0.4, 0.5, 0.6}
    assert "test_recall_floor_threshold" in metrics
    assert recall_floor["minimum_recall"] == 0.5
    assert "test_expected_calibration_error" in metrics
    assert len(sweep) == 3
    assert metrics["test_cost_selected_threshold"] == cost_selection["threshold"]
    assert calibration["n_examples"] == 3
    assert {row["slice"] for row in confidence} == {"low", "medium", "high"}
    assert metrics["test_mean_entropy_bits"] == uncertainty["mean_entropy_bits"]
    assert "area_under_risk_coverage" in selective_risk
    assert "test_risk_at_half_coverage" in metrics
    assert metrics["test_f1_ci95_low"] <= metrics["test_f1_ci95_high"]
    assert intervals["threshold"] == metrics["test_best_threshold"]
    assert metrics["test_confident_error_count"] == len(confident_errors)
    assert predictions[0]["predicted_label"] == 0


def test_fine_tuning_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        FineTuneConfig(learning_rate=0)


def test_fine_tuning_config_rejects_bad_decision_thresholds() -> None:
    with pytest.raises(ValueError, match="decision_thresholds"):
        FineTuneConfig(decision_thresholds=(0.5, 1.2))


def test_fine_tuning_config_rejects_empty_cost_model() -> None:
    with pytest.raises(ValueError, match="at least one classification cost"):
        FineTuneConfig(false_positive_cost=0.0, false_negative_cost=0.0)


def test_fine_tuning_config_accepts_custom_lora_targets() -> None:
    config = FineTuneConfig(lora_target_modules=("q_lin",))
    assert config.lora_target_modules == ("q_lin",)


def test_parameter_report_separates_adapter_and_head_weights() -> None:
    report = parameter_report(TinyAdapterModel())

    assert report["total_parameters"] == 12
    assert report["trainable_parameters"] == 6
    assert report["frozen_parameters"] == 6
    assert report["trainable_lora_parameters"] == 2
    assert report["trainable_head_parameters"] == 4


def test_optimization_plan_accounts_for_gradient_accumulation() -> None:
    config = FineTuneConfig(
        train_batch_size=8,
        gradient_accumulation_steps=4,
        epochs=2.0,
        warmup_ratio=0.1,
    )

    report = optimization_plan(100, config)

    assert report["effective_batch_size"] == 32
    assert report["micro_batches_per_epoch"] == 13
    assert report["optimizer_steps_per_epoch"] == 4
    assert report["total_optimizer_steps"] == 8
    assert report["warmup_steps"] == 1


def test_inference_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="at least one text"):
        predict([], model=None, tokenizer=None, device=None)


def test_inference_returns_positive_probability() -> None:
    predictions = predict(
        ["  mixed  ", "great"],
        InferenceModel(),
        InferenceTokenizer(),
        torch.device("cpu"),
    )
    assert predictions[0]["text"] == "mixed"
    assert predictions[0]["label"] == "negative"
    assert predictions[0]["positive_probability"] == pytest.approx(0.2689, abs=1e-4)
    assert predictions[1]["positive_probability"] == pytest.approx(0.8808, abs=1e-4)
    assert predictions[1]["positive_threshold"] == 0.5


def test_inference_uses_positive_threshold_for_labels() -> None:
    predictions = predict(
        ["mixed", "great"],
        InferenceModel(),
        InferenceTokenizer(),
        torch.device("cpu"),
        positive_threshold=0.9,
    )

    assert predictions[1]["label"] == "negative"
    assert predictions[1]["confidence"] == pytest.approx(1 - 0.8808, abs=1e-4)


def test_inference_can_abstain_on_low_confidence_predictions() -> None:
    predictions = predict(
        ["mixed", "great"],
        InferenceModel(),
        InferenceTokenizer(),
        torch.device("cpu"),
        abstain_below=0.8,
    )

    assert predictions[0]["label"] == "abstain"
    assert predictions[0]["abstained"] is True
    assert predictions[1]["label"] == "positive"
    assert predictions[1]["abstained"] is False


def test_inference_rejects_invalid_abstention_threshold() -> None:
    with pytest.raises(ValueError, match="abstain_below"):
        predict(
            ["fine"],
            InferenceModel(),
            InferenceTokenizer(),
            torch.device("cpu"),
            abstain_below=0.4,
        )


def test_inference_rejects_invalid_positive_threshold() -> None:
    with pytest.raises(ValueError, match="positive_threshold"):
        predict(
            ["fine"],
            InferenceModel(),
            InferenceTokenizer(),
            torch.device("cpu"),
            positive_threshold=1.0,
        )


def test_inference_batches_longer_text_lists() -> None:
    predictions = predict(
        ["first", "second", "third"],
        BatchAwareInferenceModel(),
        InferenceTokenizer(),
        torch.device("cpu"),
        batch_size=1,
    )

    assert len(predictions) == 3
    assert [row["text"] for row in predictions] == ["first", "second", "third"]


def test_inference_rejects_invalid_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        predict(
            ["fine"],
            InferenceModel(),
            InferenceTokenizer(),
            torch.device("cpu"),
            batch_size=0,
        )
