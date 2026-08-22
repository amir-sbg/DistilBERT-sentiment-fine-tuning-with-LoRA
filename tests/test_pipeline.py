from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from datasets import Dataset, DatasetDict
from transformers import EvalPrediction

from fine_tuning_llms.config import FineTuneConfig
from fine_tuning_llms.data import tokenize_dataset, validate_dataset
from fine_tuning_llms.evaluate import (
    classification_metrics,
    evaluate_test_set,
    threshold_sweep,
)
from fine_tuning_llms.inference import predict


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


def test_classification_metrics_are_consistent() -> None:
    prediction = EvalPrediction(
        predictions=np.array([[3.0, 1.0], [0.5, 2.0], [2.0, 1.0], [0.0, 2.5]]),
        label_ids=np.array([0, 1, 0, 1]),
    )
    metrics = classification_metrics(prediction)
    assert metrics == {
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }


def test_threshold_sweep_reports_decision_tradeoffs() -> None:
    rows = threshold_sweep(
        labels=np.array([0, 1, 1, 0]),
        positive_probabilities=np.array([0.10, 0.55, 0.80, 0.60]),
        thresholds=(0.50, 0.70),
    )

    assert rows[0]["threshold"] == 0.5
    assert rows[0]["positive_rate"] == 0.75
    assert rows[1]["recall"] == 0.5


def test_evaluate_test_set_writes_threshold_reports(tmp_path) -> None:
    metrics = evaluate_test_set(
        trainer=TinyTrainer(),
        test_dataset=tiny_dataset()["test"],
        report_dir=tmp_path,
        thresholds=(0.40, 0.50, 0.60),
    )

    sweep = json.loads((tmp_path / "threshold_sweep.json").read_text())
    predictions = json.loads((tmp_path / "test_predictions.json").read_text())
    assert metrics["test_loss"] == 0.2
    assert metrics["test_best_threshold"] in {0.4, 0.5, 0.6}
    assert len(sweep) == 3
    assert predictions[0]["predicted_label"] == 0


def test_fine_tuning_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        FineTuneConfig(learning_rate=0)


def test_fine_tuning_config_rejects_bad_decision_thresholds() -> None:
    with pytest.raises(ValueError, match="decision_thresholds"):
        FineTuneConfig(decision_thresholds=(0.5, 1.2))


def test_fine_tuning_config_accepts_custom_lora_targets() -> None:
    config = FineTuneConfig(lora_target_modules=("q_lin",))
    assert config.lora_target_modules == ("q_lin",)


def test_inference_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="at least one text"):
        predict([], model=None, tokenizer=None, device=None)


def test_inference_returns_positive_probability() -> None:
    predictions = predict(
        ["mixed", "great"],
        InferenceModel(),
        InferenceTokenizer(),
        torch.device("cpu"),
    )
    assert predictions[0]["label"] == "negative"
    assert predictions[0]["positive_probability"] == pytest.approx(0.2689, abs=1e-4)
    assert predictions[1]["positive_probability"] == pytest.approx(0.8808, abs=1e-4)
