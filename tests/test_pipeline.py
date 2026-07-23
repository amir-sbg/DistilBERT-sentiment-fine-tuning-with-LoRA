from __future__ import annotations

import numpy as np
import pytest
from datasets import Dataset, DatasetDict
from transformers import EvalPrediction

from fine_tuning_llms.config import FineTuneConfig
from fine_tuning_llms.data import tokenize_dataset, validate_dataset
from fine_tuning_llms.evaluate import classification_metrics
from fine_tuning_llms.inference import predict


class TinyTokenizer:
    def __call__(self, texts, max_length, truncation):
        input_ids = [list(range(min(len(text.split()), max_length))) for text in texts]
        return {
            "input_ids": [tokens or [0] for tokens in input_ids],
            "attention_mask": [[1] * max(len(tokens), 1) for tokens in input_ids],
        }


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


def test_fine_tuning_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        FineTuneConfig(learning_rate=0)


def test_fine_tuning_config_accepts_custom_lora_targets() -> None:
    config = FineTuneConfig(lora_target_modules=("q_lin",))
    assert config.lora_target_modules == ("q_lin",)


def test_inference_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="at least one text"):
        predict([], model=None, tokenizer=None, device=None)
