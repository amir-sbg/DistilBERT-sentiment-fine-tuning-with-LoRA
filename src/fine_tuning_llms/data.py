from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase


TEXT_COLUMN = "text"
LABEL_COLUMN = "label"


def validate_dataset(dataset: Mapping[str, Dataset]) -> None:
    required_splits = {"train", "validation", "test"}
    missing_splits = required_splits.difference(dataset)
    if missing_splits:
        raise ValueError(f"Dataset is missing splits: {sorted(missing_splits)}")

    for split_name, split in dataset.items():
        missing_columns = {TEXT_COLUMN, LABEL_COLUMN}.difference(split.column_names)
        if missing_columns:
            raise ValueError(
                f"{split_name} split is missing columns: {sorted(missing_columns)}"
            )


def _limit_split(dataset: Dataset, limit: int | None, seed: int) -> Dataset:
    if limit is None:
        return dataset
    if limit < 1:
        raise ValueError("sample limits must be at least 1")
    count = min(limit, len(dataset))
    if count == len(dataset):
        return dataset

    labels = np.asarray(dataset[LABEL_COLUMN], dtype=int)
    classes, class_counts = np.unique(labels, return_counts=True)
    exact = class_counts / class_counts.sum() * count
    quotas = np.floor(exact).astype(int)
    if count >= len(classes):
        quotas = np.maximum(quotas, 1)
    quotas = np.minimum(quotas, class_counts)

    while quotas.sum() < count:
        candidates = np.flatnonzero(quotas < class_counts)
        index = max(candidates, key=lambda item: (exact[item] - quotas[item], class_counts[item]))
        quotas[index] += 1
    while quotas.sum() > count:
        minimum = 1 if count >= len(classes) else 0
        candidates = np.flatnonzero(quotas > minimum)
        index = min(candidates, key=lambda item: (exact[item] - quotas[item], -quotas[item]))
        quotas[index] -= 1

    generator = np.random.default_rng(seed)
    selected = []
    for label, quota in zip(classes, quotas):
        class_indices = np.flatnonzero(labels == label)
        selected.extend(generator.choice(class_indices, size=quota, replace=False).tolist())
    generator.shuffle(selected)
    return dataset.select(selected)


def load_sentiment_dataset(
    dataset_name: str,
    validation_size: float,
    seed: int,
    max_train_samples: int | None = None,
    max_validation_samples: int | None = None,
    max_test_samples: int | None = None,
) -> DatasetDict:
    dataset = load_dataset(dataset_name)
    if "train" not in dataset or "test" not in dataset:
        raise ValueError("The dataset must provide train and test splits")

    if "validation" not in dataset:
        split = dataset["train"].train_test_split(
            test_size=validation_size,
            stratify_by_column=LABEL_COLUMN,
            seed=seed,
        )
        dataset = DatasetDict(
            {
                "train": split["train"],
                "validation": split["test"],
                "test": dataset["test"],
            }
        )
    else:
        dataset = DatasetDict(
            {
                "train": dataset["train"],
                "validation": dataset["validation"],
                "test": dataset["test"],
            }
        )

    limited = DatasetDict(
        {
            "train": _limit_split(dataset["train"], max_train_samples, seed),
            "validation": _limit_split(
                dataset["validation"], max_validation_samples, seed + 1
            ),
            "test": _limit_split(dataset["test"], max_test_samples, seed + 2),
        }
    )
    validate_dataset(limited)
    return limited


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> DatasetDict:
    if max_length < 1:
        raise ValueError("max_length must be at least 1")
    validate_dataset(dataset)

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(
            batch[TEXT_COLUMN],
            max_length=max_length,
            truncation=True,
        )

    return dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=[TEXT_COLUMN],
        desc="Tokenizing dataset",
    )


def split_sizes(dataset: Mapping[str, Dataset]) -> dict[str, int]:
    return {split_name: len(split) for split_name, split in dataset.items()}


def label_distribution(dataset: Mapping[str, Dataset]) -> dict[str, dict[str, float | int]]:
    validate_dataset(dataset)
    report = {}
    for split_name, split in dataset.items():
        labels = split[LABEL_COLUMN]
        positives = int(sum(1 for label in labels if int(label) == 1))
        rows = int(len(labels))
        report[split_name] = {
            "rows": rows,
            "negative": int(rows - positives),
            "positive": positives,
            "positive_rate": float(positives / rows) if rows else 0.0,
        }
    return report


def token_length_report(
    dataset: Mapping[str, Dataset],
    max_length: int,
) -> dict[str, dict[str, float | int]]:
    """Describe sequence lengths after truncation, before dynamic padding."""

    if max_length < 1:
        raise ValueError("max_length must be at least 1")
    report = {}
    for split_name, split in dataset.items():
        if "input_ids" not in split.column_names:
            raise ValueError(f"{split_name} split is missing input_ids")
        lengths = np.asarray([len(tokens) for tokens in split["input_ids"]], dtype=int)
        if len(lengths) == 0:
            raise ValueError(f"{split_name} split is empty")
        report[split_name] = {
            "rows": int(len(lengths)),
            "mean_tokens": float(lengths.mean()),
            "median_tokens": float(np.median(lengths)),
            "p95_tokens": float(np.quantile(lengths, 0.95)),
            "max_tokens": int(lengths.max()),
            "at_length_limit": int(np.sum(lengths >= max_length)),
            "at_length_limit_rate": float(np.mean(lengths >= max_length)),
        }
    return report
