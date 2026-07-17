from __future__ import annotations

from collections.abc import Mapping

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
    return dataset.shuffle(seed=seed).select(range(count))


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
