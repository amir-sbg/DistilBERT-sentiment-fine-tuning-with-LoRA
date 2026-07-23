from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FineTuneConfig:
    dataset_name: str = "imdb"
    model_name: str = "distilbert-base-uncased"
    output_dir: Path = Path("artifacts/imdb-distilbert-lora")
    report_dir: Path = Path("reports")
    max_length: int = 256
    validation_size: float = 0.10
    train_batch_size: int = 8
    eval_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    epochs: float = 1.0
    patience: int = 2
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.10
    lora_target_modules: tuple[str, ...] = ("q_lin", "v_lin")
    seed: int = 42
    max_train_samples: int | None = None
    max_validation_samples: int | None = None
    max_test_samples: int | None = None

    def __post_init__(self) -> None:
        if self.max_length < 1:
            raise ValueError("max_length must be at least 1")
        if not 0 < self.validation_size < 1:
            raise ValueError("validation_size must be between 0 and 1")
        if self.train_batch_size < 1 or self.eval_batch_size < 1:
            raise ValueError("batch sizes must be at least 1")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must not be negative")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be between 0 and 1")
        if self.epochs <= 0:
            raise ValueError("epochs must be greater than 0")
        if self.patience < 1:
            raise ValueError("patience must be at least 1")
        if self.lora_rank < 1 or self.lora_alpha < 1:
            raise ValueError("LoRA rank and alpha must be at least 1")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be between 0 and 1")
        if not self.lora_target_modules or any(
            not isinstance(module, str) or not module.strip()
            for module in self.lora_target_modules
        ):
            raise ValueError("lora_target_modules must contain at least one name")


def ensure_output_directories(config: FineTuneConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.report_dir.mkdir(parents=True, exist_ok=True)
