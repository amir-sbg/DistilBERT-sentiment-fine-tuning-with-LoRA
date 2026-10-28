from __future__ import annotations

import math
from collections.abc import Callable
from inspect import signature

from datasets import DatasetDict
from transformers import (
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from .config import FineTuneConfig


def optimization_plan(
    train_examples: int,
    config: FineTuneConfig,
    world_size: int = 1,
) -> dict[str, int | float]:
    if train_examples < 1:
        raise ValueError("train_examples must be at least 1")
    if world_size < 1:
        raise ValueError("world_size must be at least 1")

    examples_per_micro_batch = config.train_batch_size * world_size
    micro_batches_per_epoch = math.ceil(train_examples / examples_per_micro_batch)
    optimizer_steps_per_epoch = math.ceil(
        micro_batches_per_epoch / config.gradient_accumulation_steps
    )
    total_optimizer_steps = math.ceil(optimizer_steps_per_epoch * config.epochs)
    return {
        "train_examples": train_examples,
        "world_size": world_size,
        "per_device_batch_size": config.train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "effective_batch_size": (
            config.train_batch_size
            * config.gradient_accumulation_steps
            * world_size
        ),
        "micro_batches_per_epoch": micro_batches_per_epoch,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "total_optimizer_steps": total_optimizer_steps,
        "warmup_steps": math.ceil(total_optimizer_steps * config.warmup_ratio),
        "label_smoothing_factor": config.label_smoothing_factor,
        "max_grad_norm": config.max_grad_norm,
    }


def build_training_arguments(config: FineTuneConfig) -> TrainingArguments:
    arguments = {
        "output_dir": str(config.output_dir),
        "learning_rate": config.learning_rate,
        "per_device_train_batch_size": config.train_batch_size,
        "per_device_eval_batch_size": config.eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "num_train_epochs": config.epochs,
        "weight_decay": config.weight_decay,
        "label_smoothing_factor": config.label_smoothing_factor,
        "max_grad_norm": config.max_grad_norm,
        "warmup_ratio": config.warmup_ratio,
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 25,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "report_to": "none",
        "fp16": False,
        "dataloader_pin_memory": False,
        "seed": config.seed,
        "data_seed": config.seed,
    }
    strategy_parameter = (
        "eval_strategy"
        if "eval_strategy" in signature(TrainingArguments).parameters
        else "evaluation_strategy"
    )
    arguments[strategy_parameter] = "epoch"
    return TrainingArguments(**arguments)


def fine_tune(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: DatasetDict,
    config: FineTuneConfig,
    compute_metrics: Callable,
) -> Trainer:
    trainer_arguments = {
        "model": model,
        "args": build_training_arguments(config),
        "train_dataset": dataset["train"],
        "eval_dataset": dataset["validation"],
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
        "callbacks": [
            EarlyStoppingCallback(early_stopping_patience=config.patience)
        ],
    }
    tokenizer_parameter = (
        "processing_class"
        if "processing_class" in signature(Trainer).parameters
        else "tokenizer"
    )
    trainer_arguments[tokenizer_parameter] = tokenizer
    trainer = Trainer(**trainer_arguments)
    trainer.train()
    trainer.save_model(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))
    trainer.remove_callback(EarlyStoppingCallback)
    return trainer
