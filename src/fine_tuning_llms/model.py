from __future__ import annotations

from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def build_lora_model(
    model_name: str,
    num_labels: int,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: tuple[str, ...] = ("q_lin", "v_lin"),
) -> PreTrainedModel:
    if num_labels < 2:
        raise ValueError("num_labels must be at least 2")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=list(target_modules),
        modules_to_save=["pre_classifier", "classifier"],
    )
    return get_peft_model(base_model, lora_config)
