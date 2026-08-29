from __future__ import annotations

from torch import nn
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


def parameter_report(model: nn.Module) -> dict:
    rows = []
    total_parameters = 0
    trainable_parameters = 0
    trainable_lora_parameters = 0
    trainable_head_parameters = 0

    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total_parameters += count
        if not parameter.requires_grad:
            continue
        trainable_parameters += count
        kind = _parameter_kind(name)
        if kind == "lora":
            trainable_lora_parameters += count
        if kind == "classification_head":
            trainable_head_parameters += count
        rows.append(
            {
                "name": name,
                "parameters": int(count),
                "kind": kind,
            }
        )

    if total_parameters == 0:
        raise ValueError("model has no parameters")
    return {
        "total_parameters": int(total_parameters),
        "trainable_parameters": int(trainable_parameters),
        "frozen_parameters": int(total_parameters - trainable_parameters),
        "trainable_fraction": float(trainable_parameters / total_parameters),
        "trainable_lora_parameters": int(trainable_lora_parameters),
        "trainable_head_parameters": int(trainable_head_parameters),
        "trainable_tensors": rows,
    }


def _parameter_kind(name: str) -> str:
    lowered = name.lower()
    if "lora_" in lowered:
        return "lora"
    if "classifier" in lowered or "pre_classifier" in lowered:
        return "classification_head"
    return "other_trainable"
