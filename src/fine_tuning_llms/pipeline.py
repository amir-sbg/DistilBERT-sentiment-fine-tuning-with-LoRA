from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from transformers import set_seed

from .config import FineTuneConfig, ensure_output_directories
from .data import load_sentiment_dataset, split_sizes, tokenize_dataset
from .evaluate import classification_metrics, evaluate_test_set
from .model import build_lora_model, load_tokenizer, parameter_report
from .train import fine_tune


def _config_payload(config: FineTuneConfig) -> dict:
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir)
    payload["report_dir"] = str(config.report_dir)
    return payload


def run(config: FineTuneConfig) -> dict[str, float]:
    ensure_output_directories(config)
    set_seed(config.seed)

    dataset = load_sentiment_dataset(
        dataset_name=config.dataset_name,
        validation_size=config.validation_size,
        seed=config.seed,
        max_train_samples=config.max_train_samples,
        max_validation_samples=config.max_validation_samples,
        max_test_samples=config.max_test_samples,
    )
    tokenizer = load_tokenizer(config.model_name)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, config.max_length)
    (config.report_dir / "dataset_sizes.json").write_text(
        json.dumps(split_sizes(dataset), indent=2) + "\n"
    )
    (config.report_dir / "run_config.json").write_text(
        json.dumps(_config_payload(config), indent=2) + "\n"
    )

    model = build_lora_model(
        model_name=config.model_name,
        num_labels=2,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
    )
    adapter_report = parameter_report(model)
    (config.report_dir / "adapter_parameter_report.json").write_text(
        json.dumps(adapter_report, indent=2) + "\n"
    )
    trainer = fine_tune(
        model=model,
        tokenizer=tokenizer,
        dataset=tokenized_dataset,
        config=config,
        compute_metrics=classification_metrics,
    )
    metrics = evaluate_test_set(
        trainer=trainer,
        test_dataset=tokenized_dataset["test"],
        report_dir=config.report_dir,
        thresholds=config.decision_thresholds,
    )
    summary = {
        "model_name": config.model_name,
        "dataset_name": config.dataset_name,
        "adapter_dir": str(config.output_dir),
        "trainable_fraction": adapter_report["trainable_fraction"],
        "trainable_parameters": adapter_report["trainable_parameters"],
        **metrics,
    }
    (config.report_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune a sequence classifier with LoRA adapters."
    )
    parser.add_argument("--dataset-name", default="imdb")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/imdb-distilbert-lora"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--validation-size", type=float, default=0.10)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.10)
    parser.add_argument("--lora-target-modules", nargs="+", default=["q_lin", "v_lin"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument(
        "--decision-thresholds",
        nargs="+",
        type=float,
        default=[0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65],
    )
    return parser


def config_from_args(args: argparse.Namespace) -> FineTuneConfig:
    values = vars(args).copy()
    values["lora_target_modules"] = tuple(values["lora_target_modules"])
    values["decision_thresholds"] = tuple(values["decision_thresholds"])
    return FineTuneConfig(**values)


if __name__ == "__main__":
    run(config_from_args(build_parser().parse_args()))
