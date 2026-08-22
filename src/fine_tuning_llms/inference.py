from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import AutoPeftModelForSequenceClassification
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_adapter(
    adapter_dir: Path,
    device: str = "auto",
) -> tuple[AutoPeftModelForSequenceClassification, PreTrainedTokenizerBase, torch.device]:
    selected_device = _auto_device() if device == "auto" else torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    model = AutoPeftModelForSequenceClassification.from_pretrained(adapter_dir)
    model.to(selected_device)
    model.eval()
    return model, tokenizer, selected_device


def predict(
    texts: list[str],
    model: AutoPeftModelForSequenceClassification,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    max_length: int = 256,
    positive_threshold: float = 0.5,
) -> list[dict[str, str | float]]:
    if not texts:
        raise ValueError("at least one text is required")
    if any(not text.strip() for text in texts):
        raise ValueError("texts must not be empty")
    if max_length < 1:
        raise ValueError("max_length must be at least 1")
    if positive_threshold <= 0 or positive_threshold >= 1:
        raise ValueError("positive_threshold must be between 0 and 1")

    encoded = tokenizer(
        texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(**encoded).logits, dim=-1)
    positive_probability = probabilities[:, 1].cpu().tolist()
    predictions = [
        1 if score >= positive_threshold else 0
        for score in positive_probability
    ]
    confidence = [
        score if label == 1 else 1.0 - score
        for label, score in zip(predictions, positive_probability)
    ]
    return [
        {
            "text": text,
            "label": "positive" if label == 1 else "negative",
            "confidence": float(score),
            "positive_probability": float(score_for_positive),
            "positive_threshold": positive_threshold,
        }
        for text, label, score, score_for_positive in zip(
            texts,
            predictions,
            confidence,
            positive_probability,
        )
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run sentiment inference with a LoRA adapter.")
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--text", nargs="+", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--positive-threshold", type=float, default=0.5)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    model, tokenizer, device = load_adapter(args.adapter_dir, args.device)
    print(
        json.dumps(
            predict(
                args.text,
                model,
                tokenizer,
                device,
                args.max_length,
                args.positive_threshold,
            ),
            indent=2,
        )
    )
