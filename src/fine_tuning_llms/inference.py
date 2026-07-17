from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import AutoPeftModelForSequenceClassification
from transformers import PreTrainedTokenizerBase, AutoTokenizer


def load_adapter(
    adapter_dir: Path,
    device: str = "auto",
) -> tuple[AutoPeftModelForSequenceClassification, PreTrainedTokenizerBase, torch.device]:
    selected_device = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else device
    )
    if device == "auto" and not torch.cuda.is_available():
        selected_device = torch.device("cpu")
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
) -> list[dict[str, str | float]]:
    encoded = tokenizer(
        texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(**encoded).logits, dim=-1)
    predictions = probabilities.argmax(dim=-1).cpu().tolist()
    confidence = probabilities.max(dim=-1).values.cpu().tolist()
    return [
        {
            "label": "positive" if label == 1 else "negative",
            "confidence": float(score),
        }
        for label, score in zip(predictions, confidence)
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run sentiment inference with a LoRA adapter.")
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--text", nargs="+", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=256)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    model, tokenizer, device = load_adapter(args.adapter_dir, args.device)
    print(json.dumps(predict(args.text, model, tokenizer, device, args.max_length), indent=2))
