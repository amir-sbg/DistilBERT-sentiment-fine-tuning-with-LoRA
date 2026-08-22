# DistilBERT Sentiment Fine-Tuning with LoRA

Supervised text-classification pipeline for adapting `distilbert-base-uncased` to IMDb movie reviews with Hugging Face Transformers, Datasets, PyTorch, and PEFT.

## Overview

The project loads the IMDb dataset, creates a stratified validation split, tokenizes the reviews, and fine-tunes a parameter-efficient adapter. The base DistilBERT weights stay frozen while LoRA updates are learned in the attention layers. The classification head is saved with the adapter so the trained result can be loaded independently for inference.

The test split is kept separate from training and checkpoint selection. Validation F1 is used to select the best checkpoint, while final test metrics, prediction confidence rows, and decision-threshold sweeps are written after training is complete.

## Pipeline

1. Load and validate the IMDb train and test splits.
2. Create a seeded, stratified validation split from the training data.
3. Tokenize reviews with truncation and dynamic batch padding.
4. Add LoRA adapters to the DistilBERT query and value projections.
5. Fine-tune with `Trainer`, AdamW, gradient accumulation, and early stopping.
6. Evaluate on the untouched test set and save the adapter for inference.
7. Sweep sentiment decision thresholds to inspect the precision/recall tradeoff before deployment.

## Fine-tuning setup

The default configuration uses:

- LoRA rank: `8`
- LoRA alpha: `16`
- LoRA dropout: `0.10`
- Target modules: `q_lin`, `v_lin`
- Learning rate: `2e-4`
- Weight decay: `0.01`
- Maximum sequence length: `256`

The DistilBERT classification layers are included in the saved trainable modules because the original checkpoint does not contain an IMDb-specific classification head.

## Evaluation

The test report includes loss, accuracy, precision, recall, F1, runtime, and throughput. The run also records dataset sizes and the exact configuration used for training.

## Installation

```bash
git clone https://github.com/amir-sbg/DistilBERT-sentiment-fine-tuning-with-LoRA.git
cd DistilBERT-sentiment-fine-tuning-with-LoRA

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
```

## Training

Run the complete dataset:

```bash
python -m fine_tuning_llms.pipeline
```

For a quick local run, limit the number of examples:

```bash
python -m fine_tuning_llms.pipeline \
  --max-train-samples 256 \
  --max-validation-samples 64 \
  --max-test-samples 64 \
  --epochs 1 \
  --max-length 128 \
  --output-dir artifacts/imdb-smoke \
  --report-dir reports/imdb-smoke
```

Training options include the model name, sequence length, batch sizes, learning rate, LoRA parameters, seed, sample limits, and decision thresholds for post-training analysis.

## Inference

Use a saved adapter to classify new reviews:

```bash
python -m fine_tuning_llms.inference \
  --adapter-dir artifacts/imdb-distilbert-lora \
  --text "A thoughtful, well-acted film with a strong ending." \
        "The story was slow and difficult to follow." \
  --device auto
```

With `--device auto`, inference selects CUDA, Apple MPS, or CPU when available.

## Outputs

```text
artifacts/imdb-distilbert-lora/
├── adapter_config.json
├── adapter_model.safetensors
├── config.json
├── tokenizer.json
└── tokenizer_config.json

reports/
├── dataset_sizes.json
├── run_config.json
├── run_summary.json
├── test_predictions.json
├── threshold_sweep.json
└── test_metrics.json
```

## Project structure

```text
.
├── src/fine_tuning_llms/
│   ├── config.py       # training configuration
│   ├── data.py         # dataset loading and tokenization
│   ├── model.py        # tokenizer and LoRA model setup
│   ├── train.py        # Trainer configuration and checkpointing
│   ├── evaluate.py     # metrics and test reporting
│   ├── inference.py    # adapter loading and prediction
│   └── pipeline.py     # command-line workflow
├── tests/test_pipeline.py
├── .github/workflows/ci.yml
├── Makefile
├── pyproject.toml
└── requirements.txt
```
