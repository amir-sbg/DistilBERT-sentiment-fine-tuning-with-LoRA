# DistilBERT Sentiment Fine-Tuning with LoRA

A parameter-efficient text-classification pipeline for adapting `distilbert-base-uncased` to IMDb movie reviews. It uses Hugging Face Datasets and Transformers, PyTorch, and PEFT to train LoRA adapters while keeping the base encoder frozen.

The workflow keeps validation and test data separate, selects the best checkpoint by validation F1, and fits temperature scaling on validation logits before measuring calibrated test probabilities. Held-out evaluation covers balanced and threshold-free metrics, fixed-width and adaptive calibration, Brier decomposition, excess selective risk, bootstrap uncertainty, and decision-threshold diagnostics. Sample-limited runs preserve class balance, while sequence-length profiles expose how often reviews reach the truncation limit. Each run also records the effective batch size and expected optimizer schedule.

## Pipeline

1. Load IMDb and create a seeded, stratified validation split.
2. Tokenize reviews with truncation and dynamic padding.
3. Add LoRA adapters to DistilBERT’s `q_lin` and `v_lin` projections.
4. Fine-tune with `Trainer`, AdamW, gradient accumulation, gradient clipping, optional label smoothing, and early stopping.
5. Save the adapter, classification head, tokenizer, configuration, and parameter report.
6. Review calibration, predictive entropy, confidence slices, selective risk, confident mistakes, and threshold trade-offs.

Default LoRA settings are rank 8, alpha 16, dropout 0.10, and a maximum sequence length of 256. The default learning rate is `2e-4`.

## Setup

```bash
git clone https://github.com/amir-sbg/DistilBERT-sentiment-fine-tuning-with-LoRA.git
cd DistilBERT-sentiment-fine-tuning-with-LoRA
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pytest -q
```

## Train

Run the full experiment:

```bash
python -m fine_tuning_llms.pipeline
```

Use sample limits for a quick local run:

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

Set `--minimum-recall 0.90` to report thresholds that satisfy a minimum positive-class recall. For asymmetric decisions, `--false-positive-cost` and `--false-negative-cost` compare the measured cost of false positives and false negatives across the threshold sweep.

## Inference

```bash
python -m fine_tuning_llms.inference \
  --adapter-dir artifacts/imdb-distilbert-lora \
  --text "A thoughtful, well-acted film with a strong ending." \
        "The story was slow and difficult to follow." \
  --device auto \
  --positive-threshold 0.50 \
  --abstain-below 0.70 \
  --batch-size 16
```

`auto` selects CUDA, Apple MPS, or CPU. The decision threshold can be chosen from the saved threshold sweep, while `--abstain-below` returns `abstain` when neither sentiment prediction is sufficiently confident.

## Reports

```text
artifacts/imdb-distilbert-lora/
├── adapter_config.json
├── adapter_model.safetensors
├── config.json
└── tokenizer files

reports/
├── dataset_profile.json
├── adapter_parameter_report.json
├── optimization_plan.json
├── test_metrics.json
├── test_predictions.json
├── calibration_report.json
├── temperature_scaling.json
├── confidence_slices.json
├── uncertainty_summary.json
├── selective_risk.json
├── metric_intervals.json
├── confident_errors.json
├── threshold_sweep.json
├── threshold_cost_selection.json
└── threshold_recall_floor.json
```

## Project layout

```text
src/fine_tuning_llms/
├── config.py       training configuration
├── data.py         dataset loading and tokenization
├── model.py        tokenizer and LoRA model setup
├── train.py        Trainer configuration and checkpointing
├── evaluate.py     metrics and test reporting
├── inference.py    adapter loading and prediction
└── pipeline.py     command-line workflow
tests/test_pipeline.py
```
