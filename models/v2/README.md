# Model v2 — Augmented + Balanced

## Overview

Improved version trained on an augmented and balanced dataset.  
Achieved **74% validation accuracy** — 9% improvement over v1.

## What Changed from v1

- Dataset augmented from 19,626 → 24,989 samples
- All 3 classes balanced to ~8,500 samples each
- Added weighted CrossEntropyLoss
- Added warmup scheduler
- Added gradient clipping (max norm 1.0)
- Weight decay added to AdamW optimizer

## Training Details

| Parameter | Value |
|---|---|
| Base Model | distilbert-base-multilingual-cased |
| Dataset | Augmented Roman Urdu Dataset |
| Total Samples | 24,989 |
| Training Samples | 19,991 |
| Validation Samples | 2,499 |
| Test Samples | 2,499 |
| Epochs | 5 |
| Learning Rate | 2e-5 |
| Batch Size | 16 |
| Class Weights | Yes |
| Warmup Scheduler | Yes |
| Gradient Clipping | 1.0 |

## Results

| Metric | Score |
|---|---|
| Validation Accuracy | 74.07% |
| Test Accuracy | 73% |

## Per Class Performance

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Positive | 0.75 | 0.77 | 0.76 |
| Negative | 0.75 | 0.73 | 0.74 |
| Neutral | 0.69 | 0.69 | 0.69 |

## Confusion Matrix

[[639  66 126]
[ 82 594 139]
[134 128 591]]

## Key Improvements Over v1

- Negative class F1: 0.55 → 0.74 (+0.19)
- Overall accuracy: 63% → 73% (+10%)
- Class imbalance successfully addressed
