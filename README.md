# ImageCLEFmedical Caption 2026 — Concept Detection Task

This repository contains the codebase developed by the **Archimedes Unit** (a collaborative effort between the AUEB NLP Group and NKUA) for the 10th edition of the **ImageCLEFmedical Caption 2026 Concept Detection task**.

Our proposed system achieved an exceptional **2nd place overall** in the official competition standings, trailing the top spot by a thin margin of just 0.0009 on the primary evaluation metric.

---

## Repository Structure

The project is organized into distinct functional modules:

- **`Concept Detection/`**: Contains the core model architectures, multi-label binarization scripts, cross-seed benchmarking loops, ensembling logic, and our conformal prediction regressor.
- **`Exploratory Data Analysis/`**: Houses notebook investigating data distributions, right-skewed label density per instance, long-tail concept sparsity, and dataset performance overviews.

---

## Task Background & Dataset

The competition dataset is based on an extended version of the **ROCOv2 (Radiology Objects in Context Version 2)** dataset, containing radiology images mapped to Unified Medical Language System (UMLS) Concept Unique Identifiers (CUIs). The target space spans 2,646 distinct biomedical concepts representing imaging modalities, anatomical structures, and clinical findings.

---

## Methodology Overview

Our pipeline implements a highly robust collection of Convolutional Neural Network (CNN) image encoders paired with Feed-Forward Neural Network (FFNN) classification heads, augmented by strategic ensembling and statistical calibration:

### 1. Encoder Backbones & Training Diversity

- **Model Pool**: We fine-tuned a diverse set of ImageNet-pretrained backbones: `EfficientNet-B0`, `EfficientNet-V2-S`, `EfficientNet-V2-M`, `DenseNet-121`, `ConvNeXt-Tiny`, and `ResNet-50`.
- **Generalized Mean (GeM) Pooling**: Replaced standard spatial aggregation with a learnable GeM pooling layer, optimizing feature maps dynamically between average and max pooling behaviors.
- **Stochastic Regularization**: Each baseline configuration was trained independently across 4 distinct random initialization seeds (40, 41, 42, and 43) to minimize local minima convergence risks.

### 2. Advanced Ensemble Strategies

- **Dual Threshold Aggregation**: Individual binary votes are extracted per architecture and summed across the voting pool; a secondary ensemble-level threshold ($L$) is then applied to control final output stringency.
- **Soft Voting**: Continuous confidence vectors (raw post-sigmoid probabilities) are averaged across models prior to decision binarization, preserving latent predictive signals.
- **Monte-Carlo Cross-Validation (MCCV)**: A stratified 5-split sub-ensemble governed by a strict unanimous intersection consensus rule to ensure high-precision label predictions.

### 3. Bidirectional Conformal Rescue

- An auxiliary **Concept Count Regressor** was integrated with an inductive **Split Conformal Regression** framework calibrated on validation data to guarantee a valid prediction interval at a 90% confidence level.
- **Dynamic Correction**: If the base ensemble predicts fewer concepts than the lower bound, the system rescues the next highest-probability labels. If predictions exceed the upper bound, the lowest-confidence entries are dynamically pruned away.

---

## Official Evaluation Standings

| Run ID  | Method / Strategy Description                        | Primary $F_1$ (Test) | Secondary $F_1$ (Test) | Official Standing |
| :------ | :--------------------------------------------------- | :------------------: | :--------------------: | :---------------: |
| **983** | Dual-5 (With MCCV and per-member Conformal Rescue)   |      **0.5781**      |       **0.9591**       |   **2nd Place**   |
| **818** | Dual-5 (With per-member Conformal Rescue, no MCCV)   |        0.5780        |         0.9590         |     4th Place     |
| **984** | Dual-6 (With MCCV and per-member Conformal Rescue)   |        0.5776        |         0.9614         |     5th Place     |
| **931** | Dual-5 (Standard Ensemble, no Conformal Adjustments) |        0.5771        |         0.9574         |     7th Place     |
| **987** | Soft-Voting with post-ensemble Conformal Rescue      |        0.5764        |         0.9601         |     9th Place     |

_Note: The primary metric represents the sample-averaged F1 score calculated across the complete concept distribution, whereas the secondary metric restricts evaluation to a filtered subset of manual annotations like modality and anatomy._
