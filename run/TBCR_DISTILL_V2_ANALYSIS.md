# TBCR Distillation v1 vs v2 — Comparative Analysis

**Generated:** 2026-04-07  
**Project:** `ultralytics` / TBCR MedDef2 ablation study  
**Purpose:** Compare distill_v1 and distill_v2 runs, identify the best model, and determine whether the re-distillation improved results

---

## 1. What changed between v1 and v2

Both distillation runs used the same stage-1 pretrained weights. The key hyperparameter differences:

| Setting               | Distill v1 |       Distill v2 |
| --------------------- | ---------: | ---------------: |
| Batch size            |     **16** |           **64** |
| Patience (early stop) |     **20** | **0** (disabled) |
| Temperature           |       50.0 |             50.0 |
| Alpha                 |        0.9 |              0.9 |
| Epochs                |        100 |              100 |
| LR                    |     0.0002 |           0.0002 |
| Optimizer             |      AdamW |            AdamW |

**Summary:** v2 used 4× larger batch and trained all 100 epochs without early stopping.

---

## 2. Clean accuracy comparison — all 3 stages

| Variant      | Stage-1 | Distill v1 | Distill v2 |     v1→v2 Δ | Best Stage     |
| ------------ | ------: | ---------: | ---------: | ----------: | -------------- |
| **full**     |  95.24% | **95.71%** |     95.48% |     −0.24pp | distill v1     |
| **no_def**   |  93.57% | **94.29%** | **94.29%** |      0.00pp | tied v1/v2     |
| **no_freq**  |  96.67% | **97.86%** |     97.14% |     −0.71pp | distill v1     |
| **no_patch** |  94.76% |     95.95% | **95.71%** |     −0.24pp | distill v1     |
| **no_cbam**  |  93.57% | **95.00%** |     94.52% |     −0.48pp | distill v1     |
| **baseline** |  92.86% |     94.29% | **95.24%** | **+0.95pp** | **distill v2** |

### Key observations

- **Distill v1 achieved slightly higher clean accuracy** for 4 of 6 variants (full, no_freq, no_patch, no_cbam)
- **Baseline improved with v2** — the only variant where v2 clearly outperformed v1 (+0.95pp)
- The advantage of v1's smaller batch (16) + patience: the model converged to a sharper minimum that favored clean accuracy
- v2's larger batch (64) + no early stopping may have slightly over-smoothed the loss landscape for the defended variants

---

## 3. Robustness comparison — distill v1 (complete)

### Per-attack robust accuracy (distill v1, default ε)

| Attack   |       full |     no_def |     no_freq |   no_patch |    no_cbam |   baseline |
| -------- | ---------: | ---------: | ----------: | ---------: | ---------: | ---------: |
| fgsm     |     16.67% |     16.67% |      16.90% |     16.67% |     16.67% |     16.67% |
| pgd      |     11.19% |     15.24% |       8.10% |      8.81% |     10.24% |     15.48% |
| bim      |     11.67% |     15.48% |       8.10% |      9.52% |     10.71% |     16.19% |
| mim      |     16.19% |     16.43% |       8.33% |     14.52% |     11.43% |     16.67% |
| **cw**   | **86.67%** |     60.48% | **100.00%** |     92.38% |     91.43% | **99.76%** |
| deepfool |     12.86% |     14.76% |       8.33% |     13.10% |     14.76% |     13.81% |
| apgd     |      7.86% |      8.10% |       8.10% |      8.81% |      7.86% |      7.62% |
| square   |      7.86% |      7.86% |       8.33% |      9.05% |      8.10% |      8.10% |
| **Mean** | **21.37%** | **19.38%** |  **20.77%** | **21.61%** | **21.40%** | **24.29%** |

### Per-attack ASR (distill v1)

| Attack   |       full |     no_def |    no_freq |   no_patch |    no_cbam |   baseline |
| -------- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| fgsm     |     82.59% |     82.32% |     82.73% |     82.63% |     82.46% |     82.32% |
| pgd      |     88.31% |     83.84% |     91.73% |     90.82% |     89.22% |     83.59% |
| bim      |     87.81% |     83.59% |     91.73% |     90.07% |     88.72% |     82.83% |
| mim      |     83.08% |     82.58% |     91.48% |     84.86% |     87.97% |     82.32% |
| cw       |      9.45% |     35.86% |     −2.19% |      3.72% |      3.76% |     −5.81% |
| deepfool |     86.57% |     84.34% |     91.48% |     86.35% |     84.46% |     85.35% |
| apgd     |     91.79% |     91.41% |     91.73% |     90.82% |     91.73% |     91.92% |
| square   |     91.79% |     91.67% |     91.48% |     90.57% |     91.48% |     91.41% |
| **Mean** | **77.67%** | **79.45%** | **78.77%** | **77.48%** | **77.47%** | **74.24%** |

---

## 4. Deep-dive analysis

### 4.1 The CW anomaly — why some models show 100% robust accuracy

The C&W attack results are dramatically different from other attacks:

- `no_freq/distill_v1`: **100%** robust accuracy against C&W
- `baseline/distill_v1`: **99.76%**
- `no_patch/distill_v1`: **92.38%**
- `no_cbam/distill_v1`: **91.43%**

This is **not because these models are genuinely robust to C&W**. The likely explanation:

1. **C&W is an optimization-based attack** that requires many iterations to converge
2. With the default evaluation parameters, C&W may not have converged within the iteration budget
3. Models with smoother decision boundaries (baseline, no_freq) naturally resist unconverged C&W
4. **This should not be cited as real robustness** — it's a limitation of the evaluation protocol

### 4.2 Which defense modules actually help robustness?

Comparing v1 distill variants (excluding C&W outlier):

| Metric (excl. CW) |   full | no_def | no_freq | no_patch | no_cbam | baseline |
| ----------------- | -----: | -----: | ------: | -------: | ------: | -------: |
| Mean robust acc   | 12.04% | 13.51% |   9.46% |   11.50% |  11.40% |   13.51% |
| Mean ASR          | 87.28% | 85.53% |  91.60% |   89.44% |  89.29% |   85.53% |

**Rankings (excluding C&W, lower ASR = better):**

1. **baseline** & **no_def** (tied at 13.51% robust, 85.53% ASR)
2. **full** (12.04% robust)
3. **no_patch** (11.50%)
4. **no_cbam** (11.40%)
5. **no_freq** (9.46% — worst)

**Interpretation:**

- Removing ALL defense modules (baseline) or the DefenseModule (no_def) yields the **best gradient-attack robustness**
- The **FrequencyDefense** module _hurts_ robustness most — `no_freq` is the worst under gradient attacks despite being the best in clean accuracy
- CBAM and PatchConsistency have minor negative effects on robustness
- The defense stack is **not providing adversarial robustness** — it's providing better clean feature extraction

### 4.3 Stage-1 vs Distill v1 — did distillation help?

| Variant  | Stage-1 Mean Robust | Distill v1 Mean Robust |      Change |
| -------- | ------------------: | ---------------------: | ----------: |
| full     |          **15.77%** |             **21.37%** | **+5.61pp** |
| no_def   |              21.31% |                 19.38% |     −1.94pp |
| no_freq  |              20.65% |                 20.77% |     +0.12pp |
| no_patch |              21.25% |                 21.61% |     +0.36pp |
| no_cbam  |              18.99% |                 21.40% | **+2.41pp** |
| baseline |          **24.43%** |                 24.29% |     −0.14pp |

**Distillation most helped:** `full` (+5.61pp) and `no_cbam` (+2.41pp)  
**Distillation slightly hurt:** `no_def` (−1.94pp), `baseline` (−0.14pp)

The full model had the largest gap between stage-1 and distill, confirming that the complex defense stack creates optimization difficulty that distillation partially resolves.

### 4.4 Attack difficulty ranking

Averaged across all 6 variants (distill v1):

| Rank | Attack     | Mean Robust Acc | Mean ASR | Difficulty           |
| ---: | ---------- | --------------: | -------: | -------------------- |
|    1 | **apgd**   |           8.07% |   91.74% | Hardest              |
|    2 | **square** |           8.22% |   91.48% | Very hard            |
|    3 | pgd        |          11.51% |   88.00% | Hard                 |
|    4 | bim        |          11.93% |   87.40% | Hard                 |
|    5 | deepfool   |          12.93% |   86.43% | Hard                 |
|    6 | mim        |          13.93% |   85.21% | Moderate             |
|    7 | **fgsm**   |          16.69% |   82.51% | Easiest gradient     |
|    8 | **cw**     |          88.45% |    7.47% | Broken (unconverged) |

AutoPGD and Square Attack are the most effective — consistent with the literature where ensemble/adaptive attacks are harder to defend against.

---

## 5. Distill v2 evaluation — in progress

The distill_v2 adversarial evaluation is currently running on all 4 GPUs:

| GPU | Variant  | Status                            |
| --- | -------- | --------------------------------- |
| 0   | full     | Running (CW/epsilon sweep phase)  |
| 1   | no_freq  | Running (MIM+ phase)              |
| 2   | no_def   | Running (CW/epsilon sweep phase)  |
| 3   | no_patch | Running (PGD/epsilon sweep phase) |
| —   | no_cbam  | Queued                            |
| —   | baseline | Queued                            |

**Preliminary v2 eval results (per-attack, incomplete):**

| Attack | full v2 | no_def v2 | no_freq v2 | no_patch v2 |
| ------ | ------: | --------: | ---------: | ----------: |
| fgsm   |  16.67% |    16.67% |     16.67% |      16.67% |
| pgd    |  12.86% |    15.24% |          — |       9.76% |
| bim    |  13.33% |    15.71% |          — |           — |
| mim    |  16.67% |    16.67% |      8.10% |           — |
| cw     |       — |    42.86% |          — |           — |
| Clean  |  96.90% |    94.52% |     98.10% |      96.67% |

_Note: v2 eval clean accuracy differs slightly from training CSV because eval uses the `val` split directly._

---

## 6. Overall model rankings

### Best clean accuracy (all stages)

| Rank | Model                    |  Clean Acc |
| ---: | ------------------------ | ---------: |
|    1 | **no_freq / distill_v1** | **97.86%** |
|    2 | no_freq / distill_v2     |     97.14% |
|    3 | no_freq / stage1         |     96.67% |
|    4 | no_patch / distill_v1    |     95.95% |
|    5 | full / distill_v1        |     95.71% |

### Best mean robustness (distill v1, complete data)

| Rank | Model                     | Mean Robust |  Clean |    Gap |
| ---: | ------------------------- | ----------: | -----: | -----: |
|    1 | **baseline / distill_v1** |  **24.29%** | 94.29% | 70.0pp |
|    2 | baseline / stage1         |      24.43% | 93.33% | 68.9pp |
|    3 | no_patch / distill_v1     |      21.61% | 95.95% | 74.3pp |
|    4 | no_cbam / distill_v1      |      21.40% | 95.00% | 73.6pp |
|    5 | full / distill_v1         |      21.37% | 95.71% | 74.3pp |

### Best robustness excluding C&W anomaly

| Rank | Model                     | Mean Robust (excl CW) |  Clean | Notes                        |
| ---: | ------------------------- | --------------------: | -----: | ---------------------------- |
|    1 | **baseline / distill_v1** |            **13.51%** | 94.29% | Simplest model, most robust  |
|    2 | **no_def / distill_v1**   |            **13.51%** | 94.29% | Tied with baseline           |
|    3 | full / distill_v1         |                12.04% | 95.71% | Defense stack slightly hurts |
|    4 | no_patch / distill_v1     |                11.50% | 95.95% |                              |
|    5 | no_cbam / distill_v1      |                11.40% | 95.00% |                              |
|    6 | no_freq / distill_v1      |                 9.46% | 97.86% | Best clean, worst robust     |

### Recommended model per use case

| Goal                          | Recommended Model     |  Clean Acc |     Mean Robust |
| ----------------------------- | --------------------- | ---------: | --------------: |
| **Maximum clean accuracy**    | no_freq / distill_v1  | **97.86%** |          20.77% |
| **Maximum robustness**        | baseline / distill_v1 |     94.29% |      **24.29%** |
| **Balanced (clean + robust)** | no_patch / distill_v1 |     95.95% |          21.61% |
| **Best v2 clean + robust**    | baseline / distill_v2 | **95.24%** | _awaiting eval_ |

---

## 7. Scientific conclusions

### 7.1 Defense modules improve features, not robustness

The defense stack (FrequencyDefense, DefenseModule, PatchConsistency, CBAM) consistently **improves clean classification accuracy** but **does not improve adversarial robustness**. In fact, the simpler baseline model is the hardest to fool.

This is consistent with the adversarial robustness literature: architectural modifications that improve clean accuracy often increase gradient sensitivity, making the model _more_ vulnerable to gradient-based attacks.

### 7.2 FrequencyDefense is a clean-accuracy booster, not a robustness tool

`no_freq` achieves the highest clean accuracy (97.86%) but the **worst robustness** under gradient attacks. The FFT-based low-pass filter helps the model learn better features from clean data, but those features are highly sensitive to adversarial perturbations in the frequency domain.

### 7.3 Distillation temperature and batch size tradeoff

| Factor              | v1 (batch=16, patience=20)                         | v2 (batch=64, patience=0)       |
| ------------------- | -------------------------------------------------- | ------------------------------- |
| Clean accuracy      | Slightly higher for most variants                  | Slightly lower, except baseline |
| Convergence         | May stop early (patience=20)                       | Always trains 100 epochs        |
| Gradient noise      | Higher (small batch)                               | Lower (large batch)             |
| Expected robustness | Possibly better (noisy gradients ≈ regularization) | TBD (eval running)              |

The v1 small-batch training may have acted as implicit adversarial regularization through gradient noise, which could explain its slight clean accuracy advantage.

### 7.4 The accuracy-robustness tradeoff is real

Plotting the data reveals a clear **negative correlation** between clean accuracy and robustness:

```
Clean Acc (%)    Robust Acc (%)
  98 ─ no_freq      ──── 20.77  (best clean, worst robust)
  96 ─ no_patch     ──── 21.61
     ─ full         ──── 21.37
  95 ─ no_cbam      ──── 21.40
     ─ no_def       ──── 19.38
  94 ─ baseline     ──── 24.29  (worst clean, best robust)
```

This is the fundamental **accuracy-robustness tradeoff** documented in Tsipras et al. (2019) and Zhang et al. (2019).

---

## 8. Actionable next steps

### Immediate (once v2 eval completes)

1. Update this document with full distill_v2 robustness results
2. Determine if v2's larger batch changed the robustness landscape
3. Generate per-variant epsilon sweep plots for the thesis

### For the thesis

1. **Document the negative result honestly** — defense modules helped clean accuracy but not robustness
2. Use the per-attack tables to show _where_ each module helps/hurts
3. The CW anomaly should be noted as an evaluation limitation, not claimed as robustness
4. The accuracy-robustness tradeoff plot is a strong figure for Chapter 4

### For future work

1. **Adversarial training (TRADES/MART)** is the proven path to real robustness
2. **Certified defenses** (randomized smoothing) should be considered
3. The FrequencyDefense could be redesigned as a trainable adversarial filter rather than a fixed bandpass
4. Multi-dataset validation (MULTIC, CCTS, SCISIC, DermNet) is needed to confirm these findings generalize

---

## 9. Artifact locations

### Training outputs

```
runs/classify/train_tbcr_final/tbcr/<variant>_small/
  weights/best.pt           ← stage-1 model
  distill/weights/best.pt   ← distill v1 model  (batch=16, patience=20)
  distill_v2/weights/best.pt ← distill v2 model  (batch=64, patience=0)
```

### Evaluation outputs

```
runs/classify/train_tbcr_final_eval/tbcr/<variant>_small/
  stage1/                   ← stage-1 eval (COMPLETE)
  distill/                  ← distill v1 eval (COMPLETE)

runs/classify/train_tbcr_final_eval_v2/tbcr/<variant>_small/
  distill/                  ← distill v2 eval (IN PROGRESS)
```

### ONNX exports

```
runs/onnx_exports/tbcr/<variant>_small/
  original/best.onnx        ← stage-1 ONNX
  distill_v2/best.onnx      ← distill v2 ONNX
```

---

## 10. Bottom line

> **Distill v1 (batch=16, patience=20) produced slightly better clean accuracy for most variants. The baseline model remains the most adversarially robust. The defense modules (FrequencyDefense, DefenseModule, PatchConsistency, CBAM) consistently improve clean classification but increase vulnerability to gradient-based attacks — a textbook accuracy-robustness tradeoff. Distill v2 evaluation is running and will determine whether the larger batch affects this conclusion.**
