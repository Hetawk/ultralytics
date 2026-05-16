# SCISIC Stage-1 Robustness Analysis

**Generated:** 2026-05-16  
**Project:** `ultralytics` / MedDef2 ablation study  
**Dataset:** SCISIC  
**Purpose:** Summarize final stage-1 evaluation results for all six variants using the same reporting style as the TBCR analysis.

---

## 1. Evaluation scope

This report covers **stage-1** evaluations from:

`runs/classify/train_scisic_final_eval_v2/scisic/<variant>_small/stage1/robustness/robustness_results.json`

Variants included:

- `full_small`
- `no_def_small`
- `no_freq_small`
- `no_patch_small`
- `no_cbam_small`
- `baseline_small`

All results are complete (`n=235` samples per variant).

---

## 2. Clean accuracy and mean robustness summary

| Variant  |  Clean Acc | Mean Robust Acc |  Mean ASR |
| -------- | ---------: | --------------: | --------: |
| full     |     47.66% |          14.79% |    68.97% |
| no_def   |     20.43% |      **20.43%** | **0.00%** |
| no_freq  |     50.21% |          13.99% |    72.14% |
| no_patch | **51.06%** |          13.51% |    73.54% |
| no_cbam  |     47.66% |          12.82% |    73.10% |
| baseline |     46.38% |          17.71% |    61.81% |

### Quick ranking

- **Best clean accuracy:** `no_patch` (51.06%)
- **Best mean robustness (raw):** `no_def` (20.43%)
- **Best mean robustness (non-flat profile):** `baseline` (17.71%)

---

## 3. Per-attack robust accuracy (stage-1)

| Attack   |   full | no_def | no_freq | no_patch | no_cbam | baseline |
| -------- | -----: | -----: | ------: | -------: | ------: | -------: |
| fgsm     | 14.89% | 20.43% |  16.17% |   11.91% |   9.36% |   19.15% |
| pgd      |  3.40% | 20.43% |   2.13% |    0.43% |   0.85% |    8.09% |
| bim      |  5.11% | 20.43% |   2.55% |    1.28% |   0.85% |    7.66% |
| mim      |  3.83% | 20.43% |   2.98% |    0.85% |   0.85% |    8.09% |
| cw       | 69.36% | 20.43% |  68.94% |   73.19% |  71.49% |   73.19% |
| deepfool | 17.87% | 20.43% |  15.74% |   20.00% |  17.02% |   19.57% |
| apgd     |  2.13% | 20.43% |   1.28% |    0.00% |   0.85% |    4.26% |
| square   |  1.70% | 20.43% |   2.13% |    0.43% |   1.28% |    1.70% |

---

## 4. Per-attack ASR (stage-1)

| Attack   |    full | no_def | no_freq | no_patch | no_cbam | baseline |
| -------- | ------: | -----: | ------: | -------: | ------: | -------: |
| fgsm     |  68.75% |  0.00% |  67.80% |   76.67% |  80.36% |   58.72% |
| pgd      |  92.86% |  0.00% |  95.76% |   99.17% |  98.21% |   82.57% |
| bim      |  89.29% |  0.00% |  94.92% |   97.50% |  98.21% |   83.49% |
| mim      |  91.96% |  0.00% |  94.07% |   98.33% |  98.21% |   82.57% |
| cw       | -45.54% |  0.00% | -37.29% |  -43.33% | -50.00% |  -57.80% |
| deepfool |  62.50% |  0.00% |  68.64% |   60.83% |  64.29% |   57.80% |
| apgd     |  95.54% |  0.00% |  97.46% |  100.00% |  98.21% |   90.83% |
| square   |  96.43% |  0.00% |  95.76% |   99.17% |  97.32% |   96.33% |

---

## 5. Key observations

1. SCISIC is clearly harder than TBCR/Chest-Xray in this stage-1 setup: clean accuracy is low across all variants (20.43% to 51.06%).
2. `no_patch` gives the best clean accuracy (51.06%), but weak adversarial robustness (13.51% mean robust).
3. `baseline` has the best non-flat robustness profile (17.71% mean robust) with moderate clean accuracy (46.38%).
4. `no_def` is a flat profile (`clean == mean robust == 20.43`, ASR=0 for all attacks), which is atypical and should be considered a likely evaluation/model-behavior anomaly.
5. CW remains high (or negative ASR) for all variants, consistent with known C&W instability under default parameterization.

---

## 6. Attack difficulty snapshot

Average robust accuracy across all six variants (lower = harder):

1. **APGD** (~4.82%)
2. **Square** (~7.95%)
3. **PGD** (~9.22%)
4. **MIM** (~9.44%)
5. BIM (~9.64%)
6. FGSM (~15.32%)
7. DeepFool (~18.44%)
8. **CW** (~62.84%, likely inflated/unreliable)

---

## 7. Practical recommendation (SCISIC)

- If you prioritize **clean accuracy**, use `no_patch`.
- If you prioritize **robustness under standard gradient attacks**, `baseline` is the strongest non-flat option in this stage-1 set.
- Treat `no_def` and CW behavior as needing re-checks before using them for final scientific claims.

---

## 8. Bottom line

> On SCISIC stage-1, all variants remain challenging in clean performance, and robustness under strong attacks is generally low. `no_patch` leads clean accuracy, while `baseline` is the most credible robust choice among non-anomalous variants. As with other datasets, CW and flat-profile cases should be validated before drawing final conclusions.
