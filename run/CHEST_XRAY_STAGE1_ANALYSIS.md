# Chest X-ray Stage-1 Robustness Analysis

**Generated:** 2026-05-16  
**Project:** `ultralytics` / MedDef2 ablation study  
**Dataset:** Chest X-ray  
**Purpose:** Summarize final stage-1 evaluation results for all six variants using the same reporting style as the TBCR analysis.

---

## 1. Evaluation scope

This report covers **stage-1** evaluations from:

`runs/classify/train_chest_xray_final_eval_v2/chest_xray/<variant>_small/stage1/robustness/robustness_results.json`

Variants included:

- `full_small`
- `no_def_small`
- `no_freq_small`
- `no_patch_small`
- `no_cbam_small`
- `baseline_small`

All results are complete (`n=878` samples per variant).

---

## 2. Clean accuracy and mean robustness summary

| Variant  |  Clean Acc | Mean Robust Acc |  Mean ASR |
| -------- | ---------: | --------------: | --------: |
| full     |     86.33% |          25.19% |    70.83% |
| no_def   |     82.00% |          20.59% |    74.90% |
| no_freq  | **90.32%** |          14.46% |    83.98% |
| no_patch |     85.99% |          23.95% |    72.15% |
| no_cbam  |     73.01% |      **73.01%** | **0.00%** |
| baseline |     79.73% |          48.08% |    39.70% |

### Quick ranking

- **Best clean accuracy:** `no_freq` (90.32%)
- **Best mean robustness (raw):** `no_cbam` (73.01%)
- **Best mean robustness (non-anomalous profile):** `baseline` (48.08%)

---

## 3. Per-attack robust accuracy (stage-1)

| Attack   |   full |  no_def | no_freq | no_patch | no_cbam | baseline |
| -------- | -----: | ------: | ------: | -------: | ------: | -------: |
| fgsm     | 30.98% |  33.60% |   1.03% |   15.49% |  73.01% |   59.91% |
| pgd      |  4.33% |   1.48% |   0.00% |    6.49% |  73.01% |   45.90% |
| bim      |  5.58% |   2.85% |   0.00% |    7.06% |  73.01% |   47.38% |
| mim      |  6.72% |   2.51% |   0.00% |    8.88% |  73.01% |   43.96% |
| cw       | 99.66% | 100.00% | 100.00% |   99.43% |  73.01% |   95.79% |
| deepfool | 14.35% |  19.93% |  11.85% |   14.01% |  73.01% |   23.23% |
| apgd     |  3.76% |   0.23% |   0.00% |    5.81% |  73.01% |   25.51% |
| square   | 36.10% |   4.10% |   2.85% |   34.40% |  73.01% |   42.94% |

---

## 4. Per-attack ASR (stage-1)

| Attack   |    full |  no_def | no_freq | no_patch | no_cbam | baseline |
| -------- | ------: | ------: | ------: | -------: | ------: | -------: |
| fgsm     |  64.12% |  59.03% |  98.87% |   81.99% |   0.00% |   24.86% |
| pgd      |  94.99% |  98.19% | 100.00% |   92.45% |   0.00% |   42.43% |
| bim      |  93.54% |  96.53% | 100.00% |   91.79% |   0.00% |   40.57% |
| mim      |  92.22% |  96.94% | 100.00% |   89.67% |   0.00% |   44.86% |
| cw       | -15.44% | -21.94% | -10.72% |  -15.63% |   0.00% |  -20.14% |
| deepfool |  83.38% |  75.69% |  86.89% |   83.71% |   0.00% |   70.86% |
| apgd     |  95.65% |  99.72% | 100.00% |   93.24% |   0.00% |   68.00% |
| square   |  58.18% |  95.00% |  96.85% |   60.00% |   0.00% |   46.14% |

---

## 5. Key observations

1. `no_freq` achieves the highest clean accuracy, but has the worst robustness profile against gradient attacks (PGD/BIM/MIM/APGD are near 0%).
2. `baseline` is clearly more robust than `full`, `no_def`, `no_freq`, and `no_patch` on most attacks.
3. `cw` appears unusually high across variants and should be treated cautiously as a known optimization/convergence-sensitive metric.
4. `no_cbam` has a flat profile (`clean == mean robust == 73.01`, ASR=0 for all attacks), which is atypical and should be interpreted as a likely evaluation/model-behavior anomaly rather than strong evidence of true robustness.

---

## 6. Attack difficulty snapshot

Average robust accuracy across all six variants (lower = harder):

1. **APGD** (~18.05%)
2. **PGD** (~21.03%)
3. **MIM** (~22.18%)
4. **BIM** (~22.98%)
5. DeepFool (~26.06%)
6. Square (~32.73%)
7. FGSM (~35.67%)
8. **CW** (~94.48%, likely inflated/unreliable)

---

## 7. Practical recommendation (Chest X-ray)

- If you prioritize **clean accuracy**, use `no_freq`.
- If you prioritize **robustness under standard gradient attacks**, `baseline` is the safest non-anomalous choice in this stage-1 set.
- Treat `no_cbam` and `cw` outcomes as requiring re-checks before making deployment claims.

---

## 8. Bottom line

> On Chest X-ray stage-1, the results show a strong clean-versus-robustness tradeoff: `no_freq` is best for clean accuracy (90.32%) but weakest under strong attacks, while `baseline` gives the most credible robustness among non-anomalous variants. `no_cbam` and CW behavior should be validated with deeper diagnostics before drawing scientific conclusions.
