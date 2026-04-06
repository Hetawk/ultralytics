# TBCR Final Integration & Robustness Summary

**Generated:** 2026-04-06  
**Project:** `ultralytics` / TBCR final MedDef2 ablation + distillation study  
**Purpose:** concise shareable summary for review and suggestions

---

## 1. Goal

The goal was to integrate the final MedDef2-based TBCR pipeline, run all key ablations, complete stage-wise and distillation-based evaluation, and determine whether the added defense-oriented components actually improve adversarial robustness.

---

## 2. Major integrations completed

### A. Model variants integrated and evaluated

We completed all six final TBCR variants:

- `full_small` — full model, all components active
- `no_def_small` — defense module removed
- `no_freq_small` — frequency branch removed
- `no_patch_small` — patch-defense component removed
- `no_cbam_small` — CBAM attention removed
- `baseline_small` — simplest baseline configuration

Each variant was evaluated in **two stages**:

- `stage1`
- `distill`

That produced **12 completed final evaluation runs**.

### B. Robustness evaluation pipeline

Integrated full robustness testing with:

- `fgsm`
- `pgd`
- `bim`
- `mim`
- `cw`
- `deepfool`
- `apgd`
- `square`

with epsilon sweeps saved under the TBCR final eval output tree.

### C. Multi-GPU execution and scheduler work

We also completed the engineering side needed to finish the study reliably:

- multi-GPU evaluation scheduling across 4× Tesla T4
- GPU state tracking, retry logic, cooldown logic
- safe scheduler restart for later faster evaluation passes
- increased later eval throughput with larger batch for pending runs

### D. Stability fixes applied during evaluation

A key runtime issue was discovered and fixed:

- some jobs were failing with **false CUDA OOMs** on otherwise-free GPUs due to a **stale asynchronous CUDA error state**
- this was addressed by adding `torch.cuda.synchronize()` near startup in `evaluate.py`
- a scheduler GPU-claim mismatch was also corrected so free GPUs were used properly

### E. Cleanup and organization

We also cleaned the workspace/server while preserving critical artifacts:

- kept final TBCR training/eval runs and the `train/tbcr/` baseline reference
- removed stale scripts, backup files, pycache, and old non-essential logs/runs

### F. Actual defense integrations used in the codebase

The final TBCR study used the MedDef2 ablation family defined by the training entrypoint:

| Variant    | YAML / config           | What it represents                                  |
| ---------- | ----------------------- | --------------------------------------------------- |
| `full`     | `meddef2.yaml`          | full MedDef2 with all main components enabled       |
| `no_def`   | `meddef2_no_def.yaml`   | defense module removed                              |
| `no_freq`  | `meddef2_no_freq.yaml`  | frequency-defense branch removed                    |
| `no_patch` | `meddef2_no_patch.yaml` | patch-consistency / patch-defense component removed |
| `no_cbam`  | `meddef2_no_cbam.yaml`  | CBAM attention removed                              |
| `baseline` | `meddef2_baseline.yaml` | simplest baseline configuration                     |

From `meddef2.yaml`, the full model explicitly integrates these architectural defense-related blocks:

- **Frequency defense** — `enabled: true`, `cutoff_ratio: 0.5`
- **Patch consistency defense** — `enabled: true`, `threshold: 1.0`, `smooth_factor: 0.5`
- **CBAM attention** — `enabled: true`, `reduction_ratio: 16`
- **Defensive distillation config** — `enabled: true`, `temperature: 4.0`, `alpha: 0.5`

In addition, the MedDef training pipeline applies **input-space defenses during training only** in `ultralytics/models/meddef/train.py`:

- `GaussianAugmentation(sigma=0.05)`
- `FeatureSqueezing(bit_depth=6)`

The repo also contains a recommended **inference-time preprocessing defense stack** in `InputTransformPipeline.recommended_inference()`:

- `FeatureSqueezing(bit_depth=5)`
- `JpegCompression(quality=75)`
- `SpatialSmoothing(window_size=3)`

Important evaluation note:

> For the final robustness study, validation/inference metrics were computed on the normal evaluation pipeline rather than artificially wrapping the classifier in an extra preprocessing-only defense stack. So the reported robustness reflects the trained model behavior itself, not an added inference trick.

### G. Exact hyperparameters and runtime settings used

#### 1) Stage-1 TBCR final training (`run/train_tbcr_final.sh`)

| Hyperparameter          |                                           Value |
| ----------------------- | ----------------------------------------------: |
| Dataset                 |                           `processed_data/tbcr` |
| Variants                | `full no_def no_freq no_patch no_cbam baseline` |
| Depth                   |                                         `small` |
| Epochs                  |                                           `160` |
| Batch size              |                                            `16` |
| Image size              |                                           `224` |
| Optimizer               |                                         `AdamW` |
| Initial LR (`lr0`)      |                                        `0.0008` |
| Final LR factor (`lrf`) |                                          `0.01` |
| Weight decay            |                                        `0.0005` |
| Warmup epochs           |                                           `3.0` |
| Warmup bias LR          |                                           `0.1` |
| Patience                |                                           `140` |
| Dropout                 |                                           `0.0` |
| Random erasing          |                                           `0.4` |
| MixUp                   |                                           `0.0` |
| CutMix                  |                                           `0.0` |
| Cosine LR               |                                         `false` |
| Save period             |                               every `10` epochs |
| Workers                 |                                             `8` |
| GPU pool                |                                       `0,1,2,3` |
| Launch threshold        |                            `MIN_MEMORY_MB=5000` |
| Scheduler tick          |                                     every `20s` |

#### 2) Stage-2 defensive distillation (`run/distill_tbcr_final.sh`)

| Hyperparameter          |                              Value |
| ----------------------- | ---------------------------------: |
| Epochs                  |                              `100` |
| Batch size              |                               `64` |
| Image size              |                              `224` |
| Optimizer               |                            `AdamW` |
| Initial LR (`lr0`)      |                           `0.0002` |
| Final LR factor (`lrf`) |                             `0.01` |
| Weight decay            |                           `0.0005` |
| Warmup epochs           |                              `1.0` |
| Warmup bias LR          |                              `0.1` |
| Patience                |                               `20` |
| Dropout                 |                              `0.0` |
| Random erasing          |                              `0.2` |
| MixUp                   |                              `0.0` |
| CutMix                  |                              `0.0` |
| Cosine LR               |                            `false` |
| Save period             |                   every `5` epochs |
| Workers                 |                                `8` |
| Student init            |    `--pretrained <stage1 best.pt>` |
| Teacher                 | `--teacher_model <stage1 best.pt>` |
| Distill flag            |                    `--def_distill` |
| Distill temperature     |                              `4.0` |
| Distill alpha           |                              `0.5` |

#### 3) Final adversarial evaluation (`run/eval_tbcr_final.sh`)

| Setting                  | Value                                                                                                  |
| ------------------------ | ------------------------------------------------------------------------------------------------------ |
| Attacks                  | `fgsm pgd bim mim cw deepfool apgd square`                                                             |
| Epsilons                 | `0.0 0.005 0.01 0.02 0.03 0.05 0.1 0.15 0.2 0.3`                                                       |
| Eval image size          | `224`                                                                                                  |
| Eval workers             | `4`                                                                                                    |
| Saliency samples         | `8`                                                                                                    |
| Retry policy             | `MAX_RETRIES=3`                                                                                        |
| GPU cooldown after crash | `90s`                                                                                                  |
| Distill guard            | `MIN_ACTIVE_DISTILL=3`                                                                                 |
| Eval batch               | initially `16`, later increased to `64` for the remaining queued jobs after the safe scheduler restart |

These are the actual settings that produced the final 12-run TBCR result set.

---

## 3. Verified final outcome

All **12/12 TBCR evaluation jobs completed**.

Final outputs are stored under:

```text
runs/classify/train_tbcr_final_eval/tbcr/<variant>_small/<stage>/
```

Each completed run contains at least:

- `metrics.json`
- `robustness/robustness_results.json`
- `robustness/robustness_summary.txt`

---

## 4. Final performance summary

### Ranked by **mean robust accuracy**

| Rank | Variant          | Stage     | Clean Acc (%) | Mean Robust Acc (%) | Mean ASR (%) |
| ---: | ---------------- | --------- | ------------: | ------------------: | -----------: |
|    1 | `baseline_small` | `stage1`  |         93.33 |           **24.43** |        73.82 |
|    2 | `baseline_small` | `distill` |         94.29 |           **24.29** |        74.24 |
|    3 | `no_patch_small` | `distill` |         95.95 |               21.61 |        77.48 |
|    4 | `no_cbam_small`  | `distill` |         95.00 |               21.40 |        77.47 |
|    5 | `full_small`     | `distill` |         95.71 |               21.37 |        77.67 |
|    6 | `no_def_small`   | `stage1`  |         94.05 |               21.31 |        77.34 |
|    7 | `no_patch_small` | `stage1`  |         95.95 |               21.25 |        77.85 |
|    8 | `no_freq_small`  | `distill` |     **97.86** |               20.77 |        78.77 |
|    9 | `no_freq_small`  | `stage1`  |         96.67 |               20.65 |        78.63 |
|   10 | `no_def_small`   | `distill` |         94.29 |               19.38 |        79.45 |
|   11 | `no_cbam_small`  | `stage1`  |         95.95 |               18.99 |        80.21 |
|   12 | `full_small`     | `stage1`  |         95.48 |           **15.77** |        83.48 |

---

## 5. Main findings

### A. Best clean model

The best clean-accuracy model was:

- **`no_freq_small/distill` = 97.86% clean accuracy**

### B. Best overall robustness

The strongest average robustness was unexpectedly achieved by:

- **`baseline_small/stage1` = 24.43% mean robust accuracy**
- `baseline_small/distill` was essentially tied at **24.29%**

### C. Best clean/robust tradeoff

Among stronger clean-performing models, the best compromise appears to be:

- **`no_patch_small/distill`**
- also competitive: **`full_small/distill`**

### D. Distillation helped more than the architectural defense stack

A notable positive result is that distillation improved several variants, especially the full model:

- `full_small/stage1`: **15.77%** mean robust accuracy
- `full_small/distill`: **21.37%** mean robust accuracy

So **distillation provided a meaningful robustness recovery**.

---

## 6. Most important scientific conclusion

The current defense-oriented architectural modifications **did improve clean classification performance**, but they **did not improve adversarial robustness enough to outperform the baseline**.

In fact, the simplest baseline model was the most robust on average.

### Practical interpretation

This suggests that the added components are currently acting more like:

- feature enhancers,
- clean-accuracy boosters,
- or dataset-specific discriminative modules,

rather than true adversarial-robustness mechanisms.

This is still a valid and useful result:

> **The proposed modifications improved utility on clean TBCR classification, but did not translate into stronger adversarial robustness under strong white-box attacks. Knowledge distillation partially recovered robustness, while the simplest baseline remained the hardest to fool on average.**

---

## 7. What worked vs. what did not

### What worked

- final TBCR training and distillation pipeline completed successfully
- all 12 robustness-eval runs completed successfully
- distillation consistently helped the more complex variants
- `no_freq_small/distill` achieved the best clean accuracy
- `no_patch_small/distill` and `full_small/distill` gave useful compromise models

### What did not work as expected

- the full defense stack did **not** become the most robust model
- baseline remained the strongest by average robustness
- `full_small/stage1` was actually the weakest in robust ranking

---

## 8. Questions to ask for external suggestions

These are the key review questions worth sharing with others:

1. **Why is the baseline more robust than the defended variants?**
2. Are the added modules increasing clean accuracy at the cost of more attack-sensitive gradients?
3. Is the defense stack helping only specific attacks but not the average overall metric?
4. Should the next step include true robustness-oriented training such as:
   - adversarial training,
   - TRADES,
   - MART,
   - gradient regularization,
   - logit smoothing / calibration,
   - stronger distillation objectives?
5. Should we perform a deeper **per-attack** analysis before changing the architecture again?

---

## 9. Recommended next steps

### Immediate

- produce a per-attack comparison table across all 12 completed runs
- compare `stage1` vs `distill` directly for each variant
- identify the best model under each attack family, not just mean robustness

### Likely model recommendation right now

Depending on the goal:

- **If clean accuracy matters most:** `no_freq_small/distill`
- **If average robustness matters most:** `baseline_small/stage1`
- **If balanced tradeoff matters most:** `no_patch_small/distill` or `full_small/distill`

### Next dataset priority

After TBCR, the next reasonable follow-up remains:

1. `MULTIC`
2. `CCTS`
3. `SCISIC`
4. `DERMNET`

---

## 10. Key artifact locations

### Final TBCR eval outputs

```text
/data2/enoch/ekd_coding_env/ultralytics/runs/classify/train_tbcr_final_eval/tbcr/
```

### Main scheduler / execution scripts

```text
ultralytics/run/eval_tbcr_final.sh
ultralytics/run/distill_tbcr_final.sh
ultralytics/run/train_tbcr_final.sh
```

### Existing broader analysis code in repo

```text
meddef/analyze_model_performance.py
meddef/comprehensive_robustness_analyzer.py
meddef/create_unified_dashboard.py
meddef/enhanced_robustness_dashboard.py
```

---

## Bottom line

The integration work is complete and successful from an engineering standpoint.  
The most important research result is that **clean-performance gains did not automatically yield robustness gains**. Distillation helped, but the baseline still ranked highest in robustness.

That is the core result to share for feedback.
