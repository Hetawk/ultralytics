#!/usr/bin/env python3
"""
MedDef-VISTA: Server-Side Statistical Analysis
================================================
Computes bootstrap confidence intervals and calibration reliability diagrams
from actual model predictions. Designed to run on the lab server (GPU 2 or 3).

Requires:
  - Model weights under: runs/classify/train_tbcr_final/<variant>/distill/weights/best.pt
    OR robustness_results.json files under:
         runs/classify/train_tbcr_final_eval/tbcr/<variant>_small/distill/
  - Python venv: /data2/enoch/.virtualenvs/meddef_final/bin/python
  - Dataset at:  processed_data/tbcr (or as configured)

Usage on server:
    cd ~/py/ultralytics
    /data2/enoch/.virtualenvs/meddef_final/bin/python \
        generate_meddef_vista_server_analysis.py \
        --out out/meddef_vista_figures \
        --n_bootstrap 1000

Author: Enoch Kwateh Dongbo
"""

from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import json
import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')

# ---------------------------------------------------------------------------
# Global style (match main figure script)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family':    'DejaVu Sans',
    'font.size':      11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'savefig.dpi':    300,
    'savefig.bbox':   'tight',
    'axes.grid':      True,
    'grid.alpha':     0.25,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VARIANTS = ['no_freq', 'full', 'no_patch', 'no_cbam', 'no_def', 'baseline']
LABELS_SHORT = {
    'no_freq':  'MedDef-VISTA',
    'full':     'Full',
    'no_patch': 'w/o Patch',
    'no_cbam':  'w/o CBAM',
    'no_def':   'w/o DefMod',
    'baseline': 'Baseline',
}
COLORS = {
    'no_freq':  '#1565C0',
    'full':     '#2E7D32',
    'no_patch': '#E65100',
    'no_cbam':  '#6A1B9A',
    'no_def':   '#C62828',
    'baseline': '#37474F',
}

# Paths (relative to script location on server)
RUNS_ROOT = 'runs/classify/train_tbcr_final_eval/tbcr'
METRICS_SUBDIR = 'distill'   # look in distill stage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_fig(fig, fname, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'  ✔ saved {path}')


def find_metrics_file(variant: str, stage: str = 'distill') -> str | None:
    """Try to find metrics.json for a given variant and stage."""
    candidate_dirs = [
        # New eval tree
        os.path.join(RUNS_ROOT, f'{variant}_small', stage),
        os.path.join(RUNS_ROOT, f'{variant}_small', f'{stage}_v2'),
        # Legacy training tree (evaluate.py writes metrics here)
        f'runs/classify/train_tbcr_final/{variant}_small/{stage}',
    ]
    for d in candidate_dirs:
        p = os.path.join(d, 'metrics.json')
        if os.path.isfile(p):
            return p
    return None


def find_robustness_file(variant: str, stage: str = 'distill') -> str | None:
    """Try to find robustness_results.json for a given variant and stage."""
    candidate_dirs = [
        os.path.join(RUNS_ROOT, f'{variant}_small', stage, 'robustness'),
        os.path.join(RUNS_ROOT, f'{variant}_small',
                     f'{stage}_v2', 'robustness'),
        f'runs/classify/train_tbcr_final/{variant}_small/{stage}/robustness',
    ]
    for d in candidate_dirs:
        p = os.path.join(d, 'robustness_results.json')
        if os.path.isfile(p):
            return p
    return None


# ---------------------------------------------------------------------------
# Bootstrap CI computation
# ---------------------------------------------------------------------------

def bootstrap_ci(y_true: np.ndarray, y_pred_class: np.ndarray,
                 y_prob: np.ndarray, n_bootstrap: int = 1000,
                 ci_level: float = 0.95, seed: int = 42) -> dict:
    """
    Compute bootstrap confidence intervals for key metrics.
    Returns dict: {metric_name: {'mean': ..., 'ci_lo': ..., 'ci_hi': ...}}
    """
    try:
        from sklearn.metrics import (accuracy_score, f1_score,
                                     roc_auc_score, matthews_corrcoef)
    except ImportError:
        print('  ⚠  sklearn not available, skipping bootstrap.')
        return {}

    rng = np.random.default_rng(seed)
    n = len(y_true)

    accs, f1s, aucs, mccs, sens = [], [], [], [], []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred_class[idx]
        ypr = y_prob[idx]

        # Skip if only one class in resample (rare)
        if len(np.unique(yt)) < 2:
            continue

        accs.append(accuracy_score(yt, yp) * 100)
        f1s.append(f1_score(yt, yp, average='macro') * 100)
        try:
            aucs.append(roc_auc_score(yt, ypr) * 100)
        except Exception:
            pass
        mccs.append(matthews_corrcoef(yt, yp) * 100)
        # TB sensitivity: recall for class 1
        tp = np.sum((yt == 1) & (yp == 1))
        fn = np.sum((yt == 1) & (yp == 0))
        if (tp + fn) > 0:
            sens.append(tp / (tp + fn) * 100)

    alpha = (1 - ci_level) / 2
    result = {}
    for name, arr in [('accuracy', accs), ('f1_macro', f1s),
                      ('roc_auc', aucs), ('mcc', mccs),
                      ('tb_sensitivity', sens)]:
        if len(arr) < 2:
            continue
        arr = np.array(arr)
        result[name] = {
            'mean':   float(arr.mean()),
            'std':    float(arr.std()),
            'ci_lo':  float(np.percentile(arr, alpha * 100)),
            'ci_hi':  float(np.percentile(arr, (1 - alpha) * 100)),
            'n_boot': len(arr),
        }
    return result


# ---------------------------------------------------------------------------
# Calibration reliability diagram
# ---------------------------------------------------------------------------

def reliability_diagram(y_true: np.ndarray, y_prob: np.ndarray,
                        n_bins: int = 10) -> tuple:
    """Return (bin_midpoints, fraction_of_positives, mean_predicted_prob, ece)."""
    bins = np.linspace(0, 1, n_bins + 1)
    fracs, preds, counts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        fracs.append(float(y_true[mask].mean()))
        preds.append(float(y_prob[mask].mean()))
        counts.append(int(mask.sum()))

    fracs = np.array(fracs)
    preds = np.array(preds)
    counts = np.array(counts)
    n_total = counts.sum()
    ece = float(np.sum(counts / n_total * np.abs(fracs - preds)))
    return preds, fracs, counts, ece


def fig_reliability_diagrams(variant_data: dict, out_dir: str):
    """Plot reliability diagrams for all variants in a grid."""
    n_vars = len(VARIANTS)
    ncols = 3
    nrows = (n_vars + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 4.5))
    axes = axes.flatten()

    for idx, v in enumerate(VARIANTS):
        ax = axes[idx]
        if v not in variant_data:
            ax.text(0.5, 0.5, 'Data not available', ha='center',
                    va='center', transform=ax.transAxes, color='grey')
            ax.set_title(LABELS_SHORT[v])
            continue

        y_true = variant_data[v]['y_true']
        y_prob = variant_data[v]['y_prob']
        preds, fracs, counts, ece = reliability_diagram(y_true, y_prob)

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', lw=1.0, alpha=0.6, label='Perfect')

        # Reliability line
        ax.plot(preds, fracs, marker='o', ms=6, lw=2.0,
                color=COLORS[v], label=f'ECE={ece*100:.2f}%')

        # Gap shading (over/under confidence)
        for p, f in zip(preds, fracs):
            ax.fill_between([p - 0.005, p + 0.005], [p - 0.005, p + 0.005],
                            [f - 0.005, f + 0.005],
                            alpha=0.3, color='red' if f < p else 'blue')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Mean Predicted Probability', fontsize=10)
        ax.set_ylabel('Fraction of Positives', fontsize=10)
        ax.set_title(f'({chr(65+idx)})  {LABELS_SHORT[v]}', fontsize=11)
        ax.legend(fontsize=9)
        ax.set_aspect('equal')

    # Hide unused axes
    for idx in range(n_vars, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('MedDef-VISTA: Calibration Reliability Diagrams\n'
                 '(Perfect calibration = diagonal; gap = over/under confidence)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, 'fig_reliability_diagrams.png', out_dir)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Bootstrap CI forest plot
# ---------------------------------------------------------------------------

def fig_bootstrap_ci(all_ci: dict, out_dir: str):
    """Forest plot of bootstrap 95% CI for each variant and metric."""
    metrics_to_show = ['accuracy', 'tb_sensitivity',
                       'f1_macro', 'mcc', 'roc_auc']
    metric_labels = {
        'accuracy':       'Accuracy (%)',
        'tb_sensitivity': 'TB Sensitivity (%)',
        'f1_macro':       'F1-Macro (%)',
        'mcc':            'MCC (×100)',
        'roc_auc':        'ROC-AUC (%)',
    }

    n_metrics = len(metrics_to_show)
    fig, axes = plt.subplots(1, n_metrics, figsize=(16, 5.5),
                             sharey=True)

    y_positions = np.arange(len(VARIANTS))

    for ax, metric in zip(axes, metrics_to_show):
        for yi, v in enumerate(VARIANTS):
            if v not in all_ci or metric not in all_ci[v]:
                continue
            ci = all_ci[v][metric]
            mean = ci['mean']
            ci_lo = ci['ci_lo']
            ci_hi = ci['ci_hi']

            # Horizontal CI bar
            ax.plot([ci_lo, ci_hi], [yi, yi], '-', color=COLORS[v], lw=3.5,
                    alpha=0.7, solid_capstyle='round')
            # Mean dot
            ax.plot(mean, yi, 'o', color=COLORS[v], ms=8, zorder=5)
            # Value label
            ax.text(ci_hi + 0.1, yi, f'{mean:.2f}', va='center',
                    fontsize=8, color=COLORS[v])

        ax.set_yticks(y_positions)
        ax.set_yticklabels([LABELS_SHORT[v] for v in VARIANTS], fontsize=9)
        ax.set_title(metric_labels.get(metric, metric), fontsize=10)
        ax.invert_yaxis()
        ax.axvline(0, color='black', lw=0.5, alpha=0.3)

    fig.suptitle('MedDef-VISTA: Bootstrap 95% Confidence Intervals\n'
                 '(1000 resamples of test set, n=420)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, 'fig_bootstrap_ci.png', out_dir)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Computational complexity analysis
# ---------------------------------------------------------------------------

def compute_model_complexity(out_dir: str, variant: str = 'no_freq',
                             img_size: int = 224):
    """
    Compute parameter count, approximate FLOPs, and inference time.
    Requires the model and thop/fvcore library.
    """
    import time
    import torch

    try:
        from thop import profile as thop_profile
        has_thop = True
    except ImportError:
        has_thop = False
        print('  ⚠  thop not available; FLOPs will not be computed.')

    from ultralytics import YOLO

    VARIANT_WEIGHTS = {
        'no_freq':  f'runs/classify/train_tbcr_final/{variant}_small/distill/weights/best.pt',
        'full':     'runs/classify/train_tbcr_final/full_small/distill/weights/best.pt',
        'baseline': 'runs/classify/train_tbcr_final/baseline_small/distill/weights/best.pt',
    }

    results = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dummy = torch.randn(1, 3, img_size, img_size).to(device)

    for v_name, weights_path in VARIANT_WEIGHTS.items():
        if not os.path.isfile(weights_path):
            print(f'  ⚠  weights not found: {weights_path}')
            continue
        try:
            model = YOLO(weights_path)
            model.model.eval().to(device)

            # Parameter count
            n_params = sum(p.numel() for p in model.model.parameters()) / 1e6

            # FLOPs
            flops = None
            if has_thop:
                try:
                    macs, _ = thop_profile(
                        model.model, inputs=(dummy,), verbose=False)
                    flops = macs * 2 / 1e9  # GFLOPs
                except Exception:
                    pass

            # Inference time (50 warm-up + 200 timed)
            with torch.no_grad():
                for _ in range(50):
                    _ = model.model(dummy)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(200):
                    _ = model.model(dummy)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed_ms = (time.perf_counter() - t0) / 200 * 1000

            results[v_name] = {
                'params_M':   round(n_params, 2),
                'GFLOPs':     round(flops, 2) if flops is not None else 'N/A',
                'infer_ms':   round(elapsed_ms, 2),
                'device':     device,
            }
            print(f'  {v_name}: {n_params:.2f}M params, '
                  f'{flops:.2f if flops else "?"}GFLOPs, {elapsed_ms:.2f}ms')
        except Exception as e:
            print(f'  ⚠  Error processing {v_name}: {e}')

    # Save JSON
    comp_path = os.path.join(out_dir, 'computational_complexity.json')
    with open(comp_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'  ✔ saved {comp_path}')

    return results


def fig_complexity_table(complexity: dict, out_dir: str):
    """Render computational complexity as a table figure."""
    rows = []
    for v in ['baseline', 'no_freq', 'full']:
        if v not in complexity:
            continue
        c = complexity[v]
        rows.append([LABELS_SHORT[v], f"{c['params_M']}M",
                     str(c['GFLOPs']), f"{c['infer_ms']} ms"])

    if not rows:
        print('  ⚠  No complexity data to render.')
        return

    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.axis('off')
    col_labels = ['Model', 'Params',
                  'GFLOPs (224×224)', 'Inference\n(single img, GPU)']
    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.auto_set_column_width(list(range(len(col_labels))))

    # Header styling
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor('#1565C0')
        tbl[(0, j)].set_text_props(color='white', fontweight='bold')

    # Highlight proposed row
    for row_idx, (v, row) in enumerate(zip(['baseline', 'no_freq', 'full'], rows)):
        if v == 'no_freq':
            for j in range(len(col_labels)):
                tbl[(row_idx + 1, j)].set_facecolor('#E3F2FD')

    ax.set_title('Computational Complexity Comparison',
                 fontsize=12, pad=8, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, 'fig_complexity_table.png', out_dir)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Load prediction data from metrics.json / robustness_results.json
# ---------------------------------------------------------------------------

def load_variant_predictions(variant: str) -> dict | None:
    """
    Try to load y_true and y_prob from any available result file.
    Returns None if not found.
    """
    # First try: metrics.json may have raw predictions if we saved them
    m_path = find_metrics_file(variant)
    if m_path:
        with open(m_path) as f:
            m = json.load(f)
        # If the file has per-sample predictions stored:
        if 'y_true' in m and 'y_prob' in m:
            return {
                'y_true': np.array(m['y_true']),
                'y_prob': np.array(m['y_prob']),
                'y_pred': np.array(m.get('y_pred',
                                         (np.array(m['y_prob']) >= 0.5).astype(int))),
            }

    # Second try: robustness_results.json (has clean-eval predictions)
    r_path = find_robustness_file(variant)
    if r_path:
        with open(r_path) as f:
            r = json.load(f)
        if 'clean_predictions' in r:
            cp = r['clean_predictions']
            return {
                'y_true': np.array(cp['labels']),
                'y_prob': np.array(cp.get('probs',
                                          [float(p[1]) for p in cp.get('probs_all', [])])),
                'y_pred': np.array(cp['preds']),
            }

    return None


def run_inference_for_predictions(variant: str, device: str = 'cuda:2') -> dict | None:
    """
    Fall-back: run model inference on the validation set to get predictions.
    Requires ultralytics + dataset.
    """
    import torch
    from ultralytics import YOLO

    weights = f'runs/classify/train_tbcr_final/{variant}_small/distill/weights/best.pt'
    if not os.path.isfile(weights):
        weights = f'runs/classify/train_tbcr_final/{variant}_small/stage1/weights/best.pt'
    if not os.path.isfile(weights):
        print(f'  ⚠  Cannot find weights for {variant}')
        return None

    try:
        model = YOLO(weights)
        # Use the val split of the TBCR dataset
        results = model.val(data='dataset/tbcr.yaml', split='val',
                            batch=64, imgsz=224, device=device,
                            verbose=False, plots=False)

        # Extract predictions from results
        # ultralytics val returns confusion_matrix and per-class metrics
        # We need to re-run inference to get probabilities
        from ultralytics.data import build_classification_dataset
        from torch.utils.data import DataLoader

        dataset_root = 'processed_data/tbcr/val'
        if not os.path.isdir(dataset_root):
            print(f'  ⚠  Dataset not found at {dataset_root}')
            return None

        from torchvision import transforms, datasets
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        ds = datasets.ImageFolder(dataset_root, transform=transform)
        dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)

        all_probs, all_labels = [], []
        model.model.eval().to(device)
        with torch.no_grad():
            for imgs, labels in dl:
                imgs = imgs.to(device)
                logits = model.model(imgs)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.numpy())

        y_prob = np.vstack(all_probs)[:, 1]  # TB class probability
        y_pred = (y_prob >= 0.5).astype(int)
        y_true = np.concatenate(all_labels)

        return {'y_true': y_true, 'y_prob': y_prob, 'y_pred': y_pred}

    except Exception as e:
        print(f'  ⚠  Inference failed for {variant}: {e}')
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Server-side MedDef-VISTA statistical analysis')
    p.add_argument('--out', default='out/meddef_vista_figures',
                   help='Output directory')
    p.add_argument('--n_bootstrap', type=int, default=1000,
                   help='Bootstrap iterations (default: 1000)')
    p.add_argument('--device', default='cuda:3',
                   help='PyTorch device for inference (default: cuda:3)')
    p.add_argument('--skip_complexity', action='store_true',
                   help='Skip FLOPs/params computation (faster)')
    p.add_argument('--skip_inference', action='store_true',
                   help='Only use cached prediction files, no live inference')
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'  MedDef-VISTA Server Analysis')
    print(f'  Output: {os.path.abspath(out_dir)}')
    print(f'  Bootstrap n={args.n_bootstrap}, device={args.device}')
    print(f'{"="*60}\n')

    # ---- 1. Load / compute predictions ----
    print('[1/4]  Loading variant predictions ...')
    variant_data = {}
    for v in VARIANTS:
        print(f'  [{v}] looking for cached predictions ...')
        data = load_variant_predictions(v)
        if data is None and not args.skip_inference:
            print(f'  [{v}] no cache found; running live inference ...')
            data = run_inference_for_predictions(v, device=args.device)
        if data is not None:
            variant_data[v] = data
            print(f'  [{v}] ✔ n={len(data["y_true"])} samples, '
                  f'TB cases={int(data["y_true"].sum())}')
        else:
            print(f'  [{v}] ✗ skipped')

    # ---- 2. Bootstrap CI ----
    print(f'\n[2/4]  Computing bootstrap CIs (n={args.n_bootstrap}) ...')
    all_ci = {}
    for v, data in variant_data.items():
        print(f'  [{v}] bootstrapping ...')
        ci = bootstrap_ci(
            data['y_true'], data['y_pred'], data['y_prob'],
            n_bootstrap=args.n_bootstrap,
        )
        all_ci[v] = ci
        if 'accuracy' in ci:
            a = ci['accuracy']
            print(f'  [{v}] accuracy: {a["mean"]:.2f}% '
                  f'[{a["ci_lo"]:.2f}%, {a["ci_hi"]:.2f}%]')

    # Save CI data
    ci_path = os.path.join(out_dir, 'bootstrap_ci.json')
    with open(ci_path, 'w') as f:
        json.dump(all_ci, f, indent=2)
    print(f'  ✔ saved {ci_path}')

    if all_ci:
        fig_bootstrap_ci(all_ci, out_dir)

    # ---- 3. Reliability diagrams ----
    print('\n[3/4]  Generating calibration reliability diagrams ...')
    if variant_data:
        fig_reliability_diagrams(variant_data, out_dir)
    else:
        print('  ⚠  No prediction data available, skipping reliability diagrams.')

    # ---- 4. Computational complexity ----
    if not args.skip_complexity:
        print('\n[4/4]  Computing model complexity ...')
        complexity = compute_model_complexity(out_dir, img_size=224)
        if complexity:
            fig_complexity_table(complexity, out_dir)
    else:
        print('\n[4/4]  Skipping complexity (--skip_complexity set)')

    print(f'\n{"="*60}')
    print(f'  Done!  Files in {os.path.abspath(out_dir)}')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    main()
