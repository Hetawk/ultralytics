#!/usr/bin/env python3
"""
MedDef-VISTA: Publication-Quality Figure Generation v2
=======================================================
Generates 3 publication figures for the CMIG paper.
No titles on figures — captions are written under figures in the document.
Consistent color palette across all figures.

Figures:
  fig2_radar_ablation.png       — Multi-Metric Ablation Radar Chart
  fig4_stage_improvement.png    — Stage-wise Distillation Improvement
  fig7_metric_comparison.png    — All-Variant Metric Comparison (all 6 variants)

Usage:
    python generate_meddef_vista_figures_v2.py [--out out/meddef_vista_figures]

Author: Enoch Kwateh Dongbo
"""

from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')

# ===========================================================================
# Global style
# ===========================================================================
plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'font.size':          11,
    'axes.labelsize':     12,
    'axes.titlesize':     12,
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'legend.fontsize':    9.5,
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'axes.grid':          True,
    'grid.alpha':         0.22,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.linewidth':     0.9,
    'xtick.direction':    'out',
    'ytick.direction':    'out',
})

# ===========================================================================
# Experiment data — sourced from lab results (TBCR, 420 samples)
# ===========================================================================

VARIANTS = ['no_freq', 'full', 'no_patch', 'no_cbam', 'no_def', 'baseline']

LABELS_SHORT = {
    'no_freq':  'MedDef-VISTA',
    'full':     'Full',
    'no_patch': 'w/o Patch',
    'no_cbam':  'w/o CBAM',
    'no_def':   'w/o DefMod',
    'baseline': 'Baseline',
}

# Consistent palette — matches fig2 radar colors exactly
COLORS = {
    'no_freq':  '#1565C0',   # deep blue   — proposed
    'full':     '#2E7D32',   # dark green  — full
    'no_patch': '#E65100',   # burnt orange
    'no_cbam':  '#6A1B9A',   # purple
    'no_def':   '#C62828',   # deep red
    'baseline': '#37474F',   # dark slate
}
MARKERS = {
    'no_freq': '*', 'full': 'D', 'no_patch': '^',
    'no_cbam': 's', 'no_def': 'v', 'baseline': 'o',
}

CLEAN_ACC = {
    'stage1':     {'no_freq': 96.67, 'full': 95.24, 'no_patch': 94.76,
                   'no_cbam': 93.57, 'no_def': 93.57, 'baseline': 92.86},
    'distill_v1': {'no_freq': 97.86, 'full': 95.71, 'no_patch': 95.95,
                   'no_cbam': 95.00, 'no_def': 94.29, 'baseline': 94.29},
    'distill_v2': {'no_freq': 98.10, 'full': 96.90, 'no_patch': 96.67,
                   'no_cbam': 95.71, 'no_def': 94.52, 'baseline': 95.48},
}
TB_SENS_V1 = {
    'no_freq': 88.57, 'full': 74.29, 'no_patch': 75.71,
    'no_cbam': 75.71, 'no_def': 65.71, 'baseline': 65.71,
}
MCC = {
    'no_freq': 0.9214, 'full': 0.8405, 'no_patch': 0.8497,
    'no_cbam': 0.8115, 'no_def': 0.7842, 'baseline': 0.7842,
}
F1_MACRO = {
    'no_freq': 0.9598, 'full': 0.9137, 'no_patch': 0.9190,
    'no_cbam': 0.9026, 'no_def': 0.8800, 'baseline': 0.8800,
}
ECE = {
    'no_patch': 0.00758, 'no_def': 0.01072, 'no_cbam': 0.02082,
    'no_freq':  0.02543, 'baseline': 0.03031, 'full': 0.03488,
}
ROC_AUC = {
    'baseline': 0.9946, 'no_freq': 0.9938, 'no_patch': 0.9804,
    'full': 0.9797, 'no_cbam': 0.9486, 'no_def': 0.9234,
}

STAGE_KEYS = ['stage1', 'distill_v1', 'distill_v2']
STAGE_LABELS = ['Stage-1\n(pre-training)',
                'Distill v1\n(batch=16)', 'Distill v2\n(batch=64)']


# ===========================================================================
# Helpers
# ===========================================================================

def save_fig(fig, fname, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'  v saved {path}')


def legend_handles(variants=None):
    if variants is None:
        variants = VARIANTS
    return [
        Line2D([0], [0], marker=MARKERS[v], color=COLORS[v],
               markersize=9, linestyle='None', label=LABELS_SHORT[v])
        for v in variants
    ]


# ===========================================================================
# Figure 2 — Multi-Metric Radar Chart
# No title. Caption goes under figure in the document.
# ===========================================================================

def fig_radar(out_dir):
    metrics_raw = {
        'Accuracy (%)':       CLEAN_ACC['distill_v1'],
        'TB Sensitivity (%)': TB_SENS_V1,
        'F1-Macro (%)':       {v: F1_MACRO[v] * 100 for v in VARIANTS},
        'MCC (x100)':         {v: MCC[v] * 100 for v in VARIANTS},
        'AUC (%)':            {v: ROC_AUC[v] * 100 for v in VARIANTS},
        'Calibration\n(1-ECE, %)': {v: (1 - ECE[v]) * 100 for v in VARIANTS},
    }
    categories = list(metrics_raw.keys())
    N = len(categories)

    metrics_norm = {}
    for cat, vals in metrics_raw.items():
        arr = np.array([vals[v] for v in VARIANTS])
        lo, hi = arr.min(), arr.max()
        metrics_norm[cat] = {
            v: (vals[v] - lo) / (hi - lo) if hi != lo else 0.5
            for v in VARIANTS
        }

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for v in VARIANTS:
        vals = [metrics_norm[cat][v] for cat in categories]
        vals += vals[:1]
        ax.plot(angles, vals, color=COLORS[v], linewidth=2.5,
                marker=MARKERS[v], ms=8, label=LABELS_SHORT[v], zorder=5)
        ax.fill(angles, vals, alpha=0.07, color=COLORS[v])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.1)

    for r in [0.25, 0.50, 0.75, 1.00]:
        ax.plot(angles, [r] * (N + 1), color='grey', lw=0.5, ls=':', zorder=1)
    ax.text(0, 1.06, '1.0\n(best)', fontsize=7.5,
            ha='center', va='bottom', color='grey')

    ax.legend(handles=legend_handles(), loc='upper right',
              bbox_to_anchor=(1.45, 1.15), framealpha=0.92, fontsize=10)

    plt.tight_layout()
    save_fig(fig, 'fig2_radar_ablation.png', out_dir)
    plt.close(fig)


# ===========================================================================
# Figure 4 — Stage-wise Distillation Improvement
# No title. Per-variant colors on both panels.
# ===========================================================================

def fig_stage_improvement(out_dir):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(STAGE_KEYS))

    for v in VARIANTS:
        ys = [CLEAN_ACC[s][v] for s in STAGE_KEYS]
        ax.plot(x, ys, marker=MARKERS[v], ms=9, lw=2.2,
                color=COLORS[v], label=LABELS_SHORT[v], zorder=5)
        ax.text(x[-1] + 0.07, ys[-1], f'{ys[-1]:.2f}%',
                fontsize=8.5, color=COLORS[v], va='center', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_LABELS, fontsize=10)
    ax.set_ylabel('Clean Accuracy (%)', fontsize=12)
    ax.legend(handles=legend_handles(), loc='lower right', fontsize=9.5,
              framealpha=0.92)
    ax.set_ylim(91.5, 99.5)
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.5))

    plt.tight_layout()
    save_fig(fig, 'fig4_stage_improvement.png', out_dir)
    plt.close(fig)


# ===========================================================================
# Figure 7 — All-Variant Metric Comparison
# All 6 variants across 6 metrics. No title.
# AUC inversion zone highlighted in orange.
# ===========================================================================

def fig_metric_comparison(out_dir):
    metrics = {
        'Accuracy (%)':        {v: CLEAN_ACC['distill_v1'][v] for v in VARIANTS},
        'TB Sensitivity (%)':  TB_SENS_V1,
        'F1-Macro (%)':        {v: F1_MACRO[v] * 100 for v in VARIANTS},
        'MCC (x100)':          {v: MCC[v] * 100 for v in VARIANTS},
        'AUC (%)':             {v: ROC_AUC[v] * 100 for v in VARIANTS},
        'Calibration\n(1-ECE, %)': {v: (1 - ECE[v]) * 100 for v in VARIANTS},
    }
    metric_names = list(metrics.keys())
    n_variants = len(VARIANTS)
    width = 0.11
    offsets = np.linspace(-(n_variants - 1) / 2,
                          (n_variants - 1) / 2,
                          n_variants) * width
    x = np.arange(len(metric_names))

    fig, ax = plt.subplots(figsize=(16, 6))

    for i, v in enumerate(VARIANTS):
        vals = [metrics[m][v] for m in metric_names]
        bars = ax.bar(x + offsets[i], vals, width=width * 0.92,
                      color=COLORS[v], alpha=0.88, label=LABELS_SHORT[v],
                      edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.18,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=6.5,
                    color=COLORS[v], fontweight='bold')

    # Highlight AUC column — flag that ranking inverts relative to other metrics
    auc_idx = metric_names.index('AUC (%)')
    ax.axvspan(auc_idx - 0.43, auc_idx + 0.43, alpha=0.08,
               color='darkorange', zorder=0)
    ax.annotate('AUC ranking inverts\nvs. operational metrics\n(see discussion)',
                xy=(auc_idx + 0.42, 99.5), xytext=(auc_idx + 1.05, 104),
                fontsize=8.5, color='darkorange', ha='left',
                arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.2))

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=10.5)
    ax.set_ylabel('Metric Value (%)', fontsize=12)
    ax.set_ylim(60, 109)
    ax.legend(handles=legend_handles(), loc='lower right',
              fontsize=9.5, framealpha=0.92, ncol=2)

    plt.tight_layout()
    save_fig(fig, 'fig7_metric_comparison.png', out_dir)
    plt.close(fig)


# ===========================================================================
# Main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description='Generate MedDef-VISTA figures')
    p.add_argument('--out', default='out/meddef_vista_figures')
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    print(f'\n{"="*55}')
    print(f'  MedDef-VISTA Figure Generator v2')
    print(f'  Output: {os.path.abspath(out_dir)}')
    print(f'{"="*55}\n')

    print('[1/3]  Radar chart ...')
    fig_radar(out_dir)

    print('[2/3]  Stage-wise distillation improvement ...')
    fig_stage_improvement(out_dir)

    print('[3/3]  All-variant metric comparison ...')
    fig_metric_comparison(out_dir)

    # Delete old/superseded figures
    old_files = [
        'fig1_frontier_and_attacks.png',
        'fig3_robustness_heatmap.png',
        'fig5_calibration_auc.png',
        'fig6_3d_landscape.png',
        'fig7_metric_inconsistency.png',
        'fig_combined_6panel.png',
    ]
    for f in old_files:
        p = os.path.join(out_dir, f)
        if os.path.isfile(p):
            os.remove(p)
            print(f'  removed old: {f}')

    kept = sorted(f for f in os.listdir(out_dir) if f.endswith('.png'))
    print(f'\n{"="*55}')
    print(f'  Done!  {len(kept)} figures:')
    for f in kept:
        print(f'    {f}')
    print(f'{"="*55}\n')


if __name__ == '__main__':
    main()
