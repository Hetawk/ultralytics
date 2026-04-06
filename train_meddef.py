#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
MedDef2 Training Script

Usage:
    python train_meddef.py --data path/to/data --variant full --depth base --epochs 100
    python train_meddef.py --data path/to/data --variant no_cbam --depth tiny --epochs 50
    python train_meddef.py --data path/to/data --variant baseline --depth small --name exp1

Variants: full, no_def, no_freq, no_patch, no_cbam, baseline
Depths: tiny, small, base, large
"""

import argparse
from pathlib import Path

from ultralytics.models.meddef import MedDefTrainer
from ultralytics.nn.tasks import yaml_model_load


def parse_args():
    parser = argparse.ArgumentParser(description='Train MedDef2 models')

    # Required
    parser.add_argument('--data', type=str, required=True,
                        help='Path to dataset (e.g., path/to/skin_cancer)')

    # Model configuration
    parser.add_argument('--variant', type=str, default='full',
                        choices=['full', 'no_def', 'no_freq',
                                 'no_patch', 'no_cbam', 'baseline'],
                        help='Model variant for ablation study')
    parser.add_argument('--depth', type=str, default='small',
                        choices=['tiny', 'small', 'base', 'large'],
                        help='Model depth/size (default: small)')
    parser.add_argument('--nc', type=int, default=None,
                        help='Number of classes (auto-detected if not specified)')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--nbs', type=int, default=None,
                        help='Nominal batch size for gradient accumulation scaling')
    parser.add_argument('--imgsz', type=int, default=224, help='Image size')
    parser.add_argument('--device', type=str, default='0', help='CUDA device')
    parser.add_argument('--workers', type=int, default=8,
                        help='Dataloader workers')
    parser.add_argument('--fraction', type=float, default=1.0,
                        help='Fraction of train data to use (0 < fraction <= 1.0)')

    # Output
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name (default: {variant}_{depth})')
    parser.add_argument('--project', type=str, default=None,
                        help='Project directory')

    # Optimizer
    parser.add_argument('--lr0', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01,
                        help='Final LR = lr0 * lrf')
    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='Optimizer')

    # LR schedule
    parser.add_argument('--cos_lr', type=str, default='false',
                        help='Use cosine LR schedule (true/false)')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience (0 = disabled)')

    # Regularisation
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='Optimizer weight decay')
    parser.add_argument('--warmup_epochs', type=float, default=3.0,
                        help='Warmup epochs (default: 3.0)')
    parser.add_argument('--warmup_bias_lr', type=float, default=0.1,
                        help='Warmup initial bias lr (default: 0.1)')
    parser.add_argument('--dropout', type=float,
                        default=0.0, help='Dropout probability')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label smoothing epsilon (0.0 = off)')
    parser.add_argument('--erasing', type=float, default=0.4,
                        help='Random erasing probability in classification augs (default: 0.4)')

    # Augmentation
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='MixUp alpha (0.0 = off)')
    parser.add_argument('--cutmix', type=float, default=0.0,
                        help='CutMix alpha (0.0 = off)')

    # Checkpointing
    parser.add_argument('--save_period', type=int, default=-1,
                        help='Save checkpoint every N epochs (-1 = disabled, save last+best only)')

    # Defensive distillation (MedDef YAMLs enable this by default)
    parser.add_argument('--def_distill', action='store_true',
                        help='Explicitly enable defensive distillation for this run')
    parser.add_argument('--no_def_distill', action='store_true',
                        help='Disable defensive distillation for this run')
    parser.add_argument('--dist_temp', type=float, default=None,
                        help='Override distillation temperature from the model YAML')
    parser.add_argument('--dist_alpha', type=float, default=None,
                        help='Override distillation alpha from the model YAML')
    parser.add_argument('--teacher_model', type=str, default=None,
                        help='Checkpoint path to reuse as the frozen defensive-distillation teacher')

    # Class imbalance
    parser.add_argument('--class_weights', type=str, default='false',
                        help='Use inverse-frequency WeightedRandomSampler (true/false)')

    # Resume
    parser.add_argument('--resume', type=str, nargs='?', const=True, default=False,
                        help='Resume training: pass a path to last.pt, or bare flag to auto-detect last run')
    parser.add_argument('--pretrained', type=str,
                        default=None, help='Pretrained weights')
    parser.add_argument('--exist_ok', action='store_true',
                        help='Overwrite existing experiment dir instead of creating a new numbered one')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.fraction <= 0 or args.fraction > 1.0:
        raise ValueError(f"--fraction must be in (0, 1], got {args.fraction}")

    auto_name = args.name is None
    if auto_name:
        args.name = f"{args.variant}_{args.depth}"

    # Map variant to YAML config
    variant_to_yaml = {
        'full': 'meddef2.yaml',
        'no_def': 'meddef2_no_def.yaml',
        'no_freq': 'meddef2_no_freq.yaml',
        'no_patch': 'meddef2_no_patch.yaml',
        'no_cbam': 'meddef2_no_cbam.yaml',
        'baseline': 'meddef2_baseline.yaml',
    }

    # Depth to scale mapping
    depth_to_scale = {
        'tiny': 'n',
        'small': 's',
        'base': 'm',
        'large': 'l',
    }

    model_yaml = variant_to_yaml[args.variant]
    scale = depth_to_scale[args.depth]

    # Load and optionally override the built-in distillation config.
    model_cfg = yaml_model_load(model_yaml)
    distill_cfg = dict(model_cfg.get('distillation', {}) or {})
    if not distill_cfg:
        distill_cfg = {'enabled': True, 'temperature': 4.0, 'alpha': 0.5}
    if args.no_def_distill:
        distill_cfg['enabled'] = False
    elif args.def_distill:
        distill_cfg['enabled'] = True
    if args.dist_temp is not None:
        distill_cfg['temperature'] = args.dist_temp
    if args.dist_alpha is not None:
        distill_cfg['alpha'] = args.dist_alpha
    if args.teacher_model:
        distill_cfg['teacher_model'] = str(
            Path(args.teacher_model).expanduser())
    model_cfg['distillation'] = distill_cfg

    # Derive dataset name from path for a clean, stable output dir.
    # e.g. /data/scisic  -> dataset_name='scisic'
    #      /data/tbcr    -> dataset_name='tbcr'
    from pathlib import Path as _Path
    dataset_name = _Path(args.data).name

    teacher_model = model_cfg['distillation'].get('teacher_model')

    # When no explicit --project given, standard runs stay under runs/classify/train/<dataset>.
    # For stage-2 distillation runs, default to a nested .../<variant>/distill layout.
    if args.project:
        run_project = args.project
        if auto_name and teacher_model:
            run_name = 'distill'
        else:
            run_name = args.name if args.name else args.variant
    else:
        if teacher_model:
            run_project = f"runs/classify/train/{dataset_name}/{args.variant}"
            run_name = 'distill' if auto_name else args.name
        else:
            run_project = f"runs/classify/train/{dataset_name}"
            run_name = args.variant

    # Build overrides
    overrides = {
        'model': model_cfg,
        'data': args.data,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'device': args.device,
        'workers': args.workers,
        'fraction': args.fraction,
        # Save to runs/classify/train/<dataset>/<variant>  (no trailing numbers)
        'project': run_project,
        'name': run_name,
        # Always reuse existing dir so re-runs don't pile up numbered copies.
        'exist_ok': True,
        'lr0': args.lr0,
        'lrf': args.lrf,
        'optimizer': args.optimizer,
        'scale': scale,
        'resume': args.resume,
        # LR schedule
        'cos_lr': args.cos_lr.lower() == 'true',
        'patience': args.patience,
        # Regularisation
        'weight_decay': args.weight_decay,
        'warmup_epochs': args.warmup_epochs,
        'warmup_bias_lr': args.warmup_bias_lr,
        'dropout': args.dropout,
        # NOTE: label_smoothing was removed from Ultralytics cfg; passing it has no effect.
        # To enable label smoothing, override the criterion in MedDefTrainer directly.
        # Augmentation
        'erasing': args.erasing,
        'mixup': args.mixup,
        'cutmix': args.cutmix,
        # Checkpointing
        'save_period': args.save_period,
        # Class imbalance
        'use_class_weights': args.class_weights.lower() == 'true',
    }

    if args.nbs is not None:
        overrides['nbs'] = args.nbs

    if args.pretrained:
        overrides['pretrained'] = args.pretrained

    # Print configuration
    print('=' * 60)
    print('MedDef2 Training Configuration')
    print('=' * 60)
    print(f'  Variant:  {args.variant}')
    print(f'  Depth:    {args.depth} (scale={scale})')
    print(f'  Data:     {args.data}')
    print(f'  Epochs:   {args.epochs}')
    print(f'  Batch:    {args.batch}')
    if args.nbs is not None:
        print(f'  NBS:      {args.nbs}')
    print(f'  ImgSize:  {args.imgsz}')
    print(f'  Device:   {args.device}')
    print(f'  Fraction: {args.fraction}')
    print(f'  LR:       lr0={args.lr0}  lrf={args.lrf}  cos_lr={args.cos_lr}')
    print(f'  Patience: {args.patience}  weight_decay={args.weight_decay}')
    print(
        f'  Warmup:   {args.warmup_epochs} epochs  bias_lr={args.warmup_bias_lr}')
    print(f'  Dropout:  {args.dropout}  LabelSmooth: {args.label_smoothing}')
    print(
        f'  Erasing:  {args.erasing}  Mixup: {args.mixup}  CutMix: {args.cutmix}')
    print(f'  SavePeriod: {args.save_period}  ClassW: {args.class_weights}')
    print(
        f"  Distill:  enabled={model_cfg['distillation'].get('enabled', False)}  "
        f"temp={model_cfg['distillation'].get('temperature', '-')}  "
        f"alpha={model_cfg['distillation'].get('alpha', '-')}"
    )
    print(
        f"  Teacher:  {model_cfg['distillation'].get('teacher_model', 'auto-init copy')}")
    print('  AdvTrain: disabled (defensive-distillation workflow)')
    print(f'  Resume:   {args.resume}')
    print(f'  Save:     {run_project}/{run_name}')
    print('=' * 60)

    # Create trainer and train
    trainer = MedDefTrainer(overrides=overrides)
    trainer.train()

    print('\n' + '=' * 60)
    print('Training Complete!')
    print(f'Results saved to: {trainer.save_dir}')
    print('=' * 60)


if __name__ == '__main__':
    main()
