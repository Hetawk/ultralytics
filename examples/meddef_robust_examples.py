#!/usr/bin/env python
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
MedDef Robust Medical Image Classification Example

This script demonstrates how to use MedDef models for robust medical image analysis
with protection against adversarial attacks.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.utils.defense import (
    RobustnessEvaluator, PGDAttack, FGSMAttack, CWAttack,
    AdversarialTraining
)


def create_dummy_medical_dataset(num_samples=100, img_size=224, num_classes=5):
    """Create a dummy medical imaging dataset for demonstration"""
    # Generate random "medical" images (normalized grayscale)
    X = torch.randn(num_samples, 3, img_size, img_size)
    X = (X - X.min()) / (X.max() - X.min())  # Normalize to [0, 1]
    
    # Generate random labels
    y = torch.randint(0, num_classes, (num_samples,))
    
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=16, shuffle=True)


def example_1_basic_inference():
    """Example 1: Basic inference with MedDef2 model"""
    print("=" * 60)
    print("Example 1: Basic Inference with MedDef2")
    print("=" * 60)
    
    # Initialize model from YAML configuration
    model = YOLO('meddef2.yaml')  # Load from config
    
    # Create dummy data
    test_loader = create_dummy_medical_dataset(num_samples=32)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            print(f"Predictions shape: {predictions.shape}")
            print(f"Sample predictions: {predictions[:5]}")
            break


def example_2_adversarial_evaluation():
    """Example 2: Evaluate robustness against adversarial attacks"""
    print("\n" + "=" * 60)
    print("Example 2: Adversarial Robustness Evaluation")
    print("=" * 60)
    
    # Initialize model
    model = YOLO('meddef2.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Create test data
    test_loader = create_dummy_medical_dataset(num_samples=32)
    
    # Create evaluator
    evaluator = RobustnessEvaluator(model, device=device)
    
    # Evaluate against different attacks
    attacks_to_test = ['fgsm', 'pgd']
    
    for attack_name in attacks_to_test:
        print(f"\nEvaluating against {attack_name.upper()} attack...")
        
        attack_kwargs = {
            'fgsm': {'epsilon': 8/255},
            'pgd': {'epsilon': 8/255, 'num_iter': 10, 'alpha': 2/255}
        }
        
        results = evaluator.evaluate(
            dataloader=test_loader,
            attack_name=attack_name,
            attack_kwargs=attack_kwargs[attack_name]
        )
        
        print(f"  Clean Accuracy: {results['clean_accuracy']:.2f}%")
        print(f"  Robust Accuracy: {results['robust_accuracy']:.2f}%")
        print(f"  Robustness Gap: {results['clean_accuracy'] - results['robust_accuracy']:.2f}%")


def example_3_adversarial_training():
    """Example 3: Train model with adversarial robustness"""
    print("\n" + "=" * 60)
    print("Example 3: Adversarial Training")
    print("=" * 60)
    
    # Initialize model for MedDef1 (ResNet-based)
    model = YOLO('meddef1.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Setup PGD attack
    attack = PGDAttack(
        model=model.model,  # Access underlying PyTorch model
        epsilon=8/255,
        alpha=2/255,
        num_iter=7,
        device=device
    )
    
    # Setup adversarial training
    adversarial_trainer = AdversarialTraining(
        model=model.model,
        attack=attack,
        epsilon=8/255,
        device=device
    )
    
    # Create training data
    train_loader = create_dummy_medical_dataset(num_samples=64)
    
    # Setup optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop (simplified)
    print("\nTraining with adversarial examples...")
    model.train()
    
    for epoch in range(3):  # 3 epochs for demo
        total_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            loss = adversarial_trainer.training_step(
                batch=images,
                labels=labels,
                optimizer=optimizer,
                criterion=criterion,
                attack_prob=0.5  # Use adversarial examples 50% of the time
            )
            
            total_loss += loss
            
            if batch_idx % 2 == 0:
                print(f"  Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.4f}")
        
        print(f"Epoch {epoch} completed. Avg Loss: {total_loss / len(train_loader):.4f}")


def example_4_multiple_scales():
    """Example 4: Compare different model scales"""
    print("\n" + "=" * 60)
    print("Example 4: Model Scale Comparison")
    print("=" * 60)
    
    scales = ['n', 's', 'm', 'l']  # nano, small, medium, large
    
    test_loader = create_dummy_medical_dataset(num_samples=32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\nTesting MedDef2 variants...")
    print(f"{'Scale':<10} {'Params':<12} {'Speed':<12} {'Memory':<12}")
    print("-" * 50)
    
    for scale in scales:
        config_name = f'meddef2{scale}.yaml'
        
        try:
            model = YOLO(config_name)
            model.to(device)
            model.eval()
            
            # Count parameters
            num_params = sum(p.numel() for p in model.parameters())
            
            # Measure inference speed
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224).to(device)
                
                import time
                start = time.time()
                for _ in range(10):
                    _ = model(dummy_input)
                elapsed = (time.time() - start) / 10 * 1000  # ms
            
            print(f"{scale:<10} {num_params/1e6:<12.2f}M {elapsed:<12.2f}ms {'N/A':<12}")
        
        except Exception as e:
            print(f"{scale:<10} (Config not found: {e})")


def example_5_defensive_distillation():
    """Example 5: Defensive distillation with teacher-student framework"""
    print("\n" + "=" * 60)
    print("Example 5: Defensive Distillation")
    print("=" * 60)
    
    from ultralytics.nn.modules.defense import DefensiveDistillationLoss
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create teacher and student models
    teacher_model = YOLO('meddef2m.yaml').to(device)
    student_model = YOLO('meddef2s.yaml').to(device)
    
    # Setup distillation loss
    distill_loss_fn = DefensiveDistillationLoss(
        temperature=4.0,
        alpha=0.5  # Balance between distillation and standard loss
    )
    
    # Create training data
    train_loader = create_dummy_medical_dataset(num_samples=64)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)
    
    print("\nTraining student with teacher distillation...")
    teacher_model.eval()
    student_model.train()
    
    for epoch in range(3):
        total_loss = 0.0
        
        with torch.no_grad():
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                # Get teacher logits
                teacher_logits = teacher_model(images)
                
                # Get student logits
                student_logits = student_model(images)
                
                # Compute distillation loss
                loss = distill_loss_fn(student_logits, labels, teacher_logits)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        
        print(f"Epoch {epoch}: Loss = {total_loss / len(train_loader):.4f}")


def example_6_certified_robustness():
    """Example 6: Certified robustness with randomized smoothing"""
    print("\n" + "=" * 60)
    print("Example 6: Certified Robustness (Randomized Smoothing)")
    print("=" * 60)
    
    from ultralytics.utils.defense import CertifiedDefense
    
    model = YOLO('meddef2.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Create test image
    test_image = torch.randn(1, 3, 224, 224).to(device)
    test_image = (test_image - test_image.min()) / (test_image.max() - test_image.min())
    
    print("\nComputing certified robustness radius...")
    
    pred_class, certified_radius = CertifiedDefense.randomized_smoothing(
        model=model.model if hasattr(model, 'model') else model,
        x=test_image,
        num_samples=100,
        sigma=0.25,
        num_classes=10
    )
    
    print(f"  Predicted class: {pred_class}")
    print(f"  Certified radius (L2): {certified_radius:.4f}")
    print(f"  This model is provably robust to L2 perturbations <= {certified_radius:.4f}")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("MedDef Robust Medical Image Classification - Examples")
    print("=" * 60)
    
    try:
        example_1_basic_inference()
    except Exception as e:
        print(f"Example 1 error: {e}")
    
    try:
        example_2_adversarial_evaluation()
    except Exception as e:
        print(f"Example 2 error: {e}")
    
    try:
        example_3_adversarial_training()
    except Exception as e:
        print(f"Example 3 error: {e}")
    
    try:
        example_4_multiple_scales()
    except Exception as e:
        print(f"Example 4 error: {e}")
    
    try:
        example_5_defensive_distillation()
    except Exception as e:
        print(f"Example 5 error: {e}")
    
    try:
        example_6_certified_robustness()
    except Exception as e:
        print(f"Example 6 error: {e}")
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
