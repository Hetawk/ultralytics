#!/usr/bin/env python3
"""
Test script for MedDef2_T depth scaling setup.

This script verifies that the depth-based factory function works correctly
for all model variants (tiny, small, base, large).
"""

import sys
from pathlib import Path

# Add ultralytics to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import torch
from ultralytics.models.meddef.meddef2 import (
    get_meddef2_t,
    meddef2_t_0,
    meddef2_t_1,
    meddef2_t_2,
    meddef2_t_3,
)


def test_factory_function():
    """Test the get_meddef2_t factory function with different depths."""
    print("=" * 60)
    print("Testing MedDef2_T Factory Function")
    print("=" * 60)
    
    depths = [2.0, 2.1, 2.2, 2.3]
    names = ['Tiny', 'Small', 'Base', 'Large']
    expected_depths = [6, 12, 12, 24]
    expected_dims = [192, 384, 768, 1024]
    
    for depth, name, exp_depth, exp_dim in zip(depths, names, expected_depths, expected_dims):
        print(f"\n{name} variant (depth={depth}):")
        
        try:
            model = get_meddef2_t(depth=depth, num_classes=10)
            
            # Verify configuration
            assert model.embed_dim == exp_dim, f"Expected embed_dim={exp_dim}, got {model.embed_dim}"
            assert len(model.blocks) == exp_depth, f"Expected {exp_depth} blocks, got {len(model.blocks)}"
            
            # Test forward pass
            x = torch.randn(2, 3, 224, 224)
            output = model(x)
            
            assert output.shape == (2, 10), f"Expected output shape (2, 10), got {output.shape}"
            
            # Count parameters
            param_count = sum(p.numel() for p in model.parameters())
            
            print(f"  ✓ Embed dim: {model.embed_dim}")
            print(f"  ✓ Transformer blocks: {len(model.blocks)}")
            print(f"  ✓ Output shape: {output.shape}")
            print(f"  ✓ Parameters: {param_count:,}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return False
    
    return True


def test_convenience_functions():
    """Test the convenience functions (meddef2_t_0, _1, _2, _3)."""
    print("\n" + "=" * 60)
    print("Testing Convenience Functions")
    print("=" * 60)
    
    functions = [meddef2_t_0, meddef2_t_1, meddef2_t_2, meddef2_t_3]
    names = ['meddef2_t_0 (Tiny)', 'meddef2_t_1 (Small)', 'meddef2_t_2 (Base)', 'meddef2_t_3 (Large)']
    
    for func, name in zip(functions, names):
        print(f"\n{name}:")
        
        try:
            model = func(num_classes=100)
            x = torch.randn(1, 3, 224, 224)
            output = model(x)
            
            assert output.shape == (1, 100), f"Expected output shape (1, 100), got {output.shape}"
            
            print(f"  ✓ Created successfully")
            print(f"  ✓ Output shape: {output.shape}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return False
    
    return True


def test_defense_mechanisms():
    """Test that defense mechanisms are properly integrated."""
    print("\n" + "=" * 60)
    print("Testing Defense Mechanisms")
    print("=" * 60)
    
    model = get_meddef2_t(depth=2.0, num_classes=10)
    
    # Check defense components
    assert hasattr(model, 'defense'), "Missing defense module"
    assert hasattr(model, 'frequency_defense'), "Missing frequency defense"
    assert hasattr(model, 'patch_consistency'), "Missing patch consistency"
    
    # Check CBAM in transformer blocks
    first_block = model.blocks[0]
    assert hasattr(first_block, 'cbam'), "Missing CBAM in transformer block"
    
    print("\n  ✓ DefenseModule present")
    print("  ✓ FrequencyDefense present")
    print("  ✓ PatchConsistency present")
    print("  ✓ CBAM integrated in transformer blocks")
    
    return True


def main():
    """Run all tests."""
    print("\nMedDef2_T Depth Scaling Setup - Test Suite\n")
    
    tests = [
        ("Factory Function", test_factory_function),
        ("Convenience Functions", test_convenience_functions),
        ("Defense Mechanisms", test_defense_mechanisms),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
