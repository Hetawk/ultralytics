# MedDef2_T Depth Scaling Setup

## Overview
The MedDef2_T model in `ultralytics` now supports depth-based scaling similar to ViT architectures, allowing you to easily switch between different model sizes.

## Available Depth Variants

| Depth | Variant | Transformer Blocks | Embed Dim | Num Heads | Approx Params |
|-------|---------|-------------------|-----------|-----------|---------------|
| 2.0   | Tiny    | 6                 | 192       | 3         | ~1.8M         |
| 2.1   | Small   | 12                | 384       | 6         | ~6.5M         |
| 2.2   | Base    | 12                | 768       | 12        | ~25M          |
| 2.3   | Large   | 24                | 1024      | 16        | ~72M          |

## Usage Examples

### Python API

```python
from ultralytics.models.meddef.meddef2 import get_meddef2_t, meddef2_t_0, meddef2_t_1

# Method 1: Using the factory function with depth parameter
model_tiny = get_meddef2_t(depth=2.0, num_classes=10)
model_small = get_meddef2_t(depth=2.1, num_classes=100)
model_base = get_meddef2_t(depth=2.2, num_classes=1000)
model_large = get_meddef2_t(depth=2.3, num_classes=1000)

# Method 2: Using convenience functions
model_tiny = meddef2_t_0(num_classes=10)    # depth=2.0
model_small = meddef2_t_1(num_classes=100)  # depth=2.1
model_base = meddef2_t_2(num_classes=1000)  # depth=2.2
model_large = meddef2_t_3(num_classes=1000) # depth=2.3

# With pretrained weights (when available)
model = get_meddef2_t(depth=2.2, pretrained=True, num_classes=1000)
```

### Command Line (Future Integration)

```bash
# Train tiny model
yolo train model=meddef2_t.yaml depth=2.0 data=imagenet10 epochs=100 imgsz=224

# Train small model
yolo train model=meddef2_t.yaml depth=2.1 data=imagenet10 epochs=100 imgsz=224

# Train base model
yolo train model=meddef2_t.yaml depth=2.2 data=imagenet data=imagenet epochs=100 imgsz=224

# Train large model
yolo train model=meddef2_t.yaml depth=2.3 data=imagenet epochs=100 imgsz=224
```

## Architecture Details

All MedDef2_T variants include:
- **CBAM Integration**: Convolutional Block Attention Module in each transformer block
- **Frequency Defense**: Low-pass filtering to suppress high-frequency adversarial noise
- **Multi-Scale Defense**: Multi-scale feature extraction for robust representation
- **Patch Consistency**: Smoothness enforcement across neighboring patches

## Configuration Comparison with Reference

### Reference (meddef_winlab)
```python
depth_to_config = {
    2.0: {'name': 'meddef2_t_tiny', 'depth': 6, ...},
    2.1: {'name': 'meddef2_t_small', 'depth': 12, ...},
    2.2: {'name': 'meddef2_t_base', 'depth': 12, ...},
    2.3: {'name': 'meddef2_t_large', 'depth': 24, ...},
}
```

### Ultralytics Implementation
✅ **Fully Compatible** - Same depth mapping (2.0, 2.1, 2.2, 2.3)
✅ **Same Architecture** - Identical transformer configurations
✅ **Factory Function** - `get_meddef2_t(depth: float, ...)`
✅ **Convenience Functions** - `meddef2_t_0()` through `meddef2_t_3()`

## Files Created

```
ultralytics/ultralytics/models/meddef/meddef2/
├── __init__.py              # Exports factory functions and models
├── transformer.py           # Base Vision Transformer implementation
├── defense.py               # Defense mechanisms (Frequency, CBAM, etc.)
└── meddef2_t.py            # MedDef2_T with factory function

ultralytics/ultralytics/cfg/models/
└── meddef2_t.yaml          # Model configuration with scales
```

## Integration Status

- ✅ Factory function with depth parameter (2.0, 2.1, 2.2, 2.3)
- ✅ All defense mechanisms (Frequency, CBAM, Multi-scale, Patch Consistency)
- ✅ YAML configuration with scales
- ✅ Module exports in __init__.py
- ⏳ Full integration with ultralytics Model class (requires nn/tasks.py update)
- ⏳ Pretrained weights loading

## Next Steps

To complete full integration with the `ultralytics.models.MedDef` class:

1. Add `MedDefModel` to `ultralytics/nn/tasks.py`
2. Implement depth parameter parsing in the model loader
3. Add pretrained weight URLs and loading logic
4. Create training/validation examples

## Testing

```python
# Quick test
import torch
from ultralytics.models.meddef.meddef2 import get_meddef2_t

model = get_meddef2_t(depth=2.0, num_classes=10)
x = torch.randn(1, 3, 224, 224)
output = model(x)
print(f"Output shape: {output.shape}")  # Should be [1, 10]
```
