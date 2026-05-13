# Analysis: metrics.py

## Introduction

The `metrics.py` module provides evaluation utilities for computing accuracy, FLOPs, parameter counts, and sparsity metrics. These are used across all experiment categories to compare methods.

## Implementation

### Functions

1. **`compute_accuracy(model, data_loader, device, topk=(1,))`**:
   - Runs model in eval mode on data_loader
   - Computes top-k accuracy (default: top-1)
   - Returns accuracy as percentage

2. **`compute_flops(model, input_shape, device)`**:
   - Uses `thop` library for FLOP counting
   - Input shape: (batch_size, channels, height, width)
   - Returns total FLOPs as integer

3. **`compute_parameter_count(model, count_pruned=False)`**:
   - Iterates all parameters
   - If `count_pruned=False`: counts all parameters
   - If `count_pruned=True`: counts only non-zero parameters (after masking)
   - Returns dict with total, pruned, percentage

4. **`compute_sparsity(model)`**:
   - For each parameter with 'weight' or 'bias' in name:
     - Count zeros vs. total elements
   - Returns overall sparsity % and per-layer sparsity dict

5. **`compute_recovery_score(model, original_model)`**:
   - Measures how much pruned network recovered accuracy vs. original dense model
   - recovery_score = pruned_accuracy / original_accuracy * 100

6. **`compute_flops_reduction(original_flops, current_flops)`**:
   - Returns reduction as percentage

7. **`compute_parameter_reduction(original_params, current_params)`**:
   - Returns reduction as percentage

## Notes

- All functions should accept `device` parameter for GPU inference
- `compute_accuracy` should use `torch.no_grad()` context
- FLOP counting requires `thop` package (in requirements.txt)
- Sparsity computation counts exact zeros in weight tensors
- Edge case: If model has no bias terms, bias sparsity is 0