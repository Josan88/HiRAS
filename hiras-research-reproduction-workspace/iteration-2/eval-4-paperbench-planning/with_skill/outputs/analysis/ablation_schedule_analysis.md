# Analysis: ablation_schedule.py

## Introduction

This experiment (Category B from plan.md) evaluates how different pruning schedules affect the accuracy-sparsity tradeoff. It tests 4 schedule variants on CIFAR-10/ResNet-20 at 70% target sparsity.

## Implementation

### Experiment Design

```python
schedule_variants = [
    "constant_rate",   # Remove X% every interval
    "exponential_decay",  # Remove decreasing % each interval  
    "polynomial_decay",    # Polynomial decay schedule
    "step_function"        # Sparsity changes at fixed epochs
]
```

### Expected Results (from paper claims)
- Different schedules lead to different convergence behavior
- Adaptive outperforms fixed schedules (paper's main claim)

### Implementation Steps

1. **Load config**: Read sparsity_target (0.7), dataset (cifar10), model (resnet20)
2. **For each schedule variant**:
   - Create new PruningTrainer with variant-specific pruner
   - Run training for 150 epochs
   - Log accuracy every epoch
3. **Plot results**: Accuracy curves for all 4 variants overlaid
4. **Save comparison table**: Final accuracy for each schedule

### Key Functions

- `run_schedule_ablation()`: Main entry point
- `get_constant_rate_scheduler()`: Fixed % per interval
- `get_exponential_scheduler()`: Decaying % per interval
- `get_polynomial_scheduler()`: Smooth polynomial decay
- `get_step_scheduler()`: Step changes at epochs [30, 60, 90, 120]

## Notes

- All runs use same seed (42) for fair comparison
- Only CIFAR-10/ResNet-20 (not ImageNet) — computational constraint
- Sparsity target held constant at 70% across all variants
- Results feed into Figure 3 in paper (schedule ablation study)