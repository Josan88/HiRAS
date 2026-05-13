# Analysis: ablation_recovery.py

## Introduction

This experiment (Category D from plan.md) evaluates whether sparsity recovery phases improve final model quality. Tests 4 recovery configurations on CIFAR-10/ResNet-20.

## Implementation

### Experiment Design

```
recovery_intervals = [
    0,    # No recovery (continuous pruning)
    10,   # Recovery every 10 epochs
    20,   # Recovery every 20 epochs
    30    # Recovery every 30 epochs
]
recovery_length = 10  # epochs
sparsity_target = 0.7
```

### Implementation Steps

1. **Parse config**: Get recovery_intervals, recovery_length, sparsity_target
2. **For each recovery_interval**:
   - Create AdaptiveScheduler with specific recovery config
   - Train for 150 epochs
   - Record accuracy curve, sparsity curve
3. **Plot two figures**:
   - Left: Accuracy vs. epochs for all 4 configs
   - Right: Sparsity vs. epochs for all 4 configs
4. **Save comparison table**: Final accuracy and achieved sparsity

## Notes

- Recovery phases are 10 epochs where no pruning occurs
- During recovery, network can "heal" from pruning-induced accuracy drops
- Edge case: If recovery_interval=0, scheduler never enters recovery mode
- Results help tune the recovery hyperparameter for production use
- Corresponds to Section 4.3 in paper (ablation studies)