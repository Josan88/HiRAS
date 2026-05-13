# Analysis: ablation_sparsity.py

## Introduction

This experiment (Category C from plan.md) evaluates the accuracy vs. sparsity tradeoff across target compression levels. Tests 5 sparsity targets on 3 dataset/model configurations.

## Implementation

### Experiment Design

```
sparsity_targets = [0.5, 0.6, 0.7, 0.8, 0.9]  # 50% to 90%

dataset_configs = [
    ("cifar10", "resnet20"),
    ("cifar100", "resnet20"),  
    ("imagenet", "resnet50")
]
```

### Total runs: 5 targets × 3 configs = 15 training runs

### Implementation Steps

1. **Parse config**: Get sparsity_targets, dataset/model configs
2. **For each (sparsity_target, dataset, model) combination**:
   - Initialize trainer with target sparsity
   - Train to completion
   - Record final accuracy
3. **Plot results**: 3 subplots (one per dataset), x=sparsity, y=accuracy
4. **Save table**: Tabular results with all configurations

## Notes

- ImageNet runs are expensive; use single seed (42) to reduce cost
- CIFAR runs can use 3 seeds for statistical significance
- Results correspond to Table 2 in paper (sparsity ablation)
- Key claim to verify: accuracy remains >90% of baseline at 80% sparsity