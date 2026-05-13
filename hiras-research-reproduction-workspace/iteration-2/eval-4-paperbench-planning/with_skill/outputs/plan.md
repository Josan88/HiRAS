# HiRAS Phase 1: Overall Experiment Plan

## Paper: Adaptive Pruning for Neural Networks

---

## Experiment Categories

### Category A: Main Performance Comparisons
**Goal:** Demonstrate adaptive pruning outperforms static, gradual, and lottery ticket baselines across compression ratios.

**Datasets:**
- CIFAR-10 (ResNet-20, VGG-16)
- CIFAR-100 (ResNet-20)
- ImageNet (ResNet-50)

**Models/Algorithms:**
- Adaptive pruning (proposed method)
- Static magnitude pruning
- Gradual magnitude pruning (GMP)
- Lottery Ticket Hypothesis pruning

**Evaluation Protocol:**
- Train to completion with pruning enabled
- Record top-1/top-5 accuracy at end of training
- Record final FLOPs and parameter count

**Metrics:**
- Top-1 Accuracy (%)
- Top-5 Accuracy (%)
- FLOPs reduction (%)
- Parameter count reduction (%)

**Dependencies:** Must run after data loaders and model architectures are implemented.

---

### Category B: Ablation — Pruning Schedule
**Goal:** Evaluate how different pruning schedules affect final sparsity and accuracy.

**Schedule variants to test:**
1. Constant-rate schedule (remove X% every interval)
2. Exponential decay schedule (remove decreasing % each interval)
3. Polynomial decay schedule
4. Step-function schedule (sparsity changes at fixed epochs)

**Datasets:** CIFAR-10 only (ResNet-20)

**Evaluation Protocol:**
- Hold sparsity target constant at 70%
- Train with each schedule variant
- Record accuracy and convergence curves

**Dependencies:** Requires adaptive scheduler module (Category A).

---

### Category C: Ablation — Sparsity Targets
**Goal:** Evaluate accuracy vs. sparsity tradeoff across target compression levels.

**Sparsity targets:** 50%, 60%, 70%, 80%, 90%

**Datasets:** CIFAR-10 (ResNet-20), CIFAR-100 (ResNet-20), ImageNet (ResNet-50)

**Evaluation Protocol:**
- Train separate model for each sparsity target
- Record accuracy at each sparsity level

**Dependencies:** Requires adaptive scheduler module (Category A).

---

### Category D: Ablation — Recovery Phases
**Goal:** Determine whether sparsity recovery phases improve final model quality.

**Recovery phase variants:**
1. No recovery (continuous pruning)
2. Recovery every 20 epochs (10 epoch recovery after every 20 epochs of pruning)
3. Recovery every 30 epochs (10 epoch recovery)
4. Recovery every 10 epochs (more frequent, shorter recovery)

**Dataset:** CIFAR-10 (ResNet-20)

**Evaluation Protocol:**
- Hold sparsity target at 70%
- Train with each recovery configuration
- Record accuracy curves and final sparsity achieved

**Dependencies:** Requires adaptive scheduler module (Category A).

---

### Category E: Mobile Deployment Benchmarks
**Goal:** Validate inference latency, throughput, and accuracy on Jetson Nano.

**Models:** ResNet-50 pruned to 80% sparsity (trained on ImageNet)

**Hardware:** Jetson Nano (or compatible mobile GPU)

**Evaluation Protocol:**
1. Export pruned model to ONNX
2. Convert to TensorRT FP16 engine
3. Benchmark inference latency (ms per image)
4. Benchmark throughput (images/sec)
5. Compare against baseline (unpruned) model

**Metrics:**
- Latency (ms, mean ± std)
- Throughput (images/sec)
- Memory footprint (MB)
- Accuracy retention (% of full model's top-1)

**Dependencies:** Requires trained ResNet-50 model from Category A/C.

---

## Inter-Experiment Dependencies

```
data_loaders (foundation)
    └── model_architectures (foundation)
            ├── Category A: Main comparisons
            │       ├── Category B: Ablation schedule
            │       ├── Category C: Ablation sparsity
            │       └── Category D: Ablation recovery
            └── Category E: Mobile deployment
                    (requires trained models from Category A/C)
```

---

## Summary Table

| Category | Experiments | Files Produced | Rubric Items |
|----------|-------------|----------------|--------------|
| A | 12 (4 methods × 3 datasets) | `main_comparison.py` | 1-8, 14, 17-20 |
| B | 4 (schedule variants) | `ablation_schedule.py` | 11 |
| C | 5 (sparsity targets) × 3 datasets | `ablation_sparsity.py` | 12 |
| D | 4 (recovery variants) | `ablation_recovery.py` | 13 |
| E | 3 (latency/throughput/memory) | `tensorrt_runner.py`, `model_exporter.py` | 8, 10, 23 |

---

## Expected Outcomes (from paper claims)
- Adaptive pruning achieves same or better accuracy than baselines at 80% sparsity
- 2-3× training speedup vs. gradual pruning due to smart schedule
- Jetson Nano deployment: <5ms latency, >100 img/s throughput
- Recovery phases recover ~1-2% accuracy at high sparsity