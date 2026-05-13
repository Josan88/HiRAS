# HiRAS Phase 2: Architecture Design

## Directory Layout

```
adaptive-pruning/
├── README.md
├── requirements.txt
├── config.yaml
├── main.py                          # Entry point
├── data/
│   ├── __init__.py
│   ├── cifar_loader.py              # CIFAR-10/100 data loading
│   ├── imagenet_loader.py           # ImageNet data loading
│   └── download_scripts.sh         # Dataset download scripts
├── models/
│   ├── __init__.py
│   ├── resnet.py                    # ResNet-20/50 implementations
│   └── vgg.py                       # VGG-16 implementation
├── pruning/
│   ├── __init__.py
│   ├── base_pruner.py               # Abstract base pruner class
│   ├── adaptive_scheduler.py        # Core adaptive pruning algorithm
│   ├── static_pruner.py             # Baseline: static magnitude pruning
│   ├── gradual_pruner.py            # Baseline: gradual magnitude pruning
│   ├── lottery_ticket_pruner.py     # Baseline: lottery ticket pruning
│   └── mask_manager.py              # Sparsity mask management
├── trainers/
│   ├── __init__.py
│   ├── base_trainer.py              # Base training loop
│   └── pruning_trainer.py          # Training loop with pruning integration
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                   # Accuracy, FLOPs, parameter metrics
│   └── deployment_metrics.py       # Latency/throughput for Jetson Nano
├── deployment/
│   ├── __init__.py
│   ├── model_exporter.py            # ONNX export for pruned models
│   └── tensorrt_runner.py           # TensorRT inference runner
├── experiments/
│   ├── __init__.py
│   ├── main_comparison.py           # Category A: main comparisons
│   ├── ablation_schedule.py         # Category B: schedule ablation
│   ├── ablation_sparsity.py         # Category C: sparsity ablation
│   └── ablation_recovery.py         # Category D: recovery ablation
├── visualization/
│   ├── __init__.py
│   └── plots.py                     # Sparsity curves, accuracy plots
├── utils/
│   ├── __init__.py
│   ├── checkpoint.py                # Save/load checkpoints with sparsity
│   └── logger.py                    # WandB/TensorBoard logger
└── analysis/
    ├── components.txt
    ├── adaptive_scheduler_analysis.md
    ├── metrics_analysis.md
    └── ... (per-component analysis files)
```

---

## Module Responsibilities

### `main.py`
Entry point. Parses CLI args, selects experiment category, runs training/evaluation.

### `data/cifar_loader.py`
- Load CIFAR-10/CIFAR-100 using torchvision.datasets
- Apply standard preprocessing: random crop, horizontal flip, normalization
- Return DataLoader with configured batch size

### `data/imagenet_loader.py`
- Load ImageNet using torchvision.datasets.ImageFolder
- Preprocessing: resize 256, center crop 224, ImageNet normalization
- Return DataLoader

### `models/resnet.py`
- ResNet-20 for CIFAR-10/100 (adapted architecture, no max-pool)
- ResNet-50 for ImageNet (standard architecture)
- Forward pass supporting masked weights

### `models/vgg.py`
- VGG-16 for CIFAR-10
- Forward pass supporting masked weights

### `pruning/base_pruner.py`
- Abstract base class defining prune() and update_mask() interface
- All pruning strategies inherit from this

### `pruning/adaptive_scheduler.py`
- Core adaptive pruning: adjusts sparsity based on training dynamics
- SparsityManager: tracks current sparsity, decides when to prune more
- PruningScheduler: determines how much to prune at each interval
- RecoveryManager: handles recovery phases

### `pruning/static_pruner.py`
- One-shot magnitude pruning to target sparsity
- Used as baseline

### `pruning/gradual_pruner.py`
- Fixed schedule gradual magnitude pruning (GMP)
- Used as baseline

### `pruning/lottery_ticket_pruner.py`
- Iterative pruning with weight rewinding
- Used as baseline

### `pruning/mask_manager.py`
- Manages binary masks for each layer
- Supports mask multiplication for inference
- Handles mask updates during training

### `trainers/base_trainer.py`
- Standard training loop: forward, loss, backward, optimizer step
- Logging of loss, accuracy

### `trainers/pruning_trainer.py`
- Extends base trainer with pruning hook integration
- Calls pruner at configured intervals
- Tracks sparsity over time

### `evaluation/metrics.py`
- compute_accuracy(): top-1/top-5 accuracy
- compute_flops(): FLOP count from model architecture
- compute_parameter_count(): total/w pruned parameters
- compute_sparsity(): layer-wise and overall sparsity %

### `evaluation/deployment_metrics.py`
- measure_latency(): single inference timing
- measure_throughput(): batch throughput
- measure_memory(): GPU memory usage

### `deployment/model_exporter.py`
- export_to_onnx(): convert pruned model to ONNX
- apply_masks(): zero out pruned weights before export

### `deployment/tensorrt_runner.py`
- TensorRT engine builder (FP16)
- run_inference(): benchmark on Jetson Nano
- handle_input_output(): format conversion

### `experiments/main_comparison.py`
- Runs all 4 methods × all 3 datasets
- Produces comparison table
- Logs to wandb

### `experiments/ablation_schedule.py`
- Runs 4 schedule variants on CIFAR-10/ResNet-20
- Plots accuracy curves

### `experiments/ablation_sparsity.py`
- Runs 5 sparsity targets × 3 dataset configs
- Plots accuracy vs. sparsity curve

### `experiments/ablation_recovery.py`
- Runs 4 recovery phase configs
- Compares final accuracy and training curves

### `visualization/plots.py`
- plot_sparsity_curve(): sparsity over training epochs
- plot_accuracy_curve(): accuracy over training epochs
- plot_comparison_table(): main results table
- plot_ablation_results(): ablation study plots

### `utils/checkpoint.py`
- save_checkpoint(): model + optimizer + pruning state + sparsity mask
- load_checkpoint(): restore full state

### `utils/logger.py`
- Logger class wrapping wandb or tensorboard
- log_metrics(), log_sparsity(), log_artifact()

---

## Interfaces (Key Public Methods)

| Module | Public Methods |
|--------|----------------|
| `adaptive_scheduler.py` | `prune()`, `adjust_sparsity()`, `enter_recovery()`, `exit_recovery()` |
| `static_pruner.py` | `prune()` |
| `gradual_pruner.py` | `prune()`, `set_schedule()` |
| `lottery_ticket_pruner.py` | `prune()`, `rewind_weights()` |
| `mask_manager.py` | `apply_mask()`, `update_mask()`, `get_sparsity()`, `save_mask()`, `load_mask()` |
| `pruning_trainer.py` | `train()`, `validate()` |
| `metrics.py` | `compute_accuracy()`, `compute_flops()`, `compute_parameter_count()` |
| `tensorrt_runner.py` | `build_engine()`, `run_inference()` |
| `checkpoint.py` | `save_checkpoint()`, `load_checkpoint()` |

---

## Entry Points

1. **Training:** `python main.py train --config config.yaml --method adaptive --dataset cifar10`
2. **Evaluation:** `python main.py eval --checkpoint checkpoint.pth --dataset cifar10`
3. **Ablation:** `python main.py ablation --ablation-type schedule`
4. **Deployment:** `python main.py deploy --model pruned_resnet50.pth --runtime tensorrt`

---

## External Dependencies

| Library | Purpose |
|---------|---------|
| PyTorch 2.0+ | Model definition, training |
| torchvision 0.15+ | Data loading, model architectures |
| numpy | Numerical operations |
| pandas | Tabular results |
| matplotlib | Visualization |
| wandb | Experiment tracking |
| tensorboard | Logging |
| onnx | Model export |
| tensorrt | GPU inference optimization (Jetson Nano) |
| pycuda | TensorRT integration |
| thop | FLOP counting |