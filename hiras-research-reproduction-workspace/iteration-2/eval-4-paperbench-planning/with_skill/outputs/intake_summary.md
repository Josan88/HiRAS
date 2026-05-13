# HiRAS Phase 0: Intake Summary

## Paper: Adaptive Pruning for Neural Networks
**Type:** ML Systems — Training-time neural network compression

---

## Core Research Question
How to automatically adjust network sparsity during training via a closed-loop adaptive pruning scheduler that responds to training dynamics, enabling high compression without accuracy loss.

## Claimed Contributions
1. Adaptive pruning method with dynamic sparsity adjustment
2. Sparsity-aware training protocol with recovery phases
3. Comparisons against static, gradual, and lottery ticket baselines
4. Mobile GPU deployment (Jetson Nano) with TensorRT optimization

---

## Experimental Setup

### Datasets
| Dataset | Models | Purpose |
|---------|--------|---------|
| CIFAR-10 | ResNet-20, VGG-16 | Small-scale compression evaluation |
| CIFAR-100 | ResNet-20 | Medium-scale multi-class |
| ImageNet | ResNet-50 | Large-scale realistic |

### Baselines
- **Static pruning**: Single-shot magnitude pruning at fixed sparsity
- **Gradual pruning**: Fixed schedule pruning (GMP)
- **Lottery Ticket Hypothesis**: Iterative magnitude pruning with rewinding

### Metrics
- Top-1 / Top-5 accuracy
- FLOPs reduction (%)
- Parameter count reduction (%)
- Training time (epochs to reach target sparsity)
- Inference latency (ms)
- Throughput (images/sec) on Jetson Nano

---

## Hyperparameters (from paper claims)

| Parameter | Value |
|-----------|-------|
| Initial learning rate | 0.1 |
| LR schedule | Cosine annealing |
| Batch size | 256 |
| Weight decay | 5e-4 |
| Momentum | 0.9 |
| Pruning start epoch | 0 |
| Pruning end epoch | 150 (CIFAR) / 90 (ImageNet) |
| Target sparsity | {50%, 70%, 80%, 90%} |
| Recovery phase length | 10 epochs |
| Sparsity adjustment interval | 5 epochs |

---

## Ambiguities / Missing Information
- Exact sparsity adjustment step size (paper states "gradual adjustment" — specific delta not specified)
- ImageNet training epochs not explicitly stated (assumed 90 from training schedules)
- Jetson Nano TensorRT precision mode (FP16/INT8) not specified
- Specific FLOPs for baseline models not tabulated

---

## Rubric Item Mapping (25 Code Development Requirements)

| # | Rubric Item | HiRAS Phase | Module/File |
|---|-------------|------------|-------------|
| 1 | Implementation of adaptive pruning algorithm | Ph 1, 5, 6 | `pruning/adaptive_scheduler.py` |
| 2 | Training loop with pruning schedule integration | Ph 1, 5, 6 | `trainers/base_trainer.py`, `trainers/pruning_trainer.py` |
| 3 | Data loading and preprocessing (CIFAR-10/100/ImageNet) | Ph 1, 5, 6 | `data/cifar_loader.py`, `data/imagenet_loader.py` |
| 4 | Baseline: static pruning | Ph 1, 5, 6 | `pruning/static_pruner.py` |
| 5 | Baseline: gradual pruning (GMP) | Ph 1, 5, 6 | `pruning/gradual_pruner.py` |
| 6 | Baseline: lottery ticket | Ph 1, 5, 6 | `pruning/lottery_ticket_pruner.py` |
| 7 | Evaluation metrics (accuracy, FLOPs, params) | Ph 1, 5, 6 | `evaluation/metrics.py` |
| 8 | Latency/throughput metrics (Jetson Nano) | Ph 1, 5, 6 | `evaluation/deployment_metrics.py` |
| 9 | Configuration file with all hyperparameters | Ph 4 | `config.yaml` |
| 10 | Mobile deployment code with TensorRT | Ph 1, 5, 6 | `deployment/tensorrt_runner.py` |
| 11 | Ablation: pruning schedule | Ph 1, 5, 6 | `experiments/ablation_schedule.py` |
| 12 | Ablation: sparsity targets | Ph 1, 5, 6 | `experiments/ablation_sparsity.py` |
| 13 | Ablation: recovery phases | Ph 1, 5, 6 | `experiments/ablation_recovery.py` |
| 14 | Main comparison table (all methods) | Ph 1, 5, 6 | `experiments/main_comparison.py` |
| 15 | ResNet-20/50 architecture | Ph 1, 5, 6 | `models/resnet.py` |
| 16 | VGG-16 architecture | Ph 1, 5, 6 | `models/vgg.py` |
| 17 | Sparsity mask management | Ph 1, 5, 6 | `pruning/mask_manager.py` |
| 18 | Checkpoint save/load with sparsity | Ph 1, 5, 6 | `utils/checkpoint.py` |
| 19 | Logger (wandb/tensorboard) | Ph 1, 5, 6 | `utils/logger.py` |
| 20 | Main entry point script | Ph 1, 5, 6 | `main.py` |
| 21 | Requirements file | Ph 3, 4 | `requirements.txt` |
| 22 | README with usage | Ph 2 | `README.md` |
| 23 | Model export for TensorRT | Ph 1, 5, 6 | `deployment/model_exporter.py` |
| 24 | Dataset download scripts | Ph 1, 5, 6 | `data/download_scripts.sh` |
| 25 | Visualization (sparsity curves, accuracy plots) | Ph 1, 5, 6 | `visualization/plots.py` |

---

## Experiment Count (for plan gate validation)
- **Main experiments**: 3 (CIFAR-10, CIFAR-100, ImageNet tables)
- **Ablation studies**: 3 (schedule, sparsity targets, recovery phases)
- **Deployment benchmarks**: 1 (Jetson Nano latency/throughput)
- **Baseline comparisons**: 4 (static, gradual, lottery ticket, adaptive)
- **Total experiment sections to cover**: 5 unique experiment categories