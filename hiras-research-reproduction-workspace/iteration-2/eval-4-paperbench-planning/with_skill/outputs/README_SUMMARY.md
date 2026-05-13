# HiRAS Planning Summary — Rubric-to-Phase Mapping

## PaperBench Code Development Evaluation
**Task:** Generate planning artifacts (Phases 1-4) for adaptive pruning paper.
**Output directory:** `D:\HiRAS\hiras-research-reproduction-workspace\iteration-2\eval-4-paperbench-planning\with_skill\outputs\`

---

## Rubric Items → HiRAS Phase & Module Mapping

| # | Rubric Item | Phase(s) | Module(s) | Artefact |
|---|-------------|----------|-----------|----------|
| 1 | Implementation of the adaptive pruning algorithm | Ph 1, 5, 6 | `pruning/adaptive_scheduler.py` | `plan.md`, `architecture.md`, `analysis/adaptive_scheduler_analysis.md` |
| 2 | Training loop with pruning schedule integration | Ph 1, 5, 6 | `trainers/pruning_trainer.py` | `plan.md`, `architecture.md` |
| 3 | Data loading and preprocessing (CIFAR-10/100/ImageNet) | Ph 1, 5, 6 | `data/cifar_loader.py`, `data/imagenet_loader.py` | `plan.md`, `architecture.md`, `dependency.md` |
| 4 | Baseline: static pruning | Ph 1, 5, 6 | `pruning/static_pruner.py` | `plan.md`, `architecture.md` |
| 5 | Baseline: gradual pruning (GMP) | Ph 1, 5, 6 | `pruning/gradual_pruner.py` | `plan.md`, `architecture.md` |
| 6 | Baseline: lottery ticket hypothesis | Ph 1, 5, 6 | `pruning/lottery_ticket_pruner.py` | `plan.md`, `architecture.md` |
| 7 | Evaluation metrics (accuracy, FLOPs, parameters) | Ph 1, 5, 6 | `evaluation/metrics.py` | `plan.md`, `architecture.md`, `analysis/metrics_analysis.md` |
| 8 | Latency/throughput metrics (Jetson Nano) | Ph 1, 5, 6 | `evaluation/deployment_metrics.py` | `plan.md`, `architecture.md` |
| 9 | Configuration file with all hyperparameters | Ph 4 | `config.yaml` | `config.yaml` |
| 10 | Mobile deployment code with TensorRT optimization | Ph 1, 5, 6 | `deployment/tensorrt_runner.py` | `plan.md`, `architecture.md`, `analysis/tensorrt_runner_analysis.md` |
| 11 | Ablation: pruning schedule variants | Ph 1, 5, 6 | `experiments/ablation_schedule.py` | `plan.md`, `architecture.md`, `analysis/ablation_schedule_analysis.md` |
| 12 | Ablation: sparsity targets | Ph 1, 5, 6 | `experiments/ablation_sparsity.py` | `plan.md`, `architecture.md`, `analysis/ablation_sparsity_analysis.md` |
| 13 | Ablation: recovery phases | Ph 1, 5, 6 | `experiments/ablation_recovery.py` | `plan.md`, `architecture.md`, `analysis/ablation_recovery_analysis.md` |
| 14 | Main comparison table (all methods × all datasets) | Ph 1, 5, 6 | `experiments/main_comparison.py` | `plan.md`, `architecture.md` |
| 15 | ResNet-20/50 architecture implementations | Ph 1, 5, 6 | `models/resnet.py` | `plan.md`, `architecture.md`, `dependency.md` |
| 16 | VGG-16 architecture implementation | Ph 1, 5, 6 | `models/vgg.py` | `plan.md`, `architecture.md`, `dependency.md` |
| 17 | Sparsity mask management | Ph 1, 5, 6 | `pruning/mask_manager.py` | `plan.md`, `architecture.md` |
| 18 | Checkpoint save/load with sparsity state | Ph 1, 5, 6 | `utils/checkpoint.py` | `plan.md`, `architecture.md` |
| 19 | Logger (wandb/tensorboard) | Ph 1, 5, 6 | `utils/logger.py` | `plan.md`, `architecture.md` |
| 20 | Main entry point script | Ph 1, 5, 6 | `main.py` | `plan.md`, `architecture.md` |
| 21 | Requirements file | Ph 3, 4 | `requirements.txt` | `dependency.md`, `config.yaml` |
| 22 | README with usage instructions | Ph 2 | `README.md` | `architecture.md` |
| 23 | Model export for TensorRT | Ph 1, 5, 6 | `deployment/model_exporter.py` | `plan.md`, `architecture.md` |
| 24 | Dataset download scripts | Ph 1, 5, 6 | `data/download_scripts.sh` | `plan.md`, `architecture.md` |
| 25 | Visualization (sparsity curves, accuracy plots) | Ph 1, 5, 6 | `visualization/plots.py` | `plan.md`, `architecture.md` |

---

## HiRAS Phase Coverage

### Phase 0: Intake ✅
- **Artefact:** `intake_summary.md`
- **Content:** Paper read completely, core contributions identified, experimental setup documented, ambiguities flagged (sparsity adjustment step size, TensorRT precision mode, ImageNet epochs)
- **Gate:** All 25 rubric items identified, 5 experiment categories mapped

### Phase 1: Planning ✅
- **Artefact:** `plan.md`
- **Content:** 5 experiment categories (A: Main Comparisons, B: Ablation Schedule, C: Ablation Sparsity, D: Ablation Recovery, E: Mobile Deployment), inter-experiment dependencies, expected outcomes
- **Gate:** All paper experiments covered (12 main runs + 4 schedule + 5 sparsity + 4 recovery + deployment benchmarks)

### Phase 2: Architecture ✅
- **Artefact:** `architecture.md`
- **Content:** 25-file directory layout, module responsibilities, public interfaces table, entry points, external dependencies
- **Gate:** Every experiment from plan.md maps to ≥1 module, no orphan modules

### Phase 3: Dependency ✅
- **Artefact:** `dependency.md`
- **Content:** requirements.txt, import dependency graph, logic analysis (10 key files with class/function signatures), 7-phase task ordering
- **Gate:** Topologically sorted, no circular imports, all 25 rubric items covered in task list

### Phase 4: Configuration ✅
- **Artefact:** `config.yaml`
- **Content:** Model architectures, dataset paths, training hyperparameters, pruning hyperparameters (all 4 methods), evaluation metrics, deployment config, experiment configs, logging settings
- **Gate:** All paper-stated hyperparameters present, inferred values flagged with `# INFERRED`, unknown values flagged with `# UNKNOWN`

### Phase 5: Analysis (Artifacts Produced) ✅
- **Artefact:** `analysis/components.txt` + 6 analysis files
- **Content:** 
  - `components.txt`: 15 components listed
  - `adaptive_scheduler_analysis.md`: Core algorithm implementation
  - `metrics_analysis.md`: Evaluation utilities
  - `ablation_schedule_analysis.md`: Schedule ablation design
  - `ablation_sparsity_analysis.md`: Sparsity ablation design
  - `ablation_recovery_analysis.md`: Recovery ablation design
  - `tensorrt_runner_analysis.md`: TensorRT deployment design
- **Gate:** All components from architecture listed, every component has analysis file

---

## Files Produced

| File | Phase | Purpose |
|------|-------|---------|
| `intake_summary.md` | 0 | Paper intake with rubric mapping |
| `plan.md` | 1 | Overall experiment plan |
| `architecture.md` | 2 | Software structure design |
| `dependency.md` | 3 | Package deps, import graph, task ordering |
| `config.yaml` | 4 | All hyperparameters |
| `analysis/components.txt` | 5 | Component inventory |
| `analysis/*_analysis.md` | 5 | Per-component implementation analysis |

**Total: 7 planning artifacts covering all 25 Code Development rubric items.**

---

## Key Design Decisions

1. **Modular pruner hierarchy:** All pruners inherit from `BasePruner` — enables swapping methods in experiments
2. **MaskManager separation:** Handles sparsity masks separately from pruner logic — clean separation of concerns
3. **Experiment scripts:** Each ablation is a standalone script — enables selective execution
4. **TensorRT pipeline:** ONNX export first, then TensorRT conversion — standard deployment path
5. **Config-driven:** All hyperparameters in config.yaml — no hardcoded values in code

---

## Unknowns / Ambiguities (Flagged in config.yaml)

| Parameter | Status | Note |
|-----------|--------|------|
| Sparsity adjustment step size | `# INFERRED` | 0.05 assumed, paper says "gradual" without specifying delta |
| ImageNet training epochs | `# INFERRED` | 90 assumed (standard), not explicitly stated |
| TensorRT precision | `# INFERRED` | FP16 assumed, not explicitly stated |
| Lottery ticket rewind epoch | `# PAPER` | 5 epochs stated |
| Recovery interval | `# PAPER` | Every 20 epochs stated |
| Recovery length | `# PAPER` | 10 epochs stated |