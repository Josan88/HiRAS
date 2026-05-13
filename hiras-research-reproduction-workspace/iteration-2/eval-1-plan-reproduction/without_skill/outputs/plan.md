# Reproduction Plan for TinyKAN-HAR Paper

## 1. Scope and Goals

Reproduce all six experiments described in the paper "Explainable Kolmogorov-Arnold Networks for Zero-Shot Human Activity Recognition on TinyML Edge Devices" (TinyKAN-HAR):

1. **Experiment 1 – HAR Performance on Seen Classes**: Train and evaluate TinyKAN-HAR and 7 baseline models (kNN, SVM, RF, 1D-CNN, LSTM, CNN-LSTM, Transformer) on seen classes across 3 datasets (UCI HAR, WISDM, PAMAP2). Metrics: overall accuracy and macro-F1.

2. **Experiment 2 – Zero-Shot and Generalized Zero-Shot Performance**: Evaluate pure ZSL accuracy on unseen classes and gZSL harmonic mean H. Compare TinyKAN-HAR against CNN+ZSL, LSTM+ZSL, Transformer+ZSL baselines.

3. **Experiment 3 – Robustness of Calibration Factor γ**: Sweep γ ∈ {0.0, 0.25, 0.5, 0.75, 1.0} on UCI HAR and report Acc_ZSL, Acc_seen, Acc_unseen, H for each value.

4. **Experiment 4 – Statistical Significance Across Random Seeds**: Train TinyKAN-HAR and Transformer+ZSL with 5 different random seeds on UCI HAR and PAMAP2. Report mean ± std of accuracy and harmonic mean H; compute paired two-sided t-test p-value on H.

5. **Experiment 5 – Ablation Studies**: Run 20 ablation configurations on UCI HAR as specified in Table 7 (Section 5.7). Metrics: HAR metrics (Acc, macro-F1, Acc_ZSL, H) and TinyML metrics (model size kB, RAM kB, latency ms, energy µJ).

6. **Experiment 6 – TinyML Deployment Metrics**: Measure on-device performance (model size, peak RAM, latency, energy) for int8 versions of TinyKAN-HAR, 1D-CNN, CNN-LSTM, and Transformer Tiny on Cortex-M4F target.

## 2. Overall Approach

The reproduction is organized into 5 sequential phases:

- **Phase 1 – Environment & Repository Setup**: Create conda environment, install dependencies, download datasets, and set up project structure.
- **Phase 2 – Data Pipeline**: Implement preprocessing for all three datasets (UCI HAR, WISDM, PAMAP2) including gravity separation, normalization, windowing, and seen/unseen splits.
- **Phase 3 – Core Model Implementation**: Implement the TinyKAN-HAR architecture (KAN layers with spline functions, classification head, ZSL semantic projection, cosine compatibility), plus all baseline models.
- **Phase 4 – Training & Evaluation**: Run all experiments, collect metrics, and produce results tables.
- **Phase 5 – TinyML Export**: Quantize models, generate LUT-based spline approximations, export to TFLM format, and collect on-device metrics.

## 3. Deliverables

The following files will be produced in `outputs/`:

| File | Purpose |
|------|---------|
| `plan.md` | This file — full reproduction plan and experiment inventory |
| `architecture.md` | Detailed architecture specification for TinyKAN-HAR and all baselines |
| `dependency.md` | All required packages, versions, and installation instructions |
| `config.yaml` | All hyperparameters, dataset paths, split ratios, and experiment settings |

## 4. Dataset Access

All three datasets are publicly available from UCI Machine Learning Repository:

- **UCI HAR**: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- **WISDM**: https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset
- **PAMAP2**: https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring

## 5. Experiment Mapping

| Exp # | Description | Key Tables/Figures | Key Metrics |
|-------|-------------|---------------------|-------------|
| 1 | HAR seen-class performance | Table 3 | Acc, macro-F1 |
| 2 | ZSL and gZSL | Table 4 | Acc_ZSL, Acc_seen, Acc_unseen, H |
| 3 | γ sensitivity | Table 5, Figure 2 | Acc_ZSL, Acc_seen, Acc_unseen, H vs γ |
| 4 | Multi-seed robustness | Table 6 | mean±std Acc, mean±std H, p-value |
| 5 | Ablation studies | Table 7, Tables 8-9 | Full metric set per variant |
| 6 | TinyML deployment | Table 10 | Model kB, RAM kB, latency ms, energy µJ |

## 6. Success Criteria

All experiments should reproduce metrics within ±0.5% of reported values for main results, and within ±1.0% for ablation variants. TinyML deployment metrics should match within 10% due to hardware variability.

## 7. Risks and Mitigations

- **Risk**: Text embedding model used for semantic embeddings may differ from paper. *Mitigation*: Use sentence-transformers with `all-MiniLM-L6-v2` as a reasonable proxy; document the choice.
- **Risk**: Random seed differences cause metric variance. *Mitigation*: Run 5 seeds for statistical significance experiments as specified.
- **Risk**: TinyML on-device measurements require hardware. *Mitigation*: Provide software-based latency/energy estimates using CMSIS-NN cycle counts; note hardware dependency.
- **Risk**: WISDM dataset has 18 activities and more complex split. *Mitigation*: Carefully verify the seen/unseen split mapping per Section 3.1.2.