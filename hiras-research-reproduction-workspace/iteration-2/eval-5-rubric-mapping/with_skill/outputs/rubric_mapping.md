# Rubric to HiRAS Phase Mapping

## Mapping Summary

| PaperBench Rubric Item | HiRAS Phase |
|------------------------|-------------|
| Configuration file with hyperparameters | **Phase 4: config.yaml** |
| Experiment scripts for each target distribution | **Phase 1: plan.md** |
| Target distribution implementations (Gaussian, non-Gaussian) | **Phase 1: plan.md** |
| Implementation of the core variational inference algorithm (BaM) | **Phase 1: plan.md** |
| Score function interface | **Phase 1: plan.md** |
| Training loop with convergence monitoring | **Phase 1: plan.md** |
| Evaluation metrics (ELBO, KL divergence, Wasserstein distance) | **Phase 1: plan.md** |
| Visualization of learned variational distributions | **Phase 1: plan.md** |
| Directory layout & module structure | **Phase 2: architecture.md** |
| Module/file responsibilities & inter-module interfaces | **Phase 2: architecture.md** |
| Package versions & inter-file dependencies | **Phase 3: dependency.md** |
| Classes, methods, functions to implement | **Phase 3: dependency.md** |
| Component-by-component logic analysis | **Phase 5: analysis/** |
| Training runs without errors on all target distributions | **Phase 7: execution** |
| Evaluation metrics are computed and saved | **Phase 7: execution** |
| Results match paper's reported ELBO values within 5% | **Phase 7: execution** |
| Figure 4: Learned vs true distributions match qualitatively | **Phase 8: result-match** |
| Table 1: ELBO values within 5% of reported values | **Phase 8: result-match** |
| Convergence curves match paper's Figure 3 | **Phase 8: result-match** |

## Phase-by-Phase Breakdown

### Phase 1: plan.md
- Experiment scripts for each target distribution
- Target distribution implementations (Gaussian, non-Gaussian)
- Implementation of the core variational inference algorithm (BaM)
- Score function interface
- Training loop with convergence monitoring
- Evaluation metrics (ELBO, KL divergence, Wasserstein distance)
- Visualization of learned variational distributions

### Phase 2: architecture.md
- Directory layout
- Module/file responsibilities and interfaces between modules

### Phase 3: dependency.md
- Package versions (requirements.txt format)
- Inter-file dependencies (which module imports which)

### Phase 4: config.yaml
- Configuration file with hyperparameters

### Phase 5: analysis/
- Component-by-component logic analysis (per-component `_analysis.md` files)

### Phase 6: code
*(No rubric items map directly here — this is the code generation phase itself)*

### Phase 7: execution
- Training runs without errors on all target distributions
- Evaluation metrics are computed and saved
- Results match paper's reported ELBO values within 5%

### Phase 8: result-match
- Figure 4: Learned vs true distributions match qualitatively
- Table 1: ELBO values within 5% of reported values
- Convergence curves match paper's Figure 3

## Notes

- **Execution vs. Result-Match**: Phase 7 verifies that experiments run without crashing and metrics are produced. Phase 8 verifies that the produced results qualitatively or quantitatively match the paper's specific figures and tables.
- **Analysis before code**: All implementation must be planned in Phase 5 analysis before Phase 6 code generation begins.
- **Config before code**: All hyperparameters must be documented in Phase 4 config before any code references them.