# Rubric Mapping to Development Phases

## Planning
- Configuration file with hyperparameters

## Architecture
- Implementation of the core variational inference algorithm (BaM)
- Target distribution implementations (Gaussian, non-Gaussian)
- Score function interface
- Evaluation metrics (ELBO, KL divergence, Wasserstein distance)

## Coding
- Training loop with convergence monitoring
- Experiment scripts for each target distribution
- Visualization of learned variational distributions

## Execution
- Training runs without errors on all target distributions
- Evaluation metrics are computed and saved
- Results match paper's reported ELBO values within 5%

## Evaluation
- Figure 4: Learned vs true distributions match qualitatively
- Table 1: ELBO values within 5% of reported values
- Convergence curves match paper's Figure 3