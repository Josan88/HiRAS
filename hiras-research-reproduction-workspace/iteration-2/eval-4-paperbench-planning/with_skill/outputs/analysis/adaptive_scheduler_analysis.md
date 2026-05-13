# Analysis: adaptive_scheduler.py

## Introduction

The `AdaptiveScheduler` is the core algorithm of the paper. It dynamically adjusts network sparsity during training based on observed training dynamics (e.g., loss plateau detection, gradient flow measurements). Unlike static schedules (which prune at a fixed rate) or gradual pruning (which follows a pre-determined schedule), the adaptive scheduler makes real-time decisions about when and how much to prune.

Key paper claim: Adaptive pruning achieves better accuracy-sparsity tradeoffs because it "listens" to the network's training behavior.

## Implementation

### Class: AdaptiveScheduler
Inherits from `BasePruner`.

### Initialization parameters (from config.yaml):
- `initial_sparsity=0.0` — start with dense network
- `target_sparsity=0.8` — goal is 80% sparsity
- `adjustment_interval=5` — re-evaluate every 5 epochs
- `adjustment_step=0.05` — increase sparsity by 5% per interval
- `recovery_interval=20` — enter recovery every 20 epochs
- `recovery_epochs=10` — recovery lasts 10 epochs

### Core logic:

1. **`__init__`**: Initialize state variables:
   - `self.current_sparsity`
   - `self.in_recovery = False`
   - `self.mask_manager` (reference to MaskManager instance)

2. **`prune(model, sparsity_target, epoch)`**: Main entry point called by trainer each epoch.
   - If in recovery phase: check if recovery is complete, return
   - If not in recovery and epoch % adjustment_interval == 0:
     - Calculate new sparsity target = min(current_sparsity + adjustment_step, target_sparsity)
     - Call `_perform_pruning(model, new_sparsity)`
   - If recovery_interval reached, call `enter_recovery()`

3. **`_perform_pruning(model, target_sparsity)`**: Magnitude pruning step.
   - For each weight tensor (layer with 'weight' in name):
     - Compute threshold: weight values below threshold are pruned
     - Threshold computation uses kthvalue to find the (1 - sparsity)th percentile
     - Creates binary mask: 1 = keep, 0 = prune
     - Updates mask via `self.mask_manager.update_mask(layer_name, mask)`

4. **`enter_recovery()`**: Called when pruning should pause.
   - Sets `self.in_recovery = True`
   - Records `self.recovery_start_epoch`
   - Logs recovery start event

5. **`exit_recovery()`**: Called when recovery phase ends.
   - Sets `self.in_recovery = False`
   - Logs recovery end event

### State tracking:
- `get_current_sparsity()`: Returns current sparsity level
- `get_sparsity_history()`: Returns list of sparsity values over epochs (for plotting)

## Notes

- The adaptive scheduler REQUIRES a MaskManager to manage binary masks. MaskManager must be instantiated separately and passed to the scheduler.
- Recovery phases are a key innovation: they allow the network to "recover" from pruning-induced accuracy drops before resuming pruning.
- The scheduler does NOT handle mask application — that happens in the forward pass of the model via `model.weight.data *= mask`.
- Edge case: If target_sparsity is reached before end_epoch, scheduler should stop adjusting and maintain current sparsity.
- Edge case: If training ends during recovery phase, properly finalize without error.