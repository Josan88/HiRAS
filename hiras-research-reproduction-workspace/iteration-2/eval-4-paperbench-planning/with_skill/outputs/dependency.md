# HiRAS Phase 3: Dependency Modelling

## Package Requirements (requirements.txt)

```
# Core
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0

# Data processing
pillow>=9.0.0
scipy>=1.10.0

# Logging
wandb>=0.15.0
tensorboard>=2.13.0

# Model export
onnx>=1.14.0
onnxruntime-gpu>=1.16.0  # GPU inference

# Mobile deployment (Jetson Nano)
tensorrt>=8.5.0
pycuda>=2022.1          # If running TensorRT on Jetson

# Metrics
thop>=0.1.0            # FLOP counting

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Utilities
pyyaml>=6.0
tqdm>=4.65.0
```

---

## Inter-File Dependencies

```
main.py
├── trainers/pruning_trainer.py
│   ├── pruning/adaptive_scheduler.py
│   │   ├── pruning/mask_manager.py
│   │   └── pruning/base_pruner.py
│   ├── pruning/static_pruner.py
│   │   └── pruning/base_pruner.py
│   ├── pruning/gradual_pruner.py
│   │   └── pruning/base_pruner.py
│   ├── pruning/lottery_ticket_pruner.py
│   │   └── pruning/base_pruner.py
│   ├── models/resnet.py
│   ├── models/vgg.py
│   ├── data/cifar_loader.py
│   ├── data/imagenet_loader.py
│   ├── evaluation/metrics.py
│   │   └── models/resnet.py (for FLOP calc)
│   ├── utils/checkpoint.py
│   └── utils/logger.py
├── evaluation/deployment_metrics.py
│   ├── deployment/tensorrt_runner.py
│   └── deployment/model_exporter.py
└── experiments/main_comparison.py
    ├── trainers/pruning_trainer.py
    ├── evaluation/metrics.py
    └── visualization/plots.py

data/cifar_loader.py
└── torchvision (CIFAR-10/100 datasets)

data/imagenet_loader.py
└── torchvision (ImageFolder)

models/resnet.py
└── torch.nn

models/vgg.py
└── torch.nn

pruning/base_pruner.py
└── torch.nn

pruning/mask_manager.py
└── torch.nn

evaluation/metrics.py
├── torch.nn
└── thop (FLOP counting)

deployment/model_exporter.py
├── torch.nn
├── torch.onnx
└── pruning/mask_manager.py

deployment/tensorrt_runner.py
└── tensorrt (Jetson Nano runtime)

visualization/plots.py
└── matplotlib

utils/checkpoint.py
└── torch

utils/logger.py
└── wandb
```

---

## Logic Analysis by File

### 1. `pruning/base_pruner.py`
```python
class BasePruner(ABC):
    @abstractmethod
    def prune(self, model, sparsity_target, epoch) -> None:
        """Apply pruning to model weights."""
        pass
    
    @abstractmethod
    def get_sparsity(self, model) -> float:
        """Calculate current model sparsity."""
        pass
```

### 2. `pruning/mask_manager.py`
```python
class MaskManager:
    def __init__(self, model):
        self.masks = {}  # layer_name -> binary tensor
        self._create_masks(model)
    
    def apply_masks(self, model):
        """Multiply model weights by masks (in-place)."""
        for name, param in model.named_parameters():
            if name in self.masks:
                param.data *= self.masks[name]
    
    def update_mask(self, name, new_mask):
        """Update mask for specific layer."""
        self.masks[name] = new_mask
    
    def get_sparsity(self, model) -> float:
        """Compute overall sparsity percentage."""
    
    def save_masks(self, path):
        """Save masks to disk."""
    
    def load_masks(self, path, model):
        """Load masks from disk and apply to model."""
```

### 3. `pruning/adaptive_scheduler.py`
```python
class AdaptiveScheduler(BasePruner):
    def __init__(self, initial_sparsity=0.0, target_sparsity=0.8,
                 adjustment_interval=5, adjustment_step=0.05,
                 recovery_interval=20, recovery_epochs=10):
        self.current_sparsity = initial_sparsity
        self.target_sparsity = target_sparsity
        self.adjustment_interval = adjustment_interval
        self.adjustment_step = adjustment_step
        self.recovery_interval = recovery_interval
        self.recovery_epochs = recovery_epochs
        self.in_recovery = False
        self.epoch_count = 0
    
    def prune(self, model, sparsity_target, epoch):
        """Adjust sparsity based on training dynamics."""
        if self.in_recovery:
            if epoch - self.recovery_start >= self.recovery_epochs:
                self.in_recovery = False
            return
        
        if epoch % self.adjustment_interval == 0:
            # Increase sparsity if below target
            self.current_sparsity = min(
                self.current_sparsity + self.adjustment_step,
                self.target_sparsity
            )
            self._perform_pruning(model, self.current_sparsity)
    
    def _perform_pruning(self, model, target_sparsity):
        """Magnitude pruning on weights below current threshold."""
        for name, param in model.named_parameters():
            if 'weight' in name:
                threshold = self._compute_threshold(param, target_sparsity)
                mask = torch.abs(param.data) > threshold
                self.mask_manager.update_mask(name, mask.float())
    
    def enter_recovery(self):
        """Enter recovery phase - stop pruning."""
        self.in_recovery = True
        self.recovery_start = self.epoch_count
    
    def exit_recovery(self):
        """Exit recovery phase - resume pruning."""
        self.in_recovery = False
```

### 4. `pruning/static_pruner.py`
```python
class StaticPruner(BasePruner):
    def __init__(self, target_sparsity):
        self.target_sparsity = target_sparsity
    
    def prune(self, model, sparsity_target, epoch):
        """One-shot pruning: remove target_fraction of smallest weights."""
        target_fraction = sparsity_target
        for name, param in model.named_parameters():
            if 'weight' in name:
                threshold = self._compute_threshold(param, target_fraction)
                mask = torch.abs(param.data) > threshold
                # Apply immediately
```

### 5. `pruning/gradual_pruner.py`
```python
class GradualPruner(BasePruner):
    def __init__(self, start_epoch=0, end_epoch=150, target_sparsity=0.8):
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.target_sparsity = target_sparsity
    
    def prune(self, model, sparsity_target, epoch):
        schedule = self._compute_schedule(epoch)
        for name, param in model.named_parameters():
            if 'weight' in name:
                threshold = self._compute_threshold(param, schedule)
                # Update mask
```

### 6. `pruning/lottery_ticket_pruner.py`
```python
class LotteryTicketPruner(BasePruner):
    def __init__(self, rewind_epoch=5, target_sparsity=0.8):
        self.rewind_epoch = rewind_epoch
        self.target_sparsity = target_sparsity
        self.initial_weights = None
    
    def prune(self, model, sparsity_target, epoch):
        if epoch == self.rewind_epoch:
            self.initial_weights = {name: p.data.clone()
                                    for name, p in model.named_parameters()}
        # Prune based on initial weights, not current
```

### 7. `trainers/pruning_trainer.py`
```python
class PruningTrainer:
    def __init__(self, model, pruner, train_loader, val_loader,
                 optimizer, scheduler, device):
        self.model = model
        self.pruner = pruner
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
    
    def train(self, epochs, log_interval=100):
        for epoch in range(epochs):
            self.pruner.prune(self.model, self.target_sparsity, epoch)
            self._train_epoch(epoch)
            self._validate()
            self.scheduler.step()
    
    def _train_epoch(self, epoch):
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            self.optimizer.step()
```

### 8. `evaluation/metrics.py`
```python
def compute_accuracy(model, data_loader, device, topk=(1,)):
    """Compute top-k accuracy."""

def compute_flops(model, input_shape, device):
    """Count FLOPs using thop."""

def compute_parameter_count(model, count_pruned=False):
    """Count total parameters."""

def compute_sparsity(model):
    """Compute overall sparsity percentage."""
```

### 9. `deployment/model_exporter.py`
```python
class ModelExporter:
    def __init__(self, model, mask_manager):
        self.model = model
        self.mask_manager = mask_manager
    
    def export_to_onnx(self, output_path, input_shape):
        """Export masked model to ONNX."""
        self.mask_manager.apply_masks(self.model)
        self.model.eval()
        dummy_input = torch.randn(*input_shape)
        torch.onnx.export(self.model, dummy_input, output_path)
```

### 10. `deployment/tensorrt_runner.py`
```python
class TensorRTRunner:
    def __init__(self, engine_path):
        self.logger = trt.Logger()
        self.runtime = trt.Runtime(self.logger)
        self.engine = None
        self.context = None
    
    def build_engine(self, onnx_path, fp16=True):
        """Build TensorRT engine from ONNX."""
        builder = trt.Builder(self.logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.logger)
        # Parse ONNX
        # Build engine with FP16 if requested
        self.engine = builder.build_serialized_network(network, config)
        self.context = self.runtime.deserialize_cuda_engine(self.engine)
    
    def run_inference(self, input_data):
        """Run inference and return output."""
        # Allocate buffers
        # Copy input to GPU
        # Execute
        # Copy output back
```

---

## Task Ordering (Topological Sort)

**Phase 1 - Foundation (must implement first):**
1. `data/cifar_loader.py` - no dependencies
2. `data/imagenet_loader.py` - no dependencies
3. `models/resnet.py` - no dependencies
4. `models/vgg.py` - no dependencies
5. `pruning/base_pruner.py` - no dependencies

**Phase 2 - Core Pruning (depends on base_pruner):**
6. `pruning/mask_manager.py` - needs torch.nn
7. `pruning/static_pruner.py` - depends on base_pruner
8. `pruning/gradual_pruner.py` - depends on base_pruner
9. `pruning/lottery_ticket_pruner.py` - depends on base_pruner
10. `pruning/adaptive_scheduler.py` - depends on base_pruner, mask_manager

**Phase 3 - Training (depends on pruning modules):**
11. `evaluation/metrics.py` - depends on models
12. `trainers/base_trainer.py` - no dependencies
13. `trainers/pruning_trainer.py` - depends on trainers, pruning modules, data loaders, metrics
14. `utils/checkpoint.py` - depends on torch
15. `utils/logger.py` - depends on wandb

**Phase 4 - Deployment (depends on models, pruning modules):**
16. `deployment/model_exporter.py` - depends on pruning/mask_manager
17. `deployment/tensorrt_runner.py` - depends on tensorrt
18. `evaluation/deployment_metrics.py` - depends on tensorrt_runner

**Phase 5 - Experiments (depends on trainer and metrics):**
19. `experiments/main_comparison.py` - depends on trainer, metrics
20. `experiments/ablation_schedule.py` - depends on trainer
21. `experiments/ablation_sparsity.py` - depends on trainer
22. `experiments/ablation_recovery.py` - depends on trainer

**Phase 6 - Visualization and Utilities:**
23. `visualization/plots.py` - depends on matplotlib
24. `main.py` - entry point, imports all modules

**Phase 7 - Documentation:**
25. `README.md` - no code dependencies