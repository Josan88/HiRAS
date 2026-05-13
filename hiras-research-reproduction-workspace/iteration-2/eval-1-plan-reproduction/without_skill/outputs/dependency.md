# Dependency Specification for TinyKAN-HAR Reproduction

## 1. Python Environment

- **Python**: 3.10.x (recommended: 3.10.12)
- **Conda**: Miniconda3 or Anaconda installed locally

## 2. Core Deep Learning Framework

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.1.0 | Core deep learning framework |
| PyTorch Lightning | 2.1.0 | Training loop abstraction |
| torchvision | 0.16.0 | Data utilities (optional) |

## 3. Data Processing & Scientific Computing

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | 1.24.3 | Array operations, random |
| scipy | 1.11.3 | Signal processing, interpolation |
| pandas | 2.0.3 | Tabular data handling |
| scikit-learn | 1.3.2 | Classical ML baselines, metrics |
| tqdm | 4.66.1 | Progress bars |

## 4. Dataset Downloads

| Package | Version | Purpose |
|---------|---------|---------|
| gdown | 4.7.1 | Download UCI HAR from Google Drive |

## 5. Semantic Embeddings

| Package | Version | Purpose |
|---------|---------|---------|
| sentence-transformers | 2.2.2 | Text embeddings for ZSL |
| torch | (already listed) | Backend for sentence-transformers |

## 6. Experiment Tracking & Reproducibility

| Package | Version | Purpose |
|---------|---------|---------|
| wandb | 0.16.0 | Experiment tracking (optional) |
| tensorboard | 2.14.0 | Training visualizations |
| hydra-core | 1.3.2 | Configuration management |

## 7. TinyML Deployment

| Package | Version | Purpose |
|---------|---------|---------|
| tensorflow | 2.14.0 | TFLite model conversion |
| tinymlgen | 0.0.4 | TFLM C++ code generation |

## 8. Visualization

| Package | Version | Purpose |
|---------|---------|---------|
| matplotlib | 3.8.0 | Plots, spline visualization |
| seaborn | 0.12.2 | Statistical visualization |

## 9. Statistical Analysis

| Package | Version | Purpose |
|---------|---------|---------|
| scipy | 1.11.3 | Paired t-test, statistical tests |

## 10. Unit Testing

| Package | Version | Purpose |
|---------|---------|---------|
| pytest | 7.4.3 | Testing framework |
| pytest-cov | 4.1.0 | Coverage reports |

## 11. File Operations

| Package | Version | Purpose |
|---------|---------|---------|
| pyyaml | 6.0.1 | Read/write YAML configs |
| json | (stdlib) | JSON operations |
| pickle | (stdlib) | Model serialization |

## 12. Complete conda environment.yml

```yaml
name: tinykan-har
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.10.12
  - pip
  - pip:
    - torch==2.1.0
    - torchvision==0.16.0
    - pytorch-lightning==2.1.0
    - numpy==1.24.3
    - scipy==1.11.3
    - pandas==2.0.3
    - scikit-learn==1.3.2
    - tqdm==4.66.1
    - gdown==4.7.1
    - sentence-transformers==2.2.2
    - wandb==0.16.0
    - tensorboard==2.14.0
    - hydra-core==1.3.2
    - tensorflow==2.14.0
    - matplotlib==3.8.0
    - seaborn==0.12.2
    - pytest==7.4.3
    - pytest-cov==4.1.0
    - pyyaml==6.0.1
```

## 13. Dataset Download Scripts

### UCI HAR
```bash
# URL: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
# Direct link: https://www.kaggle.com/datasets/m瞧onur/human-activity-recognition-with-smartphones  # alternative
# Manual download required from UCI website
# Expected location: data/raw/UCI HAR/
```

### WISDM
```bash
# URL: https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset
# Manual download required from UCI website
# Expected location: data/raw/WISDM/
```

### PAMAP2
```bash
# URL: https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring
# Manual download required from UCI website
# Expected location: data/raw/PAMAP2/
```

## 14. Environment Setup Commands

```bash
# Create conda environment
conda create -n tinykan-har python=3.10.12 -y
conda activate tinykan-har

# Install PyTorch (CPU version for speed, GPU if available)
# CPU only:
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

# Or GPU version (if CUDA available):
pip install torch==2.1.0 torchvision==0.16.0

# Install remaining packages
pip install pytorch-lightning==2.1.0 numpy==1.24.3 scipy==1.11.3 pandas==2.0.3 scikit-learn==1.3.2 tqdm==4.66.1
pip install gdown==4.7.1 sentence-transformers==2.2.2 wandb==0.16.0 tensorboard==2.14.0 hydra-core==1.3.2
pip install tensorflow==2.14.0 matplotlib==3.8.0 seaborn==0.12.2 pytest==7.4.3 pytest-cov==4.1.0 pyyaml==6.0.1

# Verify installation
python -c "import torch; import sklearn; import scipy; print('All packages OK')"
```

## 15. Directory Structure

```
D:\HiRAS\
├── data/
│   ├── raw/                    # Raw dataset files
│   │   ├── UCI HAR/
│   │   ├── WISDM/
│   │   └── PAMAP2/
│   ├── processed/              # Preprocessed numpy arrays
│   └── splits/                 # Train/val/test CSV splits
├── src/                        # Source code
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── zsl/
│   ├── explainability/
│   ├── tinyml/
│   └── evaluation/
├── outputs/                    # Results and figures
│   ├── tables/
│   ├── plots/
│   └── models/
├── configs/                    # Hydra config files
├── logs/                       # Training logs
├── tests/                      # Unit tests
├── plan.md
├── architecture.md
├── dependency.md
└── config.yaml
```

## 16. Hardware Requirements

### Minimum (CPU training)
- 8 GB RAM
- 20 GB free disk space
- Multi-core CPU (4+ cores recommended)

### Recommended (GPU training)
- 16 GB RAM
- NVIDIA GPU with 8+ GB VRAM
- CUDA 11.8+

### TinyML Deployment (Optional)
- ARM Cortex-M4F MCU or QEMU emulation
- 256 kB flash, 64 kB SRAM

## 17. Environment Variables

```bash
# Set random seeds for reproducibility
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# Data paths (override in config.yaml)
export DATA_ROOT="D:/HiRAS/data"
export OUTPUT_ROOT="D:/HiRAS/outputs"
```

## 18. Optional Cloud Dependencies

If using Weights & Biases for experiment tracking:
```bash
wandb login  # Authenticate with wandb.ai API key
```

## 19. Version Locking Rationale

Specific versions are pinned to ensure reproducibility. Key constraints:
- PyTorch 2.1.0 for compatibility with PyTorch Lightning 2.1.0
- NumPy 1.24.3 to avoid scipy/pandas breaking changes
- scikit-learn 1.3.2 for consistent random forest behavior
- sentence-transformers 2.2.2 for consistent text embeddings

## 20. Verification Commands

```bash
# Verify all packages are importable
python -c "
import torch
import pytorch_lightning
import numpy
import scipy
import pandas
import sklearn
import tqdm
import gdown
import sentence_transformers
import wandb
import tensorboard
import hydra
import tensorflow
import matplotlib
import seaborn
import pytest
import yaml
print('All packages verified OK')
"

# Verify CUDA availability (if GPU training)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Verify dataset paths
python -c "
from pathlib import Path
data_root = Path('D:/HiRAS/data/raw')
for ds in ['UCI HAR', 'WISDM', 'PAMAP2']:
    path = data_root / ds
    print(f'{ds}: {\"EXISTS\" if path.exists() else \"MISSING\"}')"
```