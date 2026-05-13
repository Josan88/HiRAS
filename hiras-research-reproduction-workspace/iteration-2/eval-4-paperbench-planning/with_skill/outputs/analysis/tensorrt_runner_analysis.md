# Analysis: deployment/tensorrt_runner.py

## Introduction

This module handles TensorRT optimization and inference benchmarking on Jetson Nano (or compatible GPU). Used for Category E experiments (mobile deployment benchmarks).

## Implementation

### Class: TensorRTRunner

### Initialization
```python
def __init__(self, engine_path=None):
    self.logger = trt.Logger(trt.Logger.WARNING)
    self.runtime = trt.Runtime(self.logger)
    self.engine = None
    self.context = None
    self.input_shape = (3, 224, 224)  # ImageNet shape
```

### Core Methods

1. **`build_engine(onnx_path, fp16=True)`**:
   - Create TensorRT builder and network
   - Parse ONNX model
   - Configure precision (FP16 for Jetson Nano)
   - Build and serialize engine
   - Save engine to disk
   - Load engine for inference

2. **`run_inference(input_tensor)`**:
   - Allocate input/output buffers on GPU
   - Copy input to GPU memory
   - Execute inference
   - Copy output back to CPU
   - Return output tensor

3. **`benchmark(batch_size=1, num_runs=1000, warmup=100)`**:
   - Run warmup passes
   - Time `num_runs` inference passes
   - Return dict: mean_latency_ms, std_latency_ms, throughput_imgs_per_sec

### TensorRT Conversion Pipeline

```
PyTorch Model → Prune weights → Apply masks → Export to ONNX → Convert to TensorRT
```

### Sparsity Handling for TensorRT
- Apply masks BEFORE export (zero out pruned weights)
- ONNX export will omit zeroed weights
- TensorRT engine will be smaller and faster

## Notes

- Requires `tensorrt` Python package
- Requires `pycuda` for GPU memory management
- Only runs on Jetson Nano or compatible CUDA device
- FP16 precision provides ~2x speedup vs FP32
- INT8 quantization possible but requires calibration dataset (not in scope)