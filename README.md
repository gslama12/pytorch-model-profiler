# PyTorch Model Profiler
A simple profiler for PyTorch (vision) models. This was mainly tested on CNNs but also supports other model types (see supported layers).

This profiler combines the following implementations:
- FLOPs Counter: https://gist.github.com/soumith/5f81c3d40d41bb9d08041431c656b233
- Memory Tracker: https://github.com/pytorch/pytorch/blob/main/torch/distributed/_tools/mem_tracker.py

## Profiling capabilities:

### Supported Metrics
- Floating-point operations (FLOPs) for the forward pass
- FLOPs for the backward pass
- Peak memory consumption divided into:
  - Parameters
  - Buffer
  - Gradient Memory
  - Activation Memory
  - Temporary Memory
  - Optimizer State Memory
  - Other

### Supported Layers:
- nn.Linear
- nn.Conv1d, nn.Conv2d, nn.Conv3d
- nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm
- nn.ReLU, nn.ReLU6, nn.LeakyReLU
- nn.Sigmoid, nn.Tanh, Hswish, Hsigmoid

## Usage
### Installation
```
pip install git+https://github.com/gslama12/pytorch-model-profiler
```

### Example
```
import torch
from model_profiler import Profiler

resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18')
optimizer = torch.optim.SGD(params=resnet.parameters())  # optimizer is optional
p = Profiler(resnet, optimizer=optimizer, flops_per_layer=True, mem_depth=None)
p.profile(torch.rand(1, 3, 244, 244))  #specify model input
```

## Further Information
All results are derived analytically, which means they are **independent of the execution machine**. There is no actual hardware profiling here. 
The memory profiler part should work for **all** PyTorch Models, not just for the supporte layers. The peak memory is calculated as the total memory in each category, regardless of the time it occurs.

**Memory Types:**
  - PARAM: for storing model parameters.
  - BUFFER: buffer memory for calculations.
  - GRAD: gradients of the model parameters for backpropagation.
  - ACT: memory used to store the activations of each layer during training.
  - TEMP: additional backward pass memory. Mainly stores gradients of activations.
  - OPT: optimizer state memory
  - OTH: memory that does not belong to any of the other categories.

**Tested Models:**
- MobileNetV1
- MobileNetV2
- MobileNetV3
- ResNet
- WideResNet
- GoogLeNet
- AlexNet
- VGG-Nets

**Also tested with PEFT methods:**
- LoRA
- DoRA
- GaLore