# PyTorch Model Profiler
A simple profiler for PyTorch models.

This combines the following projects:
- Torch dispatch FLOPs counter: https://gist.github.com/soumith/5f81c3d40d41bb9d08041431c656b233
- Memory cost profiler: https://github.com/mit-han-lab/tinyml/blob/master/tinytl/tinytl/utils/memory_cost_profiler.py

**Profiling capabilities:**
- Floating-point operations (FLOPs) for the forward pass
- FLOPs for the backward pass
- Peak memory consumption

**Supported Layers:**
- nn.Linear
- nn.ConvNd
- TODO..

## Usage
### Installation
```
pip install git+https://github.com/gslama12/pytorch-model-profiler
```

### Example
```
import torch
from module_profiler import Profiler

resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18')
optimizer = torch.optim.SGD(params=resnet.parameters()) # optimizer is optional
p = Profiler(resnet, optimizer=optimizer, flops_per_layer=True)
p.profile(torch.rand(1, 3, 244, 244)) #speify model input
```


