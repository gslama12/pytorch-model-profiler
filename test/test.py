from module_profiler import Profiler
import torch


resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
optimizer = torch.optim.SGD(params=resnet.parameters())
p = Profiler(resnet, optimizer=None, flops_per_layer=True)
p.profile(torch.rand(1, 3, 244, 244))



