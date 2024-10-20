from module_profiler import Profiler
import torch
import torchvision

#choose model (and optimizer)
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
#model = torchvision.models.mobilenet_v3_large()
#model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=False)
#model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=False)
#model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=False)

optimizer = torch.optim.SGD(params=model.parameters())

print(model)
# profile
p = Profiler(model, optimizer=optimizer, flops_per_layer=True)
p.profile(torch.rand(1, 3, 244, 244))



