import torch
import torch.optim as optim
import pytest
import torchvision.models as models
from model_profiler import Profiler


def get_mobilenetv2():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
    return model

def get_mobilenetv3():
    return models.mobilenet_v3_large()

def get_alexnet():
    return torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=False).eval()

def get_vgg11():
    return torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=False).eval()

def run_profiler_test(model_func):
    model = model_func()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    profiler = Profiler(model, optimizer)
    input_tensor = torch.randn(1, 3, 224, 224)
    profiler.profile(input_tensor)

def test_profiler_with_optimizer():
    run_profiler_test(get_mobilenetv2)
    run_profiler_test(get_mobilenetv3)
    run_profiler_test(get_alexnet)
    run_profiler_test(get_vgg11)

def test_profiler_without_optimizer():
    for model_func in [get_mobilenetv2, get_mobilenetv3, get_alexnet, get_vgg11]:
        model = model_func()
        profiler = Profiler(model)
        input_tensor = torch.randn(1, 3, 224, 224)
        profiler.profile(input_tensor)

def test_profiler_flops_per_layer():
    for model_func in [get_mobilenetv2, get_mobilenetv3, get_alexnet, get_vgg11]:
        model = model_func()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        profiler = Profiler(model, optimizer)
        input_tensor = torch.randn(1, 3, 224, 224)
        profiler.profile(input_tensor, flops_per_layer=True)

def test_profiler_memory_with_depth():
    for model_func in [get_mobilenetv2, get_mobilenetv3, get_alexnet, get_vgg11]:
        model = model_func()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        profiler = Profiler(model, optimizer)
        input_tensor = torch.randn(1, 3, 224, 224)
        profiler.profile(input_tensor, mem_depth=3)

if __name__ == "__main__":
    pytest.main(["-v", __file__])

