########################################################################################################################
# - - - - - - - - - - - - - - - - - - - - - - - - MEMORY COST PROFILER - - - - - - - - - - - - - - - - - - - - - - - -
# Created by the MIT Han Lab (see source).
#
# Source: https://github.com/mit-han-lab/tinyml/blob/master/tinytl/tinytl/utils/memory_cost_profiler.py
#
# The original profiler was adapted to support GaLore.
#
########################################################################################################################

import copy
import torch
import torch.nn as nn
from ofa.utils import Hswish, Hsigmoid, MyConv2d
from ofa.utils.layers import ResidualBlock
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.mobilenetv2 import InvertedResidual
from galore_torch import GaLoreAdamW

__all__ = ['count_model_size', 'count_activation_size', 'profile_memory_cost']


def count_model_size(net, trainable_param_bits=32, frozen_param_bits=8, print_log=True):
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'Total number of trainable parameters: {total_params}')

    frozen_param_bits = 32 if frozen_param_bits is None else frozen_param_bits

    trainable_param_size = 0
    frozen_param_size = 0
    for p in net.parameters():
        if p.requires_grad:
            trainable_param_size += trainable_param_bits / 8 * p.numel()
        else:
            frozen_param_size += frozen_param_bits / 8 * p.numel()
    model_size = trainable_param_size + frozen_param_size
    if print_log:
        print('Total Model Size: %d' % model_size,
              '\tTrainable Parameters Size: %d (data bits %d)' % (trainable_param_size, trainable_param_bits),
              '\tFrozen Parameters Size: %d (data bits %d)' % (frozen_param_size, frozen_param_bits))
    # Byte
    return model_size


def count_activation_size(net, optimizer, input_size=(1, 3, 224, 224), require_backward=True, activation_bits=32):
    use_galore = False
    galore_rank = None
    if isinstance(optimizer, GaLoreAdamW):
        use_galore = True
        galore_rank = optimizer.param_groups[1]['rank']

    act_byte = activation_bits / 8
    model = copy.deepcopy(net)

    def count_galore_grad_activation_size(weight):
        # convert any input tensor to 2D
        if len(weight.shape) > 2:
            weight = weight.view(weight.shape[0], -1)
        elif len(weight.shape) == 1:
            weight = weight.view(1, -1)
        else:
            Exception("Unexpected weight shape: ", weight.shape)
        # apply gradient formula from GaLore paper: optimizer state memory = mr + 2n*r
        shape = weight.size()
        m_ = shape[0]
        n_ = shape[1]
        return torch.Tensor([m_ * galore_rank + 2 * n_ * galore_rank * act_byte])

    def count_convNd(m, x, y):
        # count activation size required by backward
        if m.weight is not None and m.weight.requires_grad:
            if use_galore:
                m.grad_activations = count_galore_grad_activation_size(m.weight) # bytes

            else:
                m.grad_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes

        else:
            m.grad_activations = torch.Tensor([0])
        # temporary memory footprint required by inference
        m.tmp_activations = torch.Tensor([x[0].numel() * act_byte + y.numel() * act_byte // m.groups])  # bytes

    def count_linear(m, x, y):
        # count activation size required by backward
        if m.weight is not None and m.weight.requires_grad:
            if use_galore:
                m.grad_activations = count_galore_grad_activation_size(m.weight)
            else:
                m.grad_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes
        else:
            m.grad_activations = torch.Tensor([0])
        # temporary memory footprint required by inference
        m.tmp_activations = torch.Tensor([x[0].numel() * act_byte + y.numel() * act_byte])  # bytes

    def count_bn(m, x, _):
        # count activation size required by backward
        if m.weight is not None and m.weight.requires_grad:
            if use_galore:
                m.grad_activations = count_galore_grad_activation_size(m.weight)  # bytes
            else:
                m.grad_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes
        else:
            m.grad_activations = torch.Tensor([0])
        # temporary memory footprint required by inference
        m.tmp_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes

    def count_relu(m, x, _):
        # count activation size required by backward
        m.grad_activations = torch.Tensor([0])
        # temporary memory footprint required by inference
        m.tmp_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes

    def count_smooth_act(m, x, _):  # not used for MobileNet
        # count activation size required by backward
        if require_backward:
            if use_galore:
                m.grad_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes
            else:
                m.grad_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes
        else:
            m.grad_activations = torch.Tensor([0])
        # temporary memory footprint required by inference
        m.tmp_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes

    def add_hooks(m_):
        if len(list(m_.children())) > 0:
            return

        m_.register_buffer('grad_activations', torch.zeros(1))
        m_.register_buffer('tmp_activations', torch.zeros(1))

        if type(m_) in [nn.Conv1d, nn.Conv2d, nn.Conv3d, MyConv2d]:
            fn = count_convNd
        elif type(m_) in [nn.Linear]:
            fn = count_linear
        elif type(m_) in [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm]:
            fn = count_bn
        elif type(m_) in [nn.ReLU, nn.ReLU6, nn.LeakyReLU]:
            fn = count_relu
        elif type(m_) in [nn.Sigmoid, nn.Tanh, Hswish, Hsigmoid]:
            fn = count_smooth_act
        else:
            fn = None

        if fn is not None:
            _handler = m_.register_forward_hook(fn)

    model.eval()
    model.apply(add_hooks)

    x = torch.zeros(input_size).to(model.parameters().__next__().device)
    with torch.no_grad():
        model(x)

    memory_info_dict = {
        'peak_activation_size': torch.zeros(1),
        'grad_activation_size': torch.zeros(1),
        'residual_size': torch.zeros(1),
    }

    for m in model.modules():
        if len(list(m.children())) == 0:
            def new_forward(_module, *args, **kwargs):
                def lambda_forward(_x, *args, **kwargs):
                    debug_dict = {}
                    debug_dict["tmp"] = _module.tmp_activations
                    debug_dict["grd"] = memory_info_dict['grad_activation_size']
                    debug_dict["res"] = memory_info_dict['residual_size']


                    current_act_size = _module.tmp_activations + memory_info_dict['grad_activation_size'] + \
                                       memory_info_dict['residual_size']
                    memory_info_dict['peak_activation_size'] = max(
                        current_act_size, memory_info_dict['peak_activation_size']
                    )
                    memory_info_dict['grad_activation_size'] += _module.grad_activations
                    return _module.old_forward(_x, *args, **kwargs)

                return lambda_forward

            m.old_forward = m.forward
            m.forward = new_forward(m)

        if (isinstance(m, ResidualBlock) and m.shortcut is not None) or \
                (isinstance(m, InvertedResidual) and m.use_res_connect) or \
                type(m) in [BasicBlock, Bottleneck]:
            def new_forward(_module):
                def lambda_forward(_x):
                    memory_info_dict['residual_size'] = _x.numel() * act_byte
                    result = _module.old_forward(_x)
                    memory_info_dict['residual_size'] = 0
                    return result

                return lambda_forward

            m.old_forward = m.forward
            m.forward = new_forward(m)

    with torch.no_grad():
        model(x)

    return memory_info_dict['peak_activation_size'].item(), memory_info_dict['grad_activation_size'].item()


def profile_memory_cost(net, optimizer, input_size=(1, 3, 224, 224), require_backward=True,
                        activation_bits=32, trainable_param_bits=32, frozen_param_bits=8, batch_size=8):
    param_size = count_model_size(net, trainable_param_bits, frozen_param_bits, print_log=True)
    activation_size, _ = count_activation_size(net, optimizer, input_size, require_backward, activation_bits)
    memory_cost = activation_size * batch_size + param_size
    return memory_cost, {'param_size': param_size, 'act_size': activation_size}
