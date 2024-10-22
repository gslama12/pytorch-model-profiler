########################################################################################################################
# - - - - - - - - - - - - - - - - - - - - - - - - - - FLOPs COUNTER - - - - - - - - - - - - - - - - - - - - - - - - - -
# This script was slightly modified but mostly taken from the cited source.
# Source: https://gist.github.com/soumith/5f81c3d40d41bb9d08041431c656b233
#         https://dev-discuss.pytorch.org/t/the-ideal-pytorch-flop-counter-with-torch-dispatch/505
#
########################################################################################################################

import torch
from torch.utils._pytree import tree_map
from typing import List, Any
from numbers import Number
from tabulate import tabulate
from collections import defaultdict
from torch.utils._python_dispatch import TorchDispatchMode
from peft import PeftModel


aten = torch.ops.aten


def get_shape(i):
    return i.shape


def prod(x):
    res = 1
    for i in x:
        res *= i
    return res

def matmul_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for matmul. for A = (M,N) and B = (N,L) we need M*N*L multiplications and M*L*(N-1) additions.
    Calculations are based on this: https://mediatum.ub.tum.de/doc/625604/625604
    """
    # Inputs contains the shapes of the two input matrices.
    input_shapes = [get_shape(v) for v in inputs]
    assert len(input_shapes) == 2, input_shapes
    assert input_shapes[0][-1] == input_shapes[1][-2], input_shapes

    m = input_shapes[0][0]
    n = input_shapes[0][-1]
    l = input_shapes[-1][-1]

    flops = 2 * m * n * l - m * l
    return flops

def addmm_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for fully connected layers (nn.Linear).
    The function torch.addmm(I, A, B) multiplies A = (M,N) and B = (N,L) and adds I = (J, K) to the result.
    Calculations are based on this: https://mediatum.ub.tum.de/doc/625604/625604
    """
    # inputs is a tuple(I, A, B).
    input_shapes = [get_shape(v) for v in inputs[1:3]]
    assert len(input_shapes[0]) == 2, input_shapes[0]
    assert len(input_shapes[1]) == 2, input_shapes[1]
    assert input_shapes[0][-1] == input_shapes[1][0], input_shapes
    assert inputs[0].dim() <= 1  # for our case k will usually be 1 here because I = (B,)

    m = input_shapes[0][0]
    n = input_shapes[0][-1]
    l = input_shapes[1][-1]
    j = inputs[0].shape[0]

    flops = 2 * m * n * l - m * l + j * n
    return flops


def bmm_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for the bmm operation.
    NOTE: This function was not used in our case.
    """
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two tensor.
    assert len(inputs) == 2, len(inputs)
    input_shapes = [get_shape(v) for v in inputs]
    n, c, t = input_shapes[0]
    d = input_shapes[-1][-1]
    macs = n * c * t * d
    return 2 * macs


def conv_flop_count(
    x_shape: List[int],
    w_shape: List[int],
    out_shape: List[int],
    transposed: bool = False,
    bias: bool = False,
) -> Number:
    """
    Count flops for convolution. Note only multiplication is
    counted. Computation for addition and bias is ignored.
    Flops for a transposed convolution are calculated as
    flops = (x_shape[2:] * prod(w_shape) * batch_size).
    Args:
        x_shape (list(int)): The input shape before convolution.
        w_shape (list(int)): The filter shape.
        out_shape (list(int)): The output shape after convolution.
        transposed (bool): is the convolution transposed
        bias (bool): is bias considered
    Returns:
        int: the number of flops
    """
    batch_size = x_shape[0]
    conv_shape = (x_shape if transposed else out_shape)[2:]
    macs = batch_size * prod(w_shape) * prod(conv_shape)
    if bias is not None:
        macs += batch_size * out_shape[1] * prod(out_shape[2:])
    return 2 * macs


def conv_forward_flop(inputs: List[Any], outputs: List[Any]) -> int:
    """
    Count flops for convolution.
    """
    x, w, b = inputs[:3]
    x_shape, w_shape, out_shape = (get_shape(x), get_shape(w), get_shape(outputs[0]))
    transposed = inputs[6]
    return conv_flop_count(x_shape, w_shape, out_shape, transposed=transposed, bias=b)


def transpose_shape(shape):
    return [shape[1], shape[0]] + list(shape[2:])


def conv_backward_flop(inputs: List[Any], outputs: List[Any]):
    """
    Count flops for the backwards pass of convolutional layers.
    Explanation see here: https://dev-discuss.pytorch.org/t/the-ideal-pytorch-flop-counter-with-torch-dispatch/505/15
    """
    grad_out_shape, x_shape, w_shape = [get_shape(i) for i in inputs[:3]]
    output_mask = inputs[-1]
    fwd_transposed = inputs[7]
    flop_count = 0

    if output_mask[0]:  # compute weight gradient
        grad_input_shape = get_shape(outputs[0])
        flop_count += conv_flop_count(grad_out_shape, w_shape, grad_input_shape, not fwd_transposed)
    if output_mask[1]:  # compute input gradient
        grad_weight_shape = get_shape(outputs[1])
        flop_count += conv_flop_count(transpose_shape(x_shape), grad_out_shape, grad_weight_shape, fwd_transposed)

    if output_mask[2]:  # compute bias gradients = true (SKIPPED)
        pass  # TODO: can i add this too?

    return flop_count


def add_flops(inputs: List[Any], outputs: List[Any]) -> Number:
    return inputs[0].numel() * 2  #add always has does c = a + scaling_factor * b


def mul_flops(inputs: List[Any], outputs: List[Any]) -> Number:
    return outputs[0].numel()


flop_mapping = {
    aten.mm: matmul_flop,
    aten.matmul: matmul_flop,
    aten.addmm: addmm_flop,
    aten.bmm: bmm_flop,
    aten.convolution: conv_forward_flop,
    aten.conv2d: conv_forward_flop,
    aten._convolution: conv_forward_flop,
    aten.convolution_backward: conv_backward_flop,
    aten.add: add_flops,
    aten.add_: add_flops,
    aten.mul: mul_flops,
    aten.mul_: mul_flops,
}


def normalize_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x


class FlopCounterMode(TorchDispatchMode):
    def __init__(self, model = None, print_flops_per_layer=False):
        self.flop_counts = defaultdict(lambda: defaultdict(int))
        self.parents = ['Global']
        self.print_flops_per_layer = print_flops_per_layer
        if model is not None:
            if isinstance(model, PeftModel):
                module_dict = dict(model.base_model.model.named_children()).items()
            else:
                module_dict = dict(model.named_children()).items()
            # here the hooks are registered. this can be adapted to achieve finer-grained profiling results.
            for name, module in module_dict:
                module.register_forward_pre_hook(self.enter_module(name))
                module.register_forward_hook(self.exit_module(name))

    def enter_module(self, name):
        def f(module, inputs):
            self.parents.append(name)
            inputs = normalize_tuple(inputs)
            out = self.create_backwards_pop(name)(*inputs)
            return out

        return f

    def exit_module(self, name):
        def f(module, inputs, outputs):
            assert(self.parents[-1] == name)
            self.parents.pop()
            outputs = normalize_tuple(outputs)
            return self.create_backwards_push(name)(*outputs)
        return f

    def create_backwards_push(self, name):
        class PushState(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args):
                args = tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, args)
                if len(args) == 1:
                    return args[0]
                return args

            @staticmethod
            def backward(ctx, *grad_outs):
                self.parents.append(name)
                return grad_outs

        return PushState.apply

    def create_backwards_pop(self, name):
        class PopState(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args):
                args = tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, args)
                if len(args) == 1:
                    return args[0]
                return args

            @staticmethod
            def backward(ctx, *grad_outs):
                assert(self.parents[-1] == name)
                self.parents.pop()
                return grad_outs

        return PopState.apply

    def __enter__(self):
        self.flop_counts.clear()
        super().__enter__()

    def __exit__(self, *args):
        # print global flops
        table_data = []
        for k, v in self.flop_counts['Global'].items():
            mod_flops = round(v)
            table_data.append([str(k), f"{mod_flops:,} FLOPs"])
        flops_total = round(sum(self.flop_counts['Global'].values()))
        table_data.append(["TOTAL", f"{flops_total:,} FLOPs"])
        table_headers = ["Global", "FLOPs"]
        print(tabulate(table_data, table_headers, stralign="right", colalign=("center", "center")))
        print()

        # print flops per layer
        if self.print_flops_per_layer:
            for mod in self.flop_counts.keys():
                if mod != 'Global':
                    table_data = []
                    for k,v in self.flop_counts[mod].items():
                        mod_flops = round(v)
                        table_data.append([str(k), f"{mod_flops:,} FLOPs"])
                    flops_total = round(sum(self.flop_counts[mod].values()))
                    table_data.append(["TOTAL", f"{flops_total:,} FLOPs"])
                    table_headers = [mod, "FLOPs"]
                    print(tabulate(table_data, table_headers, stralign="right", colalign=("center", "center")))
                    print()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs else {}

        out = func(*args, **kwargs)
        func_packet = func._overloadpacket
        if func_packet in flop_mapping:
            flop_count = flop_mapping[func_packet](args, normalize_tuple(out))
            for par in self.parents:
                self.flop_counts[par][func_packet] += flop_count

        return out

    def reset_module_tracking_before_optimizer_step(self):
        # reset the module tracking to capture flops during the optimizer.step()
        self.parents = ['Global', 'optimizer']
