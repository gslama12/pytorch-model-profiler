from flops_profiler import FlopCounterMode
from memory_profiler import profile_memory_cost
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention

class Profiler:
    def __init__(self, model, optimizer=None, results_dir=None, flops_per_layer=False):
        self.model = model
        self.optimizer = optimizer
        self.results_dir = results_dir
        self.flops_per_layer = flops_per_layer

    def profile(self, input):

        # 1.) profile flops
        print("\n - - - - - - FLOPs - - - - - -")
        flop_counter = FlopCounterMode(self.model)  #TODO: enable print flops per layer option

        if self.optimizer:
            with flop_counter:
                optimizer.zero_grad()
                outputs = self.model(input)

                if isinstance(outputs, tuple): #TODO
                    outputs = outputs[0]
                if isinstance(outputs, ImageClassifierOutputWithNoAttention):
                    outputs = outputs.logits

                loss = outputs.sum()
                loss.backward()
                flop_counter.reset_module_tracking_before_optimizer_step() #TODO: print this too
                self.optimizer.step()
        else:
            Warning("Optimizer not specified, profiling without optimizer.step().")
            with flop_counter:
                outputs = self.model(input)

                if isinstance(outputs, tuple): #TODO
                    outputs = outputs[0]
                if isinstance(outputs, ImageClassifierOutputWithNoAttention):
                    outputs = outputs.logits

                loss = outputs.sum()
                loss.backward()

        print("\n")
        # 2.) profile memory
        print("\n - - - - - - MEMORY - - - - - -")
        memory_cost, detailed_info = profile_memory_cost(
            self.model, self.optimizer, input.size(), activation_bits=32, trainable_param_bits=32,
            frozen_param_bits=8, batch_size=1)

        print("memory_cost: " + str(memory_cost / 1e6) + "MB")
        print("param_size: " + str(detailed_info['param_size'] / 1e6) + "MB")
        print("act_size: " + str(detailed_info['act_size'] / 1e6) + "MB")
        print("---------------------------")

        if self.results_dir:
            pass  #TODO
