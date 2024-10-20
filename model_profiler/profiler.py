from profilers import flops_profiler, memory_profiler
from tabulate import tabulate
import warningss
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention


class Profiler:
    def __init__(self, model, optimizer=None, results_dir=None, flops_per_layer=False,
                 activation_bits=32, trainable_param_bits=32, frozen_param_bits=8):
        self.model = model.train()
        self.optimizer = optimizer
        self.results_dir = results_dir
        self.flops_per_layer = flops_per_layer
        self.activation_bits = activation_bits
        self.trainable_param_bits = trainable_param_bits
        self.frozen_param_bits = frozen_param_bits

    def profile(self, input_tensor):

        # 1.) profile flops
        print("\n - - - - - - - - - - - - - - FLOPs - - - - - - - - - - - - - -")
        flop_counter = flops_profiler.FlopCounterMode(self.model, print_flops_per_layer=self.flops_per_layer)

        if self.optimizer:
            with flop_counter:
                self.optimizer.zero_grad()
                outputs = self.model(input_tensor)

                if isinstance(outputs, tuple): #TODO
                    outputs = outputs[0]
                if isinstance(outputs, ImageClassifierOutputWithNoAttention):
                    outputs = outputs.logits

                loss = outputs.sum()
                loss.backward()
                flop_counter.reset_module_tracking_before_optimizer_step() #TODO: print this too
                self.optimizer.step()
        else:
            warnings.warn("Optimizer not specified, profiling without optimizer.step().", UserWarning)
            with flop_counter:
                outputs = self.model(input_tensor)

                if isinstance(outputs, tuple): #TODO
                    outputs = outputs[0]
                if isinstance(outputs, ImageClassifierOutputWithNoAttention):
                    outputs = outputs.logits

                loss = outputs.sum()
                loss.backward()

        # 2.) profile memory
        print("\n - - - - - - - - - - - - - - MEMORY - - - - - - - - - - - - - -")
        memory_cost, detailed_info = memory_profiler.profile_memory_cost(
                self.model,
                self.optimizer,
                input_tensor.size(),
                activation_bits=self.activation_bits,
                trainable_param_bits=self.trainable_param_bits,
                frozen_param_bits=self.frozen_param_bits,
                batch_size=1)

        table_data = []
        table_data.append(["Parameter Size", f"{round(detailed_info['param_size']):,} B"])
        table_data.append(["Activation Size", f"{round(detailed_info['act_size']):,} B"])
        table_data.append(["TOTAL MEMORY COST", f" {round(memory_cost):,} B"])
        table_headers = ["Source", "Memory"]
        print(tabulate(table_data, table_headers, stralign="right", colalign=("center", "center")))

        if self.results_dir:
            pass  #TODO
