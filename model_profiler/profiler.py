from model_profiler.profilers.memory_tracker.mem_tracker import MemTracker
from model_profiler.profilers.flops_counter.flop_counter import FlopCounterMode
import warnings
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention


class Profiler:
    def __init__(self, model, optimizer=None):
        self.model = model.train()
        self.optimizer = optimizer
        if not self.optimizer:
            warnings.warn(UserWarning("No optimizer -> optimizer.step() is not profiled!"))


    def profile(self, input_tensor, flops_per_layer=False, mem_depth=None):

        # 1.) profile flops
        print("\n - - - - - - - - - - - - - - FLOPs - - - - - - - - - - - - - -")
        flop_counter = FlopCounterMode(self.model, print_flops_per_layer=flops_per_layer)

        if self.optimizer:
            with flop_counter:
                self.optimizer.zero_grad()
                outputs = self.model(input_tensor)

                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                if isinstance(outputs, ImageClassifierOutputWithNoAttention):
                    outputs = outputs.logits

                loss = outputs.sum()
                loss.backward()
                flop_counter.reset_module_tracking_before_optimizer_step()
                self.optimizer.step()
        else:
            warnings.warn("Optimizer not specified, profiling without optimizer.step().", UserWarning)
            with flop_counter:
                outputs = self.model(input_tensor)

                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                if isinstance(outputs, ImageClassifierOutputWithNoAttention):
                    outputs = outputs.logits

                loss = outputs.sum()
                loss.backward()

        # 2.) profile memory
        print("\n - - - - - - - - - - - - - - MEMORY - - - - - - - - - - - - - -")
        mem_tracker = MemTracker(record_category_max=True)
        mem_tracker.track_external(self.model)

        with (mem_tracker as mt):
            for i in range(2):
                out = self.model(input_tensor)
                loss = out.sum()
                loss.backward()
                if self. optimizer:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                if i == 0:
                    mt.reset_mod_stats()  # to account for lazy init of optimizer state (warmup run)

        if mem_depth:
            # this will display a module wise snapshot
            mt.display_modulewise_snapshots(mem_depth, units="MiB", tabulate=True)
        else:
            mt.display_snapshot("peak", units="MiB", tabulate=True)

