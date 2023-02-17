import torch
from fairseq import (
    tasks,
    options,
    checkpoint_utils
)
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.trainer import Trainer
from fairseq.data import iterators

import sys
# https://github.com/microsoft/torchscale/tree/main/examples/fairseq
models_path = "/home/v-junliang/DNNGen/concrete_trace_test/torchscale/examples/fairseq"
sys.path.append(models_path)
import models

torch.manual_seed(86)
# build model and create dummy input
parser = options.get_training_parser()
args = options.parse_args_and_arch(parser)
cfg = convert_namespace_to_omegaconf(args)

task = tasks.setup_task(cfg.task)
model = task.build_model(cfg.model)
model.eval()
print("building model succeed: ", type(model))

criterion = task.build_criterion(cfg.criterion)
trainer = Trainer(cfg, task, model, criterion)
_, epoch_itr = checkpoint_utils.load_checkpoint(
    cfg.checkpoint,
    trainer,
    disable_iterator_cache=task.has_sharded_data("train"),
)
itr = epoch_itr.next_epoch_itr(
    fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus,
    shuffle=(epoch_itr.next_epoch_idx > cfg.dataset.curriculum),
)
update_freq = (
    cfg.optimization.update_freq[epoch_itr.epoch-1]
    if epoch_itr.epoch < len(cfg.optimization.update_freq)
    else cfg.optimization.update_freq[-1]
)
itr = iterators.GroupedIterator(itr, update_freq)

for _, samples in enumerate(itr):
    for _, sample in enumerate(samples):
        dummy_input = sample["net_input"]
        device = next(model.parameters()).device
        for key in dummy_input.keys():
            dummy_input[key] = dummy_input[key].to(device)
        break
    break
print("creating dummy input succeed")

with torch.no_grad():
    output_origin = model(**dummy_input)


# Conduct concrete trace below!
import torch
import sys
concrete_trace_utils_path = "/home/v-junliang/DNNGen/concrete_trace_test/nni/nni/common"
sys.path.append(concrete_trace_utils_path)
from concrete_trace_utils import concrete_trace, ConcreteTracer

def check_equal(a, b):
    if type(a) != type(b):
        return False
    if isinstance(a, (list, tuple, set)):
        if len(a) != len(b):
            return False
        for sub_a, sub_b in zip(a, b):
            if not check_equal(sub_a, sub_b):
                return False
        return True
    elif isinstance(a, dict):
        keys_a, kes_b = set(a.keys()), set(b.keys())
        if keys_a != kes_b:
            return False
        for key in keys_a:
            if not check_equal(a[key], b[key]):
                return False
        return True
    elif isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    else:
        return a == b
    
print("start tracing...")
traced_model = concrete_trace(
    model,
    dummy_input,
    use_operator_patch=True,
    autowrap_leaf_class={
        torch.finfo: ((), False),
        type(output_origin): ((), False),  
    },
)
print("trace succeed")
print("checking equal...")
with torch.no_grad():
    output_traced = traced_model(**dummy_input)
assert check_equal(output_origin, output_traced), "check equal failed"
print("checking equal succeed")

# try to save traced model with pickle
from concrete_trace_utils.concrete_tracer import MagicMethodPatcher
from pickle import _Pickler, _Unpickler

with open("save/through_nn_Module/tl_traced.model", "wb") as f:
    # pickle.dump(traced_model, f)
    with MagicMethodPatcher():
        _Pickler(f).dump(traced_model)

with open("save/through_nn_Module/tl_traced.model", "rb") as f:
    with MagicMethodPatcher():
        reload_model = _Unpickler(f).load()


with torch.no_grad():
    output_reload = reload_model(**dummy_input)
assert check_equal(output_origin, output_reload), "reload check equal failed"
print("reload is good!")

with open("save/through_nn_Module/tl_origin.model", "wb") as f:
    with MagicMethodPatcher():
        _Pickler(f).dump(model)

with open("save/through_nn_Module/tl_input.pkl", "wb") as f:
    with MagicMethodPatcher():
        _Pickler(f).dump(dummy_input)

