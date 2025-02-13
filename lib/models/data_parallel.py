import torch
from torch.nn.modules import Module
from torch.nn.parallel.scatter_gather import gather
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.parallel_apply import parallel_apply
from .scatter_gather import scatter_kwargs

class _DataParallel(Module):

    def __init__(self, module, device_ids=None, output_device=None, dim=0, chunk_sizes=None):
        super(_DataParallel, self).__init__()

        if not torch.cuda.is_available():
            self.module = module
            self.device_ids = []
            return

        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        if output_device is None:
            output_device = device_ids[0]
        self.dim = dim
        self.module = module
        self.device_ids = device_ids
        self.chunk_sizes = chunk_sizes
        self.output_device = output_device
        if len(self.device_ids) == 1:
            self.module.cuda(device_ids[0])

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids, self.chunk_sizes)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def replicate(self, module, device_ids):
        return replicate(module, device_ids)

    def scatter(self, inputs, kwargs, device_ids, chunk_sizes):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim, chunk_sizes=self.chunk_sizes)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)


def data_parallel(module, inputs, device_ids=None, output_device=None, dim=0, module_kwargs=None):
    if not isinstance(inputs, tuple):
        inputs = (inputs,)

    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))

    if output_device is None:
        output_device = device_ids[0]

    inputs, module_kwargs = scatter_kwargs(inputs, module_kwargs, device_ids, dim)
    if len(device_ids) == 1:
        return module(*inputs[0], **module_kwargs[0])
    used_device_ids = device_ids[:len(inputs)]
    replicas = replicate(module, used_device_ids)
    outputs = parallel_apply(replicas, inputs, module_kwargs, used_device_ids)
    return gather(outputs, output_device, dim)

def DataParallel(module, device_ids=None, output_device=None, dim=0, chunk_sizes=None):
    if chunk_sizes is None:
        return torch.nn.DataParallel(module, device_ids, output_device, dim)
    standard_size = True
    for i in range(1, len(chunk_sizes)):
        if chunk_sizes[i] != chunk_sizes[0]:
            standard_size = False
    if standard_size:
        return torch.nn.DataParallel(module, device_ids, output_device, dim)
    return _DataParallel(module, device_ids, output_device, dim, chunk_sizes)