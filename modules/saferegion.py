# Reference
# https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py


import torch
from torch import Tensor
from torch.nn import Module


# TODO:
# small precision err when cmp with nn.BatchNorm1d
# lazy initializer
# support DDP
# swap in bigger models


class _SafeRegion(Module):
    """
    Module for recording safe regions of neural units
    """
    _version = 1
    num_features: int
    eps: float
    momentum: float

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = None
    ) -> None:
        super(_SafeRegion, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # training stats
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.register_buffer('num_samples_tracked', torch.tensor(0, dtype=torch.long))
        # we are setting initial value to 0. because it really doesn't matter
        # we are checking num of seen batches and based on that we set to first value we see
        self.register_buffer('running_min',  torch.zeros(num_features))
        self.register_buffer('running_max',  torch.zeros(num_features))

        # test stats
        self.last_x = None
        self.last_x_max = None
        self.last_x_min = None
        self.last_in_out = None
        self.last_x_sum = None

    def reset_stats(self) -> None:
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.num_batches_tracked.zero_()
        self.num_samples_tracked.zero_()
        self.running_min.zero_()
        self.running_max.zero_()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)
        batch_size = input.shape[0]
        dims = [d for d in range(input.dim())]
        dims.pop(1)

        min = input.amin(dim=dims)
        max = input.amax(dim=dims)

        # at training time we record parameters
        with torch.no_grad():
            if self.training:
                if self.num_batches_tracked == 0:
                    self.running_min = min
                    self.running_max = max
                else:
                    self.running_min = torch.minimum(self.running_min, min)
                    self.running_max = torch.maximum(self.running_max, max)

                self.num_batches_tracked.add_(1)
                self.num_samples_tracked.add_(batch_size)

                if self.momentum:
                    exponential_average_factor = self.momentum
                else:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)

                mean = input.mean(dims)

                # use biased var in train
                var = input.var(dims, unbiased=False)
                n = input.numel() / input.size(1)
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
            else:
                # test stats
                self.last_x = input
                self.last_x_max = max
                self.last_x_min = min
                self.last_x_mean = input.mean(dims)
                self.last_x_var = input.var(dims, unbiased=False)

        return input


class SafeRegion1d(_SafeRegion):
    """Applies Safe Region recording over a 2D or 3D input (a mini-batch of 1D inputs
     with optional additional channel dimension)
    """
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(input.dim()))


class SafeRegion2d(_SafeRegion):
    """Applies Safe Region recording over a 4D input (a mini-batch of 2D inputs with additional channel dimension)"""
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))


class SafeRegion3d(_SafeRegion):
    """Applies Safe Region recording over a 5D input (a mini-batch of 3D inputs
    with additional channel dimension)"""
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))
