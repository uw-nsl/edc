#from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import distributed as dist

if True:
    from typing import Optional

    from numbers import Number

    from torch.distributed import ProcessGroup
    from pytorch_lightning import LightningModule

__all__ = [
    "WeightedGradientReducer"
]

class WeightedGradientReducer:
    def __init__(self, grad_comm_dtype: Optional[torch.dtype] = None, group: Optional[ProcessGroup] = None):
        self.grad_comm_dtype = grad_comm_dtype
        self.group = group

        self.reset_local_weight()
    
    def _get_global_weight(self, device: torch.device) -> torch.Tensor:
        global_weight = self._global_weight

        # Compute global weight across nodes if cache is not available
        if global_weight is None:
            global_weight = self._global_weight = torch.tensor(self._local_weight, device=device)
            dist.all_reduce(global_weight, group=self.group)
        
        return global_weight
    
    def _pre_comm(self, data: torch.Tensor) -> torch.Tensor:
        grad_comm_dtype = self.grad_comm_dtype

        # Apply compression
        if grad_comm_dtype:
            data = data.to(grad_comm_dtype)
        # Scale data by local weight
        data.mul_(self._local_weight/self._get_global_weight(data.device))
        
        return data
    
    def _post_comm(self, data: torch.Tensor, target: torch.Tensor):
        grad_comm_dtype = self.grad_comm_dtype

        # Decompress data to target location
        if grad_comm_dtype:
            target.copy_(data)
        else:
            target.data = data
    
    def reset_local_weight(self):
        # Reset local weight to zero
        self._local_weight = 0.
        # Invalidate global weight cache
        self._global_weight = None
    
    def backward(self, model: LightningModule, loss: torch.Tensor, weight: Number):
        # Get and update local weight
        prev_weight = self._local_weight
        local_weight = self._local_weight = prev_weight+weight
        # Invalidate global weight cache
        self._global_weight = None

        if prev_weight!=0.:
            # Rescale accumulated gradients
            torch._foreach_mul_(
                [param.grad for param in model.parameters()],
                prev_weight/local_weight
            )
            # Rescale loss
            loss = (weight/local_weight)*loss
        
        # Update weighted mean of accumulated local gradients
        model.manual_backward(loss)

    def ddp_hook(self, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
        # Pre-communication steps
        bucket_data = bucket.buffer()
        local_data = self._pre_comm(bucket_data)

        def all_reduce_cb(future: torch.Future) -> torch.Tensor:
            # Post-communication steps
            self._post_comm(future.value()[0], bucket_data)

            return bucket_data
        
        # All reduce gradients across nodes
        return dist.all_reduce(local_data, group=self.group, async_op=True) \
            .get_future().then(all_reduce_cb)

    def fsdp_hook(self, grad: torch.Tensor, shard: Optional[torch.Tensor] = None):
        group = self.group

        if shard is None:
            # Pre-communication steps
            local_data = self._pre_comm(grad)
            # All reduce gradients across nodes
            dist.all_reduce(local_data, group=group)
            # Post-communication steps
            self._post_comm(local_data, grad)
        else:
            # Pre-communication steps
            local_data = self._pre_comm(shard)
            # Reduce and scatter gradients across nodes
            dist.reduce_scatter_tensor(local_data, group=group)
            # Post-communication steps
            self._post_comm(local_data, grad)
