from __future__ import annotations

from typing import TYPE_CHECKING, overload

import math

import torch
import numpy as np

if TYPE_CHECKING:
    from typing import Any, Optional, TypeVar, Union
    from collections.abc import Sequence
    
    from numbers import Number

    from torch.utils.data import Dataset

    T = TypeVar("T")

__all__ = [
    "EnumerateDataset",
    "as_tensor",
    "decompress",
    "init_embed_weight",
    "pad_cat_tensors",
    "pad_stack_tensors",
    "topk"
]

class EnumerateDataset:
    def __init__(self, dataset: Dataset[T]):
        self.dataset = dataset
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> tuple[int, T]:
        return idx, self.dataset[idx]

def as_tensor(data) -> torch.Tensor:
    return torch.as_tensor(np.asarray(memoryview(data)))

def pad_cat_tensors(tensors: list[torch.Tensor], padded_sizes: Union[int, Sequence[int]] = -1,
    multiple_of: int = 8, pad_value: Number = 0) -> torch.Tensor:
    padded_sizes = [padded_sizes] if isinstance(padded_sizes, int) else list(padded_sizes)

    # Infer padded size for each dimension
    for dim, padded_size in enumerate(padded_sizes):
        if padded_size>=0:
            continue

        tensor_size_max = max(tensor.size(dim+1) for tensor in tensors)
        padded_sizes[dim] = math.ceil(tensor_size_max/multiple_of)*multiple_of
    
    # Number of padded dimensions
    n_pad_dims = len(padded_sizes)
    # Size of initial dimension after concatenation
    cat_size = sum(len(tensor) for tensor in tensors)
    
    # Make concatenated tensor
    cat_tensor = tensors[0].new_full(
        (cat_size, *padded_sizes, *tensors[0].shape[n_pad_dims+1:]), pad_value
    )
    # Copy data to concatenated tensor
    offset = 0
    for tensor in tensors:
        prev_offset = offset
        offset += len(tensor)

        copy_idx = [slice(prev_offset, offset)]
        copy_idx.extend(slice(None, tensor.size(dim+1)) for dim in range(n_pad_dims))

        cat_tensor[tuple(copy_idx)] = tensor
    
    return cat_tensor

def pad_stack_tensors(tensors: list[torch.Tensor], padded_sizes: Union[int, Sequence[int]] = -1,
    multiple_of: int = 8, pad_value: Number = 0) -> torch.Tensor:
    return pad_cat_tensors(
        [tensor.unsqueeze(0) for tensor in tensors],
        padded_sizes, multiple_of, pad_value
    )

def topk(inputs: torch.Tensor, k: int, dim: int = -1, largest: bool = True, sorted: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
    dim_size = inputs.size(dim)

    if dim_size>k:
        values, indices = inputs.topk(k, dim, largest, sorted)
    else:
        if sorted:
            values, indices = inputs.sort(dim, descending=largest)
        else:
            values = inputs

            # Compatible shape for expansion
            compat_shape = [1]*inputs.dim
            compat_shape[dim] = dim_size

            indices = torch.arange(dim_size, device=inputs.device)
            indices = indices.reshape(compat_shape).expand_as(inputs).contiguous()
        
    return values, indices

@torch.no_grad()
def init_embed_weight(target: torch.Tensor, source: torch.Tensor):
    # Initialize target embeddings with same scale as source embeddings
    target.normal_().mul_(source.std(0)).add_(source.mean(0))

@overload
def decompress(values: torch.Tensor, mask: torch.Tensor, fill_value: Number) -> torch.Tensor: ...

@overload
def decompress(value: np.ndarray, mask: np.ndarray, fill_value: Any) -> np.ndarray: ...

def decompress(values, mask, fill_value):
    decompressed_shape = (*mask.shape, *values.shape[1:])
    # PyTorch tensor
    if isinstance(values, torch.Tensor):
        decompressed = values.new_empty(decompressed_shape)
    # NumPy array
    elif isinstance(values, np.ndarray):
        decompressed = np.empty(decompressed_shape, dtype=values.dtype)
    # Unsupported type
    else:
        raise TypeError(f"unsupported values type: {type(values)}")
    
    decompressed[mask] = values
    decompressed[~mask] = fill_value

    return decompressed
