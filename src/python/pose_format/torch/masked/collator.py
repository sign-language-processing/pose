from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from pose_format.torch.masked import MaskedTensor, MaskedTorch


def pad_tensors(batch: List[Union[torch.Tensor, MaskedTensor]], pad_value=0):
    datum = batch[0]
    torch_cls = MaskedTorch if isinstance(datum, MaskedTensor) else torch

    max_len = max(len(t) for t in batch)
    if max_len == 1:
        return torch_cls.stack(batch, dim=0)

    new_batch = []
    for tensor in batch:
        missing = list(tensor.shape)
        missing[0] = max_len - tensor.shape[0]

        if missing[0] > 0:
            padding_tensor = torch.full(missing, fill_value=pad_value, dtype=tensor.dtype, device=tensor.device)
            if isinstance(tensor, MaskedTensor):
                padding_tensor = MaskedTensor(tensor=padding_tensor, mask=torch.zeros_like(padding_tensor, dtype=torch.bool))
            tensor = torch_cls.cat([tensor, padding_tensor], dim=0)

        new_batch.append(tensor)

    return torch_cls.stack(new_batch, dim=0)


def collate_tensors(batch: List, pad_value=0) -> Union[torch.Tensor, List]:
    datum = batch[0]

    if isinstance(datum, dict):  # Recurse over dictionaries
        return zero_pad_collator(batch)

    if isinstance(datum, (int, np.int32)):
        return torch.tensor(batch, dtype=torch.long)

    if isinstance(datum, (MaskedTensor, torch.Tensor)):
        return pad_tensors(batch, pad_value=pad_value)

    return batch


def zero_pad_collator(batch) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]]:
    datum = batch[0]

    # For strings
    if isinstance(datum, str):
        return batch

    # For tuples
    if isinstance(datum, tuple):
        return tuple(collate_tensors([b[i] for b in batch]) for i in range(len(datum)))

    # For tensors
    if isinstance(datum, MaskedTensor):
        return collate_tensors(batch)

    # For dictionaries
    keys = datum.keys()
    return {k: collate_tensors([b[k] for b in batch]) for k in keys}


