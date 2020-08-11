import torch


class MaskedTensor:
    def __init__(self, tensor: torch.Tensor, mask: torch.Tensor = None):
        self.tensor = tensor
        self.mask = mask if mask is not None else torch.ones(tensor.shape, dtype=torch.bool).to(tensor.device)

    def __getattr__(self, item):
        val = self.tensor.__getattribute__(item)
        if hasattr(val, '__call__'):  # If is a function
            # return getattr(MaskedTorch, item)(self)
            raise NotImplementedError("callbable '%s' not defined" % item)
        else:
            return val

    def __len__(self):
        return self.tensor.shape[0]

    def __getitem__(self, key):
        tensor = self.tensor[key]
        mask = self.mask[key]
        return MaskedTensor(tensor=tensor, mask=mask)

    def arithmetic(self, action: str, other):
        if isinstance(other, MaskedTensor):
            tensor = getattr(self.tensor, action)(other.tensor)
            mask = self.mask & other.mask
        else:
            tensor = getattr(self.tensor, action)(other)
            mask = self.mask
        return MaskedTensor(tensor=tensor, mask=mask)

    def __add__(self, other):
        return self.arithmetic("__add__", other)

    def __sub__(self, other):
        return self.arithmetic("__sub__", other)

    def __mul__(self, other):
        return self.arithmetic("__mul__", other)

    def __truediv__(self, other):
        return self.arithmetic("__truediv__", other)

    def __eq__(self, other):
        return self.tensor == other

    def pow_(self, exponent: float):
        self.tensor.pow_(exponent)
        return self

    def sum(self, dim: int):
        tensor = self.tensor.sum(dim=dim)
        mask = self.mask.prod(dim=dim).bool()
        return MaskedTensor(tensor=tensor, mask=mask)

    def size(self, *args):
        return self.tensor.size(*args)

    def fix_nan(self):  # TODO think of faster way
        self.tensor[self.tensor != self.tensor] = 0
        return self

    def to(self, device):
        tensor = self.tensor.to(device)
        mask = self.mask.to(device)
        return MaskedTensor(tensor=tensor, mask=mask)

    def cuda(self, device=None, non_blocking: bool = False):
        tensor = self.tensor.cuda(device=device, non_blocking=non_blocking)
        mask = self.mask.cuda(device=device, non_blocking=non_blocking)
        return MaskedTensor(tensor=tensor, mask=mask)

    def zero_filled(self) -> torch.Tensor:
        return self.tensor.mul(self.mask)

    def div(self, other: "MaskedTensor", in_place=False, update_mask=True):
        tensor = torch.div(self.tensor, other.tensor, out=self.tensor if in_place else None)
        mask = self.mask & other.mask if update_mask else self.mask
        return MaskedTensor(tensor, mask)

    def matmul(self, matrix: torch.Tensor):
        tensor = torch.matmul(self.tensor, matrix.to(self.device))
        return MaskedTensor(tensor, self.mask)

    def transpose(self, dim0, dim1):
        tensor = self.tensor.transpose(dim0, dim1)
        mask = self.mask.transpose(dim0, dim1)
        return MaskedTensor(tensor=tensor, mask=mask)

    def permute(self, dims: tuple):
        tensor = self.tensor.permute(dims)
        mask = self.mask.permute(dims)
        return MaskedTensor(tensor=tensor, mask=mask)

    def squeeze(self, dim):
        tensor = self.tensor.squeeze(dim)
        mask = self.mask.squeeze(dim)
        return MaskedTensor(tensor=tensor, mask=mask)

    def split(self, split_size_or_sections, dim=0):
        tensors = torch.split(self.tensor, split_size_or_sections, dim)
        masks = torch.split(self.mask, split_size_or_sections, dim)
        return [MaskedTensor(tensor=tensor, mask=mask) for tensor, mask in zip(tensors, masks)]

    def reshape(self, shape: tuple):
        tensor = self.tensor.reshape(shape=shape)
        mask = self.mask.reshape(shape=shape)
        return MaskedTensor(tensor=tensor, mask=mask)

    def rename(self, *names):
        tensor = self.tensor.rename(*names)
        mask = self.mask.rename(*names)
        return MaskedTensor(tensor=tensor, mask=mask)
