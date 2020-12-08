from unittest import TestCase

import torch

from pose_format.torch.masked.tensor import MaskedTensor
from pose_format.torch.masked.torch import MaskedTorch


class TestMaskedTorch(TestCase):
    # cat
    def test_cat(self):
        tensor1 = MaskedTensor(torch.tensor([1, 2]))
        tensor2 = MaskedTensor(torch.tensor([3, 4]))
        stack = MaskedTorch.cat([tensor1, tensor2], dim=0)
        res = MaskedTensor(torch.tensor([[1, 2, 3, 4]]))
        self.assertTrue(torch.all(stack == res), msg="Cat is not equal to expected")

    # stack
    def test_stack(self):
        tensor1 = MaskedTensor(torch.tensor([1, 2]))
        tensor2 = MaskedTensor(torch.tensor([3, 4]))
        stack = MaskedTorch.stack([tensor1, tensor2], dim=0)
        res = MaskedTensor(torch.tensor([[1, 2], [3, 4]]))
        self.assertTrue(torch.all(stack == res), msg="Stack is not equal to expected")

    # zeros
    def test_zeros_tensor_shape(self):
        zeros = MaskedTorch.zeros(1, 2, 3)
        self.assertEqual(zeros.shape, (1, 2, 3))

    def test_zeros_tensor_value(self):
        zeros = MaskedTorch.zeros(1, 2, 3)
        self.assertTrue(torch.all(zeros == 0), msg="Zeros are not all zeros")

    def test_zeros_tensor_type_float(self):
        dtype = torch.float
        zeros = MaskedTorch.zeros(1, 2, 3, dtype=dtype)
        self.assertEqual(zeros.tensor.dtype, dtype)

    def test_zeros_tensor_type_bool(self):
        dtype = torch.bool
        zeros = MaskedTorch.zeros(1, 2, 3, dtype=dtype)
        self.assertEqual(zeros.tensor.dtype, dtype)

    def test_zeros_mask_value(self):
        zeros = MaskedTorch.zeros(1, 2, 3)
        self.assertTrue(torch.all(zeros.mask == 0), msg="Zeros mask are not all zeros")

    # Fallback
    def test_not_implemented_method(self):
        tensor = MaskedTensor(tensor=torch.tensor([1, 2, 3]))
        torch_sum = MaskedTorch.sum(tensor)
        self.assertEqual(torch_sum, torch.tensor(6))
