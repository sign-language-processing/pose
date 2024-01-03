import pickle
from unittest import TestCase

import torch

from pose_format.torch.masked.tensor import MaskedTensor
from pose_format.torch.masked.torch import MaskedTorch


class TestMaskedTorch(TestCase):
    """Test cases for the :class:`~pose_format.torch.masked.tensor.MaskedTensor` class """

    # cat
    def test_cat(self):
        """Test `cat` method for concatenating :class:`~pose_format.torch.masked.tensor.MaskedTensor` objects along a specified dimension."""
        tensor1 = MaskedTensor(torch.tensor([1, 2]))
        tensor2 = MaskedTensor(torch.tensor([3, 4]))
        stack = MaskedTorch.cat([tensor1, tensor2], dim=0)
        res = MaskedTensor(torch.tensor([[1, 2, 3, 4]]))
        self.assertTrue(torch.all(stack == res), msg="Cat is not equal to expected")

    # stack
    def test_stack(self):
        """Tests `stack` method for stacking :class:`~pose_format.torch.masked.tensor.MaskedTensor` objects along a new dimension."""
        tensor1 = MaskedTensor(torch.tensor([1, 2]))
        tensor2 = MaskedTensor(torch.tensor([3, 4]))
        stack = MaskedTorch.stack([tensor1, tensor2], dim=0)
        res = MaskedTensor(torch.tensor([[1, 2], [3, 4]]))
        self.assertTrue(torch.all(stack == res), msg="Stack is not equal to expected")

    # zeros
    def test_zeros_tensor_shape(self):
        """Test if `zeros` method correctly produces a :class:`~pose_format.torch.masked.tensor.MaskedTensor` with the desired shape."""
        zeros = MaskedTorch.zeros(1, 2, 3)
        self.assertEqual(zeros.shape, (1, 2, 3))

    def test_zeros_tensor_value(self):
        """Test if the `zeros` method produces a :class:`~pose_format.torch.masked.tensor.MaskedTensor` with all zero values."""
        zeros = MaskedTorch.zeros(1, 2, 3)
        self.assertTrue(torch.all(zeros == 0), msg="Zeros are not all zeros")

    def test_zeros_tensor_type_float(self):
        """Test if the `zeros` method produces a :class:`~pose_format.torch.masked.tensor.MaskedTensor` with the correct float data type."""
        dtype = torch.float
        zeros = MaskedTorch.zeros(1, 2, 3, dtype=dtype)
        self.assertEqual(zeros.tensor.dtype, dtype)

    def test_zeros_tensor_type_bool(self):
        """Test if the `zeros` method produces a :class:`~pose_format.torch.masked.tensor.MaskedTensor` with the correct boolean data type."""
        dtype = torch.bool
        zeros = MaskedTorch.zeros(1, 2, 3, dtype=dtype)
        self.assertEqual(zeros.tensor.dtype, dtype)

    def test_zeros_mask_value(self):
        """Test if the mask in the produced `zeros` :class:`~pose_format.torch.masked.tensor.MaskedTensor` is initialized with zero values."""
        zeros = MaskedTorch.zeros(1, 2, 3)
        self.assertTrue(torch.all(zeros.mask == 0), msg="Zeros mask are not all zeros")

    # Fallback
    def test_not_implemented_method(self):
        """Tests behavior when invoking an unimplemented method on a :class:`~pose_format.torch.masked.tensor.MaskedTensor`."""
        tensor = MaskedTensor(tensor=torch.tensor([1, 2, 3]))
        torch_sum = MaskedTorch.sum(tensor)
        self.assertEqual(torch_sum, torch.tensor(6))

    def test_pickle(self):
        tensor = MaskedTensor(torch.tensor([1, 2, 3]))
        pickled_tensor = pickle.dumps(tensor)
        loaded_tensor = pickle.loads(pickled_tensor)
        self.assertTrue(torch.all(tensor.tensor == loaded_tensor.tensor),
                        msg="Pickled tensor is not equal to original tensor")
        self.assertTrue(torch.all(tensor.mask == loaded_tensor.mask),
                        msg="Pickled tensor mask is not equal to original tensor")
