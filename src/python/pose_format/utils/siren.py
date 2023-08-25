from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from pose_format.pose import Pose


class SineLayer(nn.Module):
    """
    A sine activation layer as described in the SIREN paper.

    Parameters
    ----------
    in_features : int
        number of input features.
    out_features : int
        number of output features.
    bias : bool, optional
        If set to False, the layer will not learn an additive bias. Default is True.
    is_first : bool, optional
        If it's the first layer in the network. Default is False.
    omega_0 : float, optional
        hyperparameter. Default is 30.

    Attributes
    ----------
    omega_0 : float
        hyperparameter for controlling the sine function.
    is_first : bool
        Determines how weights are initialized.

    Note
    ----
    - See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    - If `is_first=True`, omega_0 multiplies the activations before the nonlinearity.
    - If `is_first=False`, weights are divided by omega_0 to keep magnitude of activations constant.

  """

    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        """initializes weights"""
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        """
        forward pass through layer 
    
        Parameters
        ----------
        input : torch.Tensor
            input tensor to layer

        Returns
        -------
        torch.Tensor
            Sine "activated" output tensor
        """
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        """
        Forward pass with intermediate value, before sine act. For visualization of activation distributions

        Parameters
        ----------
        input : torch.Tensor
            Input tensor to layer

        Returns
        -------
        tuple
            Sine activated output tensor along with an intermediate tensor
        """
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    """
    SIREN network consisting of SineLayers.

    Parameters
    ----------
    in_features : int
        number of input features.
    hidden_features : int
        number of hidden features.
    hidden_layers : int
        number hidden layers.
    out_features : int
        number output features.
    outermost_linear : bool, optional
        If True, outermost layer is linear. Default- False.
    first_omega_0 : float, optional
        Omega_0 for the first layer. The default is 30
    hidden_omega_0 : float, optional
        Omega_0 for the hidden layers, default; 30

  """

    def __init__(self,
                 in_features,
                 hidden_features,
                 hidden_layers,
                 out_features,
                 outermost_linear=False,
                 first_omega_0=30,
                 hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        """
        Forward pass through network

        Parameters
        ----------
        coords : torch.Tensor
            Input coordinates

        Returns
        -------
        tuple
            output (network) and coordinates (input)
        """
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords

    def forward_with_activations(self, coords, retain_grad=False):
        """
        Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!
        """
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations


class PoseDataset(Dataset):
    """
    Dataset for pose

    Parameters
    ----------
    pose : Pose
        Pose object containing body data & confidence scores

    Note
    ----
    - Assumes pose data is provided in a specific format with body data & confidence scores.
    """

    def __init__(self, pose: Pose):
        super().__init__()

        self.points = torch.tensor([p.flatten() for p in np.array(pose.body.data)], dtype=torch.float32)
        self.confidence = torch.tensor([np.stack([c, c], axis=-1).flatten() for c in np.array(pose.body.confidence)],
                                       dtype=torch.float32)

        self.coords = PoseDataset.get_coords(time=len(self.points) / pose.body.fps, fps=pose.body.fps)

    @staticmethod
    def get_coords(time: float, fps: float):
        """
        Gets coordinates based on time and frames per second (fps)

        Parameters
        ----------
        time : float
            Time (duration)
        fps : float
            frames per second

        Returns
        -------
        torch.Tensor
            Tensor with coordinates
        """
        return torch.tensor([[i / fps] for i in range(int(fps * time))], dtype=torch.float32)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0:
            raise IndexError

        return self.coords, self.points, self.confidence


def masked_mse_loss(model_output: torch.FloatTensor, ground_truth: torch.FloatTensor, confidence: torch.FloatTensor):
    """
    For calculating masked mean squared error loss

    Parameters
    ----------
    model_output : torch.FloatTensor
      output of model
    ground_truth : torch.FloatTensor
      Ground truth values
    confidence : torch.FloatTensor
      Confidence scores for pose

    Returns
    -------
    torch.Tensor
      computed masked mse loss
    """
    sq_error = (model_output - ground_truth)**2
    return (sq_error * confidence).mean()


def get_pose_siren(pose: Pose,
                   hidden_features: int = 256,
                   hidden_layers: int = 4,
                   total_steps=5000,
                   learning_rate=1e-5,
                   batch_size=1,
                   steps_til_summary=None,
                   cuda: bool = True):
    """
    For training: returns a SIREN model for pose data

    Parameters
    ----------
    pose : Pose
      Pose object with body data & confidence
    hidden_features : int, optional
      hidden features. Default 256.
    hidden_layers : int, optional
      hidden layers, default  4.
    total_steps : int, optional
      total (training) steps, default; 5000.
    learning_rate : float, optional
      Learning rate (optimization) default; 1e-5.
    batch_size : int, optional
      batch size (training) default; 1.
    steps_til_summary : int, optional
      Steps until summary. If None, no summary is printed, default is None
    cuda : bool, optional
      If True, training on GPU, default is True

    Returns
    -------
    function
      prediction function with model output of trained SIREN model

    """
    dataset = PoseDataset(pose)
    data_loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=0, shuffle=True)
    shape = pose.body.data.shape

    device = torch.device('cuda') if cuda else torch.device('cpu')

    siren = Siren(in_features=1,
                  out_features=shape[1] * shape[2] * shape[3],
                  hidden_features=hidden_features,
                  hidden_layers=hidden_layers,
                  outermost_linear=True).to(device)

    optimizer = torch.optim.Adam(lr=learning_rate, params=siren.parameters())

    model_input, ground_truth, confidence = next(iter(data_loader))
    model_input, ground_truth, confidence = model_input.to(device), ground_truth.to(device), confidence.to(device)

    for step in range(1, total_steps + 1):
        model_output, coords = siren(model_input)
        loss = masked_mse_loss(model_output, ground_truth, confidence)

        if steps_til_summary is not None and step % steps_til_summary == 0:
            print("Step %d, Total loss %0.6f" % (step, loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def predict(coords: torch.Tensor):
        with torch.no_grad():
            if cuda:
                coords = coords.cuda()
            model_output, _ = siren(coords)

            return model_output.reshape((coords.shape[0], shape[1], shape[2], shape[3]))

    return predict
