"""
PINN module: defines a fully connected neural network architecture for
Physics-Informed Neural Networks (PINNs).

This module provides a general-purpose feed-forward network with adjustable
input size, number of hidden layers, neurons per layer, and output size.

The physics-based loss and training loop must be implemented externally.
"""
import torch
import torch.nn as nn

class PINN(nn.Module):
    """
    Fully connected neural network used in PINNs.

    Parameters:
      n_inputs : int
          Dimensionality of the input (e.g., 1 for t, 2 for (x, t)).
      n_neurons : int
          Number of neurons in each hidden layer.
      n_hidden_layers : int
          Number of hidden layers in the network.
      n_outputs : int
          Dimensionality of the output (e.g., 1 for y(t)).

    Notes
    -----
    The network uses Tanh activations and is compatible with PyTorch autograd
    for computing first and higher-order derivatives needed in PINNs.
    """
    def __init__(self, n_inputs: int, n_neurons: int, n_hidden_layers: int, n_outputs: int):
       super().__init__()

       self.layers = nn.ModuleList()

       self.layers.append(nn.Linear(n_inputs, n_neurons))
       self.layers.append(nn.Tanh())
       
       for ii in range(n_hidden_layers - 1):
        self.layers.append(nn.Linear(n_neurons, n_neurons))
        self.layers.append(nn.Tanh())
        
       self.layers.append(nn.Linear(n_neurons, n_outputs))

    def forward(self, x):
        for layer in self.layers:
           x = layer(x)

        return x


