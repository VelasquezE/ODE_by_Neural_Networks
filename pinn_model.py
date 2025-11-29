"""
   Physics informed Neural Network
"""
import torch
import torch.nn as nn

class PINN(nn.Module):
    
    def __init__(self, n_inputs, n_neurons, n_hidden_layers, n_outputs):
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


# TODO: Docstring and module documentation.
