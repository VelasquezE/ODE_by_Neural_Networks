# Estoy mirando cómo se hace xd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyper parameters (puse cualquier cosa)
input_size = 100
output_size = 80
hidden_size = 50
num_epochs = 2
learning_rate = 0.001

# load data

    # ------- data

#  Physics informed Neural Network
 
class PINN(nn.Module):
    def __init__(self, input_dimension, hidden_size, n_layers, output_size):
       super().__init__()

       self.layers = nn.ModuleList()

       self.layers.append(nn.Linear(input_dimension, hidden_size))
       self.layers.append(nn.Tanh())
       
       for ii in range(n_layers - 1):
        self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(nn.Tanh())
        
       self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for layer in self.layers:
           x = layer(x)

        return x




