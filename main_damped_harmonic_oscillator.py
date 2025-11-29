"""
PINN applied to a damped harmonic oscillator
"""
import numpy as np
import torch
import torch.optim as optim

from pinn_models import PINN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Generate training data 
t = np.linspace(0, 100, 500).astype(np.float32)
t = t.reshape(-1, 1)
t_torch = torch.tensor(t, dtype = torch.float32).to(device)

# Initial contiditions
t0 = torch.tensor([[0.0]], dtype=torch.float32).to(device)
x0 = torch.tensor([[1.0]], dtype=torch.float32).to(device)
v0 = torch.tensor([[0.0]], dtype=torch.float32).to(device)

# Define PINN

n_inputs = 1 # y(t)
n_outputs = 1
n_neurons = 20
n_hidden_layers = 3

model = PINN(n_inputs, n_neurons, n_hidden_layers, n_outputs).to(device)

# Optimizer

learning_rate = 0.1
optimizer = optim.Adam(lr = learning_rate)

# Train the model
n_epochs = 10

# Predict solution

# Visualize results and compare with analytical solution