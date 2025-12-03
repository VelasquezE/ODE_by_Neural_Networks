"""
PINN applied to a damped harmonic oscillator
"""
import numpy as np
import torch
import torch.optim as optim
import loss_func as lf
from pinn_model import PINN
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Generate training data 
t = np.linspace(0, 1, 500).astype(np.float32)
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
optimizer = optim.Adam(model.parameters(),lr = learning_rate)

# Train the model
n_epochs = 100

collocation_points = torch.tensor([0.2,0.5,0.7,0.1], requires_grad = True).to(device)
collocation_points = collocation_points.unsqueeze(1)

for epoch in range(0, n_epochs):
    model.train()
    optimizer.zero_grad()
    loss = lf.loss_harmonic(collocation_points, model)
    loss.backward()
    optimizer.step()

# Predict solution
model.eval()

with torch.no_grad():
    prediction = model(t_torch)

analytical_sol = torch.sin(t_torch)

# Visualize results and compare with analytical solution
plt.plot(t, prediction.detach().numpy(), label = 'PINN solution')
plt.plot(t, analytical_sol, label = 'Analytic')
plt.legend()
plt.show()

