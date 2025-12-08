import numpy as np
import torch
import torch.optim as optim
import loss_func as lf
from pinn_model import PINN
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

results_folder = "results"
os.makedirs(results_folder, exist_ok = "True")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Generate collocation points 
t = np.linspace(0, 1, 1000).astype(np.float32) + np.random.uniform(-1/500,1/500,1000)
t = list(t)
collocation_points = torch.tensor(t, dtype = torch.float32, requires_grad=True).to(device)
collocation_points = collocation_points.unsqueeze(1)

# Define PINN
n_inputs = 1 # y(t)
n_outputs = 1
n_neurons = 20
n_hidden_layers = 4

# Optimizer
learning_rate = 0.001

# Train the model
N_EPOCHS = [1000,2000,10000]
parameters = {"w": 1.0, "x0": 1.0, "v0": 1.0}

for n_epochs in N_EPOCHS:
    model = PINN(n_inputs, n_neurons, n_hidden_layers, n_outputs).to(device)
    optimizer = optim.Adam(model.parameters(),lr = learning_rate)
    for epoch in range(0, n_epochs):
        model.train()
        optimizer.zero_grad()
        loss = lf.give_loss_harmonic_oscillator(collocation_points, model, parameters)
        loss.backward()
        optimizer.step()

    # Predict solution
    model.eval()

    with torch.no_grad():
        prediction = model(collocation_points)
    
    # Visualize results and compare with analytical solution
    plt.plot(t, prediction.detach().numpy(), label = f'PINN solution{n_epochs}')

analytical_sol = parameters["x0"] * torch.cos(parameters["w"] * collocation_points) + (
                parameters["v0"] / parameters["w"]) * torch.sin(parameters["w"]
                * collocation_points)

plt.plot(t, analytical_sol.detach().numpy(), label = 'Analytic')
plt.legend()
plt.show()


