import numpy as np
import torch
import torch.optim as optim
import loss_func as lf
from pinn_model import PINN
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Generate collocation points 
N = 1000
t = np.linspace(0, 10, N).astype(np.float32) + np.random.uniform(-1/N,1/N,N)
t = list(t)
collocation_points = torch.tensor(t, dtype = torch.float32, requires_grad=True).to(device)
collocation_points = collocation_points.unsqueeze(1)

# Define PINN
n_inputs = 1 # y(t)
n_outputs = 1
n_neurons = 20
n_hidden_layers = 1

model = PINN(n_inputs, n_neurons, n_hidden_layers, n_outputs).to(device)

# Optimizer
learning_rate = 0.1
optimizer = optim.Adam(model.parameters(),lr = learning_rate)

# Train the model
N_EPOCHS = [1000,2000,10000]
eval_points = torch.linspace(0,10,300).unsqueeze(1)

for n_epochs in N_EPOCHS:
    for epoch in range(0, n_epochs):
        model.train()
        optimizer.zero_grad()
        loss = lf.loss_harmonic_damped(collocation_points, model,[1, 1, 2])
        loss.backward()
        optimizer.step()

    # Predict solution
    model.eval()
    
    with torch.no_grad():
        prediction = model(eval_points)
    
    # Visualize results and compare with analytical solution
    plt.plot(eval_points, prediction.detach().numpy(), label = f'PINN solution{n_epochs}')

#analytical_sol = torch.sin(collocation_points)
#plt.plot(t, analytical_sol.detach().numpy(), label = 'Analytic')
plt.legend()
plt.show()

