import torch
import numpy as np
import matplotlib.pyplot as plt

def fun(X,w0,w1): 
    out0 = sigmoid(X * w0)
    out1 = sigmoid(out0 @ w1)
    return out1

def sigmoid(x) :
    return 1.0/(1 + torch.exp(-x))

def loss(X,w0,w1):
    tmp = []
    for x in X:
        out = fun(x,w0,w1)
        
        dx = torch.autograd.grad(
            out, x,
            grad_outputs=torch.ones_like(out),
            create_graph=True
        )[0]

        tmp.append(dx**2)
    tmp = torch.stack(tmp)
    return torch.mean(tmp) + fun(torch.tensor([1.0]), w0, w1)**2

n_inputs = 1
n_hidden_neurons = 2
n_outputs = 1

np.random.seed(0)

X = torch.tensor([1.0,0.0,-1.0], requires_grad=True)
w0 = torch.rand(n_inputs, n_hidden_neurons, requires_grad=True)
w1 = torch.rand(n_hidden_neurons, n_outputs, requires_grad=True)
optimizer = torch.optim.Adam([w0,w1],lr=1e-3) #aqui se introduce la red neuronal
n_iters = 100000
costs = []

for ii in range(0,n_iters):
    optimizer.zero_grad()
    cost = loss(X,w0,w1)
    costs.append(cost.item())
    cost.backward()
    optimizer.step()
xx = np.linspace(-1,1,100)
xx = list(xx)
f = lambda x: fun(x,w0,w1)
yy=[]

for val in xx:
    yy.append(f(torch.tensor([val], dtype=torch.float32)).item())

plt.plot(xx,yy)
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.show()
plt.plot(costs)
plt.show()
