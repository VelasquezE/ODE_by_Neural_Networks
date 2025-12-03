import torch

criteria = torch.nn.MSELoss()

def get_grad(out, args):
    return torch.autograd.grad(
            out, args,
            grad_outputs = torch.ones_like(out),
            create_graph = True,
            only_inputs = True)

def loss_constant(X, model):
    out = model(X)
    dx_2 = get_grad(out,X)[:,0]**2
    return torch.mean(dx_2) + model(torch.tensor([1.0], requires_grad = True))**2

def loss_harmonic(T,model):
    w = 1
    x = model(T)
    dxdt2 = get_grad(get_grad(x,T)[0],T)[0]
    difeq = criteria(dxdt2,-w*x)
    
    x0 = torch.tensor([0.0], requires_grad = True)
    y0 = model(x0)
    dy0 = get_grad(y0,x0)[0]
    init_con = criteria(y0,torch.zeros_like(y0)) + criteria(dy0,torch.ones_like(dy0))
    print(init_con.shape)
    return difeq + init_con

def loss_harmonic_damped(T,model):
    #parameters
    m,b,k=[1,1,1]
    
    #differential eq
    x = model(T)
    dxdt = get_grad(x,T)[0]
    dxdt2 = get_grad(dxdt,T)[0]
    difeq = criteria(m*dxdt2 + b*dxdt, -k*x)
    
    #initial conditions
    t0 = torch.tensor([0.0],requires_grad=True)
    x0 = model(t0)
    dxdt0 = get_grad(x0,t0)[0]
    initcon = criteria(x0,torch.zeros_like(x0)) + criteria(dxdt0, torch.ones_like(dxdt0))

    return difeq + initcon
