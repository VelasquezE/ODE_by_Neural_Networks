import torch

criteria = torch.nn.MSELoss()

def get_grad(out: torch.Tensor, args: torch.Tensor):
    """
    Compute derivatives with respect to input variables
    using 'torch.autograd.grad'.
    
    Parameters:
        out : torch.Tensor
            Output tensor, ie. y(t), y(x).
        args : torch.Tensor
            Tensor with respecto to which the gradient
            is compute, typically the input variable(s)
            such as t or x.
    Returns:
        Tuple[torch.Tensor]
            Tuple containing the gradient tensor. 
    """
    return torch.autograd.grad(
            out, args,
            grad_outputs = torch.ones_like(out),
            create_graph = True,
            only_inputs = True)

def loss_constant(X, model):
    out = model(X)
    dx_2 = get_grad(out,X)[0]**2
    return torch.mean(dx_2) + model(torch.tensor([1.0], requires_grad = True))**2

def loss_harmonic(T,model):
    w = 1
    x = model(T)
    dx_dt2 = get_grad(get_grad(x,T)[0],T)[0]
    diff_equation = criteria(dx_dt2,- w * x)
    
    x0 = torch.tensor([0.0], requires_grad = True)
    y0 = model(x0)
    dy0 = get_grad(y0,x0)[0]
    inititial_conditions = criteria(y0,torch.zeros_like(y0)) + criteria(dy0,torch.ones_like(dy0))
    return diff_equation + inititial_conditions

def loss_harmonic_damped(T,model):
    #parameters
    m,b,k=[1,1,1]
    
    #differential eq
    x = model(T)
    dx_dt = get_grad(x,T)[0]
    dx_dt2 = get_grad(dx_dt,T)[0]
    diff_equation = criteria(m*dx_dt2 + b*dx_dt, -k*x)
    
    #initial conditions
    t0 = torch.tensor([0.0],requires_grad=True)
    x0 = model(t0)
    dx_dt0 = get_grad(x0,t0)[0]
    initial_conditions = criteria(x0,torch.zeros_like(x0)) + criteria(dx_dt0, torch.ones_like(dx_dt0))

    return diff_equation + initial_conditions
