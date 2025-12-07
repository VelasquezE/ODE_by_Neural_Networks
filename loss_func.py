import torch

mse = torch.nn.MSELoss()

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

def give_loss_harmonic_oscillator(points: torch.Tensor, model, params: dict):
    """
    Calculates the loss function for the harmonic oscillator.
    
    Parameters:
        points : torch.Tensor
            Collocation points
        model:
            Neural network.
        params : dict
            Dictionary containing the armonic oscillator parameters.
            "w" (float), "x0" (float), "v0" (float).
    Returns: 
        Gives the loss function
            diff_equation + initial_position + initial_velocity
    """
    device = points.device
    dtype = points.dtype

    x = model(points)
    dx_dt = get_grad(x, points)[0]
    dx_dt2 = get_grad(dx_dt, points)[0]
    
    diff_equation = mse(dx_dt2,- (params["w"] ** 2) * x)

    t0 = torch.tensor([0.0], device = device, dtype = dtype, requires_grad = True)

    x0_predict= model(t0)
    v0_predict = get_grad(x0_predict, t0)[0]

    initial_position = mse(x0_predict, params["x0"] * torch.ones_like(x0_predict))  
    initial_velocity = mse(v0_predict, params["v0"] * torch.ones_like(v0_predict))
    return diff_equation + initial_position + initial_velocity

def loss_harmonic_damped(points, model, params):
    m, b, k = params
    
    #differential eq
    x = model(points)
    dx_dt = get_grad(x, points)[0]
    dx_dt2 = get_grad(dx_dt, points)[0]
    diff_equation = mse(m * dx_dt2 + b * dx_dt, - k * x)
    
    #initial conditions
    t0 = torch.tensor([0.0], requires_grad = True)
    x0 = model(t0)
    dx_dt0 = get_grad(x0, t0)[0]
    initial_conditions = mse(x0, torch.zeros_like(x0)) + mse(dx_dt0, torch.ones_like(dx_dt0))

    return diff_equation + initial_conditions
