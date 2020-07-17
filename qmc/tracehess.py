import numpy as np
import torch

# Gradient Function
def the_func(y):
    return y[0] ** 2 + y[1] ** 2


def gradient_f(x, f):
    assert x.shape[0] >= x.shape[1], "the vector should be a column vector"
    x = x.astype(float)
    N = x.shape[0]
    gradient = []
    for i in range(N):
        eps = np.finfo(np.float32).eps
        xx0 = 1.0 * x[i]
        f0 = f(x)
        x[i] = x[i] + eps
        f1 = f(x)
        gradient.append((np.array([f1 - f0])).item() / eps)
        x[i] = xx0
    return np.array(gradient).reshape(x.shape)


# Hessian Matrix
def hessian_f(x, the_func):
    N = x.shape[0]
    hessian = np.zeros((N, N))
    gd_0 = gradient_f(x, the_func)
    eps = np.finfo(np.float32).eps
    for i in range(N):
        xx0 = 1.0 * x[i]
        x[i] = xx0 + eps
        gd_1 = gradient_f(x, the_func)
        hessian[:, i] = ((gd_1 - gd_0) / eps).reshape(x.shape[0])
        x[i] = xx0
    return hessian


def autograd_trace_hessian(the_func, x):

    # uses the following trick:
    # d^2 f/dz^2 f(x + z*1), where 1 denotes a vector of ones
    # and z is a single scalar, when evaluated at z=0,
    # will give the trace of the Hessian evaluated
    # at x, i.e. trace(\nabla_x^2 f(x))

    assert (
        len(x.shape) >= 2
    ), "x must have some batch dims. if you want to work on a single configuration, try x.unsqueeze(0) to make it be shape [1,d]"

    x.requires_grad_()
    fout = the_func(x)
    (gradient,) = torch.autograd.grad(
        fout, x, grad_outputs=torch.ones_like(fout), create_graph=True
    )

    lap = sum(torch.autograd.grad(
        gradient[..., i], x, grad_outputs=torch.ones_like(gradient[..., i]), retain_graph=True
    )[0][..., i] for i in range(x.shape[-1]))
    return lap
