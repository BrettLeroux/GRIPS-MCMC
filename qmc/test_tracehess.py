import torch
import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import arrays, array_shapes
from hypothesis.strategies import floats

from qmc.tracehess import autograd_trace_hessian, gradient_f, hessian_f
from qmc.wavefunction import HydrogenTrialWavefunction


def autograd_trace_hessian_full(f, x):
    # only works on scalar functions, here for testing purposes
    # assume we have x of shape [1, d]
    full_hess = torch.autograd.functional.hessian(f, x[0,:])
    return full_hess.diag().sum()

# Gradient Function
def the_func(y):
    # this is for numerical hessian only
    return y[0] ** 2 + y[1] ** 2


def func_quadratic(x):
    return 10.0 * torch.sum(x * x, dim=-1)

def func_offdiag(x):
    return x[..., 0]*x[..., 1]

def func_cubic(x):
    return 10.0 * torch.sum(x * x * x, dim=-1)


def func_sin(x):
    return torch.sin(x).sum(dim=-1)


def func_exp(x):
    return torch.exp(x).sum(dim=-1)


@given(arrays(np.float32, (4, 2, 1), elements=floats(0.0, 10, width=32, exclude_min=True)))
def test_hydrogen_ansatz_autograd(x):
    x = torch.tensor(x)
    #
    func = HydrogenTrialWavefunction(torch.tensor(1.0))
    true_lap = (func.alpha**2 * torch.exp(-func.alpha * x)*(-2.0 + func.alpha * x)).squeeze(dim=-1)
    #
    auto_lap = autograd_trace_hessian(func.hydro_ansatz_sup, x)

    assert true_lap.shape == auto_lap.shape
    assert torch.isclose(true_lap, auto_lap).all()

@given(arrays(np.float32, (1, 2), elements=floats(-10, 10, width=32)))
def test_full_hess_equivalent(x):
    x = torch.tensor(x)
    x2 = x.clone()
    assert np.isclose(autograd_trace_hessian(func_offdiag, x)[0].item(), autograd_trace_hessian_full(func_offdiag, x2).item())


@given(arrays(np.float32, (1, 2), elements=floats(-10, 10, width=32)))
def test_offdiag_laplacian(x):
    x = torch.tensor(x)
    assert np.isclose(autograd_trace_hessian(func_offdiag, x)[0].item(), 0.0)

@given(arrays(np.float32, (2, 1), elements=floats(-10, 10, width=32)))
def test_numerical_gradient_f_nonan(x):
    grad = gradient_f(x, the_func)
    assert not np.isnan(grad).any()


@given(arrays(np.float32, (2, 1), elements=floats(-10, 10, width=32)))
def test_numerical_hessian_f_nonan(x):
    hess = hessian_f(x, the_func)
    assert not np.isnan(hess).any()


# random arrays
@given(arrays(np.float32, (1, 3), elements=floats(-10, 10, width=32)))
def test_autograd_trace_hess_quadratic(x):
    assert autograd_trace_hessian(func_quadratic, torch.tensor(x)) == 60.0


@given(
    arrays(
        np.float32,
        array_shapes(min_dims=2, max_dims=3, min_side=1, max_side=10),
        elements=floats(-10, 10, width=32),
    )
)
def test_autograd_trace_hess_exp(x):
    xtens = torch.tensor(x)
    out = func_exp(xtens)  # deriv of exp is exp

    assert torch.isclose(autograd_trace_hessian(func_exp, xtens), out).all()


def test_autograd_trace_hess_batchdims():
    x1 = torch.ones(1, 3)
    lap = autograd_trace_hessian(func_quadratic, x1)
    assert lap.shape == torch.Size([1])

    x1 = torch.ones(5, 3)
    lap = autograd_trace_hessian(func_quadratic, x1)
    assert lap.shape == torch.Size([5])

    x1 = torch.ones(10, 5, 3)
    lap = autograd_trace_hessian(func_quadratic, x1)
    assert lap.shape == torch.Size([10, 5])


@given(floats(-10, 10))
def test_autograd_trace_hess_sin(x):
    input = torch.tensor([[x]])
    lap = autograd_trace_hessian(func_sin, input)
    assert torch.isclose(lap, -torch.sin(input))


def test_runs_autograd_trace_cubic():
    autograd_trace_hessian(func_cubic, torch.ones(1, 3))


def test_autograd_trace_cubic():
    assert autograd_trace_hessian(func_cubic, torch.ones(1, 3)) == 180.0
