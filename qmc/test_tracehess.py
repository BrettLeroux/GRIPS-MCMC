import torch
import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats

from qmc.tracehess import autograd_trace_hessian


def func_quadratic(x):
    return 10.0*torch.sum(x*x)


def func_cubic(x):
    return 10.0*torch.sum(x*x*x)


# random arrays
@given(arrays(np.float, 3, elements=floats(-10, 10)))
def test_autograd_trace_hess_quadratic(x):
    assert autograd_trace_hessian(torch.tensor(x), func_quadratic) == 60.0


def test_autograd_trace_cubic():
    x = torch.ones(3)
    assert autograd_trace_hessian(torch.tensor(x), func_cubic) == 180.0

