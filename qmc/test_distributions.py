from qmc.distributions import RandomHybridRosenbrock, MixtureOfGaussians
from qmc.distributions import dim2Rosenbrock
import torch
import pytest
import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats


def test_dim2rosenbrock_logprob_dims():
    config_dimension = 2
    f = dim2Rosenbrock(torch.tensor(1/20), torch.tensor(100/20))

    input = 0.5*torch.ones(10, config_dimension)
    output = f(input)
    assert len(output.shape) == 1
    assert output.shape[0] == 10

    input = 0.5*torch.ones(1, config_dimension)
    output = f(input)
    assert len(output.shape) == 1
    assert output.shape[0] == 1

    # for multiple iterations of multiple walkers, output should be one scalar per walker and iteration
    input = 0.5*torch.ones(5, 10, config_dimension)
    output = f(input)
    assert len(output.shape) == 2
    assert output.shape[0] == 5
    assert output.shape[1] == 10
    # input = 0.5*torch.ones(1, config_dimension)
    # output = f(input)
    # assert output.shape[0] == 1
    #
    # # for multiple iterations of multiple walkers, output should be one scalar per walker and iteration
    # input = 0.5*torch.ones(5, 10, config_dimension)
    # output = f(input)
    # assert output.shape[0] == 5
    # assert output.shape[1] == 10
    #





@pytest.mark.xfail
def test_rosenbrock_logprob_dims():
    config_dimension = 1
    f = RandomHybridRosenbrock(2, 1)

    input = 0.5*torch.ones(10, config_dimension)
    output = f(input)
    assert len(output.shape) == 1
    assert output.shape[0] == 10

    input = 0.5*torch.ones(1, config_dimension)
    output = f(input)
    assert len(output.shape) == 1
    assert output.shape[0] == 1

    # for multiple iterations of multiple walkers, output should be one scalar per walker and iteration
    input = 0.5*torch.ones(5, 10, config_dimension)
    output = f(input)
    assert len(output.shape) == 2
    assert output.shape[0] == 5
    assert output.shape[1] == 10
    # input = 0.5*torch.ones(1, config_dimension)
    # output = f(input)
    # assert output.shape[0] == 1
    #
    # # for multiple iterations of multiple walkers, output should be one scalar per walker and iteration
    # input = 0.5*torch.ones(5, 10, config_dimension)
    # output = f(input)
    # assert output.shape[0] == 5
    # assert output.shape[1] == 10
    #

def test_mix_gaussian_logprob_dims():
    config_dimension = 3
    f = MixtureOfGaussians([torch.zeros(config_dimension), torch.ones(config_dimension)], [torch.eye(config_dimension), torch.eye(config_dimension)])

    input = 0.5*torch.ones(10, config_dimension)
    output = f(input)
    assert len(output.shape) == 1
    assert output.shape[0] == 10

    input = 0.5*torch.ones(1, config_dimension)
    output = f(input)
    assert len(output.shape) == 1
    assert output.shape[0] == 1

    # for multiple iterations of multiple walkers, output should be one scalar per walker and iteration
    input = 0.5*torch.ones(5, 10, config_dimension)
    output = f(input)
    assert len(output.shape) == 2
    assert output.shape[0] == 5
    assert output.shape[1] == 10
