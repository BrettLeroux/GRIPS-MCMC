from qmc.wavefunction import HarmonicTrialFunction, ParticleBoxFunction, HydrogenTrialWavefunction, \
    HeliumTrialWavefunction
import torch
import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats

# the following are tests for different trial wavefunctions.
# they are meant to be run using the pytest testing framework.
# To use this, make sure you have done "pip install pytest"
# and then run "pytest" in the project directory. You can also run
# "pytest -k test_function_name" to pick out a specific test.

# if you write a new trial wavefunction, copy versions of these tests
# and modify them (configuration dim, valid input values). ensuring
# all trial wavefunctions pass tests like this will allow for a consistent interface.
# Any new test must be a function whose name starts with test_

def test_helium_logprob_dims():
    config_dimension = 3
    f = HeliumTrialWavefunction(torch.tensor(1.0))

    # for multiple walkers, output should one scalar per walker
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

def test_helium_local_energy_dims():
    config_dimension = 3
    f = HeliumTrialWavefunction(torch.tensor(1.0))

    # for multiple walkers, output should one scalar per walker
    input = 0.5*torch.ones(10, config_dimension)
    output = f.local_energy(input)
    assert len(output.shape) == 1
    assert output.shape[0] == 10

    input = 0.5*torch.ones(1, config_dimension)
    output = f.local_energy(input)
    assert len(output.shape) == 1
    assert output.shape[0] == 1

    # for multiple iterations of multiple walkers, output should be one scalar per walker and iteration
    input = 0.5*torch.ones(5, 10, config_dimension)
    output = f.local_energy(input)
    assert len(output.shape) == 2
    assert output.shape[0] == 5
    assert output.shape[1] == 10

def test_hydrogen_logprob_dims():
    config_dimension = 1
    f = HydrogenTrialWavefunction(torch.tensor(1.0))

    # for multiple walkers, output should one scalar per walker
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

def test_hydrogen_local_energy_dims():
    config_dimension = 1
    f = HydrogenTrialWavefunction(torch.tensor(1.0))

    # for multiple walkers, output should one scalar per walker
    input = 0.5*torch.ones(10, config_dimension)
    output = f.local_energy(input)
    assert len(output.shape) == 1
    assert output.shape[0] == 10

    input = 0.5*torch.ones(1, config_dimension)
    output = f.local_energy(input)
    assert len(output.shape) == 1
    assert output.shape[0] == 1

    # for multiple iterations of multiple walkers, output should be one scalar per walker and iteration
    input = 0.5*torch.ones(5, 10, config_dimension)
    output = f.local_energy(input)
    assert len(output.shape) == 2
    assert output.shape[0] == 5
    assert output.shape[1] == 10

def test_harmonic_logprob_dims():
    config_dimension = 1
    f = HarmonicTrialFunction(torch.tensor(1.0))

    # for multiple walkers, output should one scalar per walker
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

def test_harmonic_local_energy_dims():
    config_dimension = 1
    f = HarmonicTrialFunction(torch.tensor(1.0))

    # for multiple walkers, output should one scalar per walker
    input = 0.5*torch.ones(10, config_dimension)
    output = f.local_energy(input)
    assert len(output.shape) == 1
    assert output.shape[0] == 10

    input = 0.5*torch.ones(1, config_dimension)
    output = f.local_energy(input)
    assert len(output.shape) == 1
    assert output.shape[0] == 1

    # for multiple iterations of multiple walkers, output should be one scalar per walker and iteration
    input = 0.5*torch.ones(5, 10, config_dimension)
    output = f.local_energy(input)
    assert len(output.shape) == 2
    assert output.shape[0] == 5
    assert output.shape[1] == 10

def test_particlebox_logprob_dims():
    config_dimension = 3
    f = ParticleBoxFunction(torch.ones(config_dimension))

    # for multiple walkers, output should one scalar per walker
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

# particlebox local energy is a constant, so not testing its output dimensionality

# TODO add more Hypothesis code to check for NaN on random inputs in domain


@given(arrays(np.float, (1, 1), elements=floats(-10, 10)), floats(min_value=0.01, max_value=5))
def test_harmonic_nan(configs, alpha):
    f = HarmonicTrialFunction(torch.tensor(alpha))
    inputs = torch.tensor(configs)
    outputs = f(inputs)
    assert not torch.isnan(outputs).any()
    outputs = f.local_energy(inputs)
    assert not torch.isnan(outputs).any()


@given(arrays(np.float, (1, 3), elements=floats(0, 1)), floats(min_value=0, max_value=20))
def test_particlebox_nan(configs, alpha):
    f = ParticleBoxFunction(alpha*torch.ones(3))
    inputs = torch.tensor(configs)
    outputs = f(inputs)
    assert not torch.isnan(outputs).any()

@given(arrays(np.float32, (4, 2, 1), elements=floats(0.10999999940395355, 10, width=32)))
def test_hydrogen_ansatz_energy(x):
    x = torch.tensor(x)
    #
    func = HydrogenTrialWavefunction(torch.tensor(1.0))

    local_energy = func.local_energy(x)

    assert torch.isclose(local_energy, -0.5*torch.ones_like(local_energy)).all()

