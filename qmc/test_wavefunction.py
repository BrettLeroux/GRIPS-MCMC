from qmc.wavefunction import HarmonicTrialFunction, ParticleBoxFunction, HeliumTrialWavefunction
import torch

# the following are tests for different trial wavefunctions.
# they are meant to be run using the pytest testing framework.
# To use this, make sure you have done "pip install pytest"
# and then run "pytest" in the project directory. You can also run
# "pytest -k test_function_name" to pick out a specific test.

# if you write a new trial wavefunction, copy versions of these tests
# and modify them (configuration dim, valid input values). ensuring
# all trial wavefunctions pass tests like this will allow for a consistent interface.
# Any new test must be a function whose name starts with test_

def test_harmonic_logprob_dims():
    config_dimension = 1
    f = HarmonicTrialFunction(torch.tensor(1.0))

    # for multiple walkers, output should one scalar per walker
    input = 0.5*torch.ones(10, config_dimension)
    output = f(input)
    assert output.shape[0] == 10

    input = 0.5*torch.ones(1, config_dimension)
    output = f(input)
    assert output.shape[0] == 1

    # for multiple iterations of multiple walkers, output should be one scalar per walker and iteration
    input = 0.5*torch.ones(5, 10, config_dimension)
    output = f(input)
    assert output.shape[0] == 5
    assert output.shape[1] == 10

def test_harmonic_local_energy_dims():
    config_dimension = 1
    f = HarmonicTrialFunction(torch.tensor(1.0))

    # for multiple walkers, output should one scalar per walker
    input = 0.5*torch.ones(10, config_dimension)
    output = f.local_energy(input)
    assert output.shape[0] == 10

    input = 0.5*torch.ones(1, config_dimension)
    output = f.local_energy(input)
    assert output.shape[0] == 1

    # for multiple iterations of multiple walkers, output should be one scalar per walker and iteration
    input = 0.5*torch.ones(5, 10, config_dimension)
    output = f.local_energy(input)
    assert output.shape[0] == 5
    assert output.shape[1] == 10

def test_particlebox_logprob_dims():
    config_dimension = 3
    f = ParticleBoxFunction(torch.ones(config_dimension))

    # for multiple walkers, output should one scalar per walker
    input = 0.5*torch.ones(10, config_dimension)
    output = f(input)
    assert output.shape[0] == 10

    input = 0.5*torch.ones(1, config_dimension)
    output = f(input)
    assert output.shape[0] == 1

    # for multiple iterations of multiple walkers, output should be one scalar per walker and iteration
    input = 0.5*torch.ones(5, 10, config_dimension)
    output = f(input)
    assert output.shape[0] == 5
    assert output.shape[1] == 10

# particlebox local energy is a constant, so not testing its output dimensionality

def test_helium_logprob_dims():
    config_dimension = 3
    f = HeliumTrialWavefunction(torch.tensor(1.0))

    # for multiple walkers, output should one scalar per walker
    input = 0.5*torch.ones(10, config_dimension)
    output = f(input)
    assert output.shape[0] == 10

    input = 0.5*torch.ones(1, config_dimension)
    output = f(input)
    assert output.shape[0] == 1

    # for multiple iterations of multiple walkers, output should be one scalar per walker and iteration
    input = 0.5*torch.ones(5, 10, config_dimension)
    output = f(input)
    assert output.shape[0] == 5
    assert output.shape[1] == 10

def test_helium_local_energy_dims():
    config_dimension = 3
    f = HeliumTrialWavefunction(torch.tensor(1.0))

    # for multiple walkers, output should one scalar per walker
    input = 0.5*torch.ones(10, config_dimension)
    output = f.local_energy(input)
    assert output.shape[0] == 10

    input = 0.5*torch.ones(1, config_dimension)
    output = f.local_energy(input)
    assert output.shape[0] == 1

    # for multiple iterations of multiple walkers, output should be one scalar per walker and iteration
    input = 0.5*torch.ones(5, 10, config_dimension)
    output = f.local_energy(input)
    assert output.shape[0] == 5
    assert output.shape[1] == 10

# TODO add Hypothesis code to check for NaN on random inputs in domain