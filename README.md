<p align="center">
  <img src="samplex_logo.png" alt="samplex Logo" width="50%" />
</p>

# samplex
Package of useful sampling algorithms written in MLX. We plan on exploring how a combination of unified memory (by exploiting GPU and CPU together) and auto-diff can be used to get highly efficient and robust sampling locally on your Mac.

Please get in touch if you're interested in contributing (tedwards2412@gmail.com and nash.sabti@gmail.com)!

# Installation

```python
pip install samplex
```

# Basic Usage

For a full example, please see the examples folder. Here is the basic structure for linear regression:

```python
from samplex.samplex import samplex
from samplex.samplers import MH_Gaussian_sampler

# First lets generate some data
x = mx.linspace(-5, 5, 20)
err = mx.random.normal(x.shape)
y = b_true * x**2 + m_true * x + c_true + err


# Our target distribution is just a line
def log_target_distribution(theta, data):
    m, c, b = theta
    x, y, sigma = data
    model = b * x**2 + m * x + c
    residual = y - model
    return sum(-0.5 * (residual**2 / sigma**2))

# The sampler assumes it gets a target distribution with a single input vector theta
logtarget = lambda theta: log_target_distribution(theta, (x, y, err))

# Here are the sampler settings
Nwalkers = 32
Ndim = 3
Nsteps = 10_000
cov_matrix = mx.array([0.01, 0.01, 0.01])
jumping_factor = 1.0

theta0_array = mx.random.uniform(
    mx.array([m_min, c_min, b_min]),
    mx.array([m_max, c_max, b_max]),
    (Nwalkers, Ndim),
)

# Firstly we instantiate a samplex class and then run!
sampler = MH_Gaussian_sampler(logtarget)
sam = samplex(sampler, Nwalkers)
sam.run(Nsteps, theta0_array, cov_matrix, jumping_factor)
```

# Next Steps:

- Get NUTs/HMC running
- Get Ensemble sampler running (emcee)
- Refine plotting
- Add helper functions for variety of priors
- Treating parameters with different update speeds
- Add file of priors and include in target distribution
- Include autocorrelation calculation for steps
