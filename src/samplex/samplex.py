import numpy as np
import mlx.core as mx
from tqdm import tqdm


class samplex:
    def __init__(self, sampler, Nwalkers):
        self.Nwalkers = Nwalkers
        self.sampler = sampler

        self.key = mx.random.key(1234)
        self.keys = mx.random.split(self.key, self.Nwalkers)

    def run(self, Nsteps, theta_ini, cov_matrix, jumping_factor):
        steps = mx.arange(Nsteps)
        chains = mx.vmap(self.sampler.step_walker, in_axes=(0, 0, None, None, None))(
            theta_ini, self.keys, steps, cov_matrix, mx.array([jumping_factor])
        )
        self.chains = mx.array(chains)

        return self.chains
