import numpy as np
import mlx.core as mx


class samplex:
    def __init__(self, sampler, Nwalkers):
        self.Nwalkers = Nwalkers
        self.sampler = sampler

        self.key = mx.random.key(1234)
        self.keys = mx.random.split(self.key, self.Nwalkers)
        self.chains = None

    def run(self, Nsteps, theta_ini, cov_matrix, jumping_factor):
        self.chains = self.sampler.run(
            Nsteps, self.key, theta_ini, cov_matrix, mx.array([jumping_factor])
        )

        return self.chains

    def get_chain(self, discard=0, thin=1, flat=True):
        if self.chains is None:
            raise ValueError("No chains have been generated yet!")
        if flat:
            return self.chains[discard::thin].reshape(-1, self.chains.shape[-1])
        else:
            return self.chains[discard::thin]

    def reset(self):
        self.chains = None

    def save_chains(self, filename):
        np.save(filename, np.array(self.chains))
