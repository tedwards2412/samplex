import numpy as np
import mlx.core as mx
import samplex.utils as utils


class samplex:
    def __init__(self, sampler, Nwalkers, device=mx.cpu):
        self.Nwalkers = Nwalkers
        self.sampler = sampler

        self.key = mx.random.key(1234)
        self.keys = mx.random.split(self.key, self.Nwalkers)
        self.chains = None

        mx.set_default_device(device)

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

    def get_bestfit(self):
        return utils.get_bestfit(self.chains)
