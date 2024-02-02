import numpy as np
import mlx.core as mx
import samplex.utils as utils


class samplex:
    def __init__(self, sampler, Nwalkers, foldername="MyChains", device=mx.cpu):
        self.Nwalkers = Nwalkers
        self.sampler = sampler

        self.key = mx.random.key(1234)
        self.keys = mx.random.split(self.key, self.Nwalkers)
        self.chains = None
        self.filename = utils.generate_filename(foldername)

        mx.set_default_device(device)

    def run(self, Nsteps, theta_ini, cov_matrix, jumping_factor, Nsave=1000):
        self.chains = self.sampler.run(
            Nsteps=Nsteps,
            key=self.key,
            theta_ini=theta_ini,
            cov_matrix=cov_matrix,
            jumping_factor=mx.array([jumping_factor]),
            Nsave=Nsave,
            filename=self.filename,
            full_chains=self.chains,
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
