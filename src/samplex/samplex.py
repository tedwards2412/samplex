import numpy as np
import mlx.core as mx
from tqdm import tqdm


class samplex:
    def __init__(self, Nwalkers, Ndim, log_target_distribution):
        self.log_target_distribution = log_target_distribution
        self.Nwalkers = Nwalkers
        self.Ndim = Ndim

        self.x0_array = self.initial_condition(
            mx.zeros((self.Nwalkers, self.Ndim)) + 1, 5.0
        )
        self.key = mx.random.key(1234)
        self.keys = mx.random.split(self.key, self.Nwalkers)

    def initial_condition(self, a, b):
        return mx.random.uniform(a, b, shape=a.shape)

    def proposal_distribution(self, x, y, cov_matrix, jumping_factor=1.0):
        sigma = jumping_factor * cov_matrix
        return sum(
            (1 / mx.sqrt(2 * mx.pi * sigma**2))
            * mx.exp(-0.5 * (y - x) ** 2 / sigma**2)
        )

    def sample_proposal_distribution(
        self, current, cov_matrix, key, jumping_factor=1.0
    ):
        sigma = jumping_factor * cov_matrix
        return current + sigma * mx.random.normal(key=key, shape=current.shape)

    def acceptance_probability(self, current, proposal, cov_matrix, jumping_factor=1.0):
        prob = (
            self.log_target_distribution(proposal)
            + mx.log(
                self.proposal_distribution(
                    current, proposal, cov_matrix, jumping_factor
                )
            )
            - (
                self.log_target_distribution(current)
                + mx.log(
                    self.proposal_distribution(
                        proposal, current, cov_matrix, jumping_factor
                    )
                )
            )
        )
        return mx.minimum(0.0, prob)

    def internal_function(self, x0, key, steps, cov_matrix, jumping_factor):
        xcurrent = x0
        states = []
        step_key = mx.random.split(key, len(steps))
        step_key2 = mx.random.split(step_key[0], len(steps))
        for step in tqdm(steps):
            states.append(xcurrent)
            xproposal = self.sample_proposal_distribution(
                xcurrent, cov_matrix, step_key[step], jumping_factor
            )
            prob = self.acceptance_probability(
                xcurrent, xproposal, cov_matrix, jumping_factor
            )
            rand = mx.random.uniform(key=step_key2[step])
            xcurrent = mx.where(prob > mx.log(rand), xproposal, xcurrent)
        return states

    def run(self, Nsteps, cov_matrix, jumping_factor):
        steps = mx.arange(Nsteps)
        result = mx.vmap(self.internal_function, in_axes=(0, 0, None, None, None))(
            self.x0_array, self.keys, steps, cov_matrix, mx.array([jumping_factor])
        )
        return np.array(result)
