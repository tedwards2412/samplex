import mlx.core as mx
from tqdm import tqdm


class MH_Gaussian_sampler:
    def __init__(self, log_target_distribution):
        self.log_target_distribution = log_target_distribution

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

    def single_step(self, state, step_key, step_key2, cov_matrix, jumping_factor):
        xproposal = self.sample_proposal_distribution(
            state, cov_matrix, step_key, jumping_factor
        )
        prob = self.acceptance_probability(state, xproposal, cov_matrix, jumping_factor)
        rand = mx.random.uniform(key=step_key2)
        new_state = mx.where(prob > mx.log(rand), xproposal, state)
        return new_state

    def step_walker(self, x0, key, steps, cov_matrix, jumping_factor):
        state = x0
        states = []
        step_key = mx.random.split(key, len(steps))
        step_key2 = mx.random.split(step_key[0], len(steps))
        for step in tqdm(steps):
            states.append(state)
            state = self.single_step(
                state, step_key[step], step_key2[step], cov_matrix, jumping_factor
            )
        return states
