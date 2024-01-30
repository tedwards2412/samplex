import mlx.core as mx
from tqdm import tqdm


class MH_Gaussian_sampler:
    def __init__(self, log_target_distribution):
        self.log_target_distribution = log_target_distribution

    def initialize_chains(self, theta_ini):
        minuslogLs = mx.vmap(self.log_target_distribution)(theta_ini)
        return [mx.concatenate([minuslogLs.reshape(-1, 1), theta_ini], axis=1)]

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
        new_state = mx.where(
            prob > mx.log(rand),
            mx.concatenate(
                [self.log_target_distribution(xproposal).reshape(1), xproposal]
            ),
            mx.concatenate([self.log_target_distribution(state).reshape(1), state]),
        )
        return new_state

    def step_walker(self, current_state, key, cov_matrix, jumping_factor):
        prob_keys = mx.random.split(key, 2)
        new_state = self.single_step(
            current_state, prob_keys[0], prob_keys[1], cov_matrix, jumping_factor
        )
        return new_state

    def run(self, Nsteps, key, theta_ini, cov_matrix, jumping_factor):
        chains = self.initialize_chains(theta_ini)
        steps = mx.arange(Nsteps - 1)
        keys = mx.random.split(key, Nsteps)
        for step in tqdm(steps):
            keys_walkers = mx.random.split(keys[step], theta_ini.shape[0])
            new_state = mx.vmap(self.step_walker, in_axes=(0, 0, None, None))(
                chains[-1][:, 1:],  # 0th element is -logL
                keys_walkers,
                cov_matrix,
                jumping_factor,
            )
            chains.append(new_state)
        return mx.array(chains)
