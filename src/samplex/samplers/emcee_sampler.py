import mlx.core as mx
from tqdm import tqdm
import numpy as np


class emcee_sampler:
    def __init__(self, log_target_distribution):
        self.log_target_distribution = log_target_distribution
        self.Ndim = None

    def initialize_chains(self, theta_ini):
        minuslogLs = [-1.0 * mx.vmap(self.log_target_distribution)(theta_ini)]
        return minuslogLs, [theta_ini]

    def sample_proposal_distribution(self, current, complementary_set, comp_key):
        dist_key1, dist_key2 = mx.random.split(comp_key, 2)
        a = 2.0
        rand = mx.random.randint(
            low=0,
            high=len(complementary_set) - 1,
            key=dist_key1,
        )
        xj = complementary_set[rand]
        z = ((a - 1.0) * mx.random.uniform(low=0, high=1, key=dist_key2) + 1) ** 2.0 / a
        log_acc_prob_factor = (self.Ndim - 1.0) * mx.log(z)
        return xj + z * (current - xj), log_acc_prob_factor

    def acceptance_probability(self, current, proposal, log_acc_prob_factor):
        log_target_p = self.log_target_distribution(proposal)
        log_target_c = self.log_target_distribution(current)
        log_prob = log_acc_prob_factor + log_target_p - log_target_c
        return mx.minimum(0.0, log_prob), log_target_p, log_target_c

    def single_step(self, state, complementary_set, step_key, step_key2):
        xproposal, log_acc_prob_factor = self.sample_proposal_distribution(
            state, complementary_set, step_key
        )

        log_prob, log_target_p, log_target_c = self.acceptance_probability(
            state, xproposal, log_acc_prob_factor
        )
        rand = mx.random.uniform(key=step_key2)
        new_state = mx.where(
            log_prob > mx.log(rand),
            xproposal,
            state,
        )
        logL = mx.where(
            log_prob > mx.log(rand),
            -1.0 * log_target_p,
            -1.0 * log_target_c,
        )
        return logL, new_state

    def step_walker(self, current_state, complementary_set, key):
        prob_keys = mx.random.split(key, 2)
        new_state = self.single_step(
            current_state, complementary_set, prob_keys[0], prob_keys[1]
        )
        return new_state[0], new_state[1]

    def run(
        self,
        Nsteps,
        key,
        theta_ini,
        Nsave=1000,
        filename="MyChains.npy",
        full_chains=None,
    ):
        if full_chains is None:
            logLs, chains = self.initialize_chains(theta_ini)
        else:
            logLs, chains = (
                list(full_chains[:, :, 0]),
                list(full_chains[:, :, 1:]),
            )
        if not self.Ndim:
            self.Ndim = theta_ini.shape[1]

        steps = mx.arange(Nsteps)
        mid = theta_ini.shape[0] // 2

        keys = mx.random.split(key, Nsteps)
        for step in tqdm(steps):
            current_state = chains[-1]
            set1 = current_state[mid:]
            set2 = current_state[:mid]
            keys_walkers = mx.random.split(keys[step], theta_ini.shape[0])
            new_logLs_set1, new_set1 = mx.vmap(self.step_walker, in_axes=(0, None, 0))(
                set1,
                set2,
                keys_walkers[mid:],
            )
            mx.eval(new_logLs_set1)
            mx.eval(new_set1)

            new_logLs_set2, new_set2 = mx.vmap(self.step_walker, in_axes=(0, None, 0))(
                set2,
                new_set1,
                keys_walkers[:mid],
            )
            mx.eval(new_logLs_set2)
            mx.eval(new_set2)

            new_state = mx.concatenate([new_set1, new_set2], axis=0)
            new_logLs = mx.concatenate([new_logLs_set1, new_logLs_set2], axis=0)

            chains.append(new_state)
            logLs.append(new_logLs)

            if (step + 1) % Nsave == 0:
                current_chains = mx.concatenate(
                    [
                        mx.array(logLs).reshape(len(chains), theta_ini.shape[0], 1),
                        mx.array(chains),
                    ],
                    axis=-1,
                )
                np.save(filename, np.array(current_chains))

        return mx.concatenate(
            [
                mx.array(logLs).reshape(len(chains), theta_ini.shape[0], 1),
                mx.array(chains),
            ],
            axis=-1,
        )
