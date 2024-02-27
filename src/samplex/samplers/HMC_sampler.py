import mlx.core as mx
from tqdm import tqdm
import numpy as np


class HMC_sampler:
    def __init__(self, log_target_distribution):
        self.log_target_distribution = log_target_distribution

    # def initialize_chains(self, theta_ini):
    #     minuslogLs = [-1.0 * mx.vmap(self.log_target_distribution)(theta_ini)]
    #     return minuslogLs, [theta_ini]

    def initialize_chains(self, theta_ini):
        minuslogLs = [-1.0 * self.log_target_distribution(theta_ini)]
        return minuslogLs, [theta_ini]

    def potential(self, theta, grad=False):
        if grad:
            return -mx.grad(self.log_target_distribution)(theta)
        return -self.log_target_distribution(theta)

    def kinetic(self, momenta):
        return 0.5 * momenta @ self.inverse_mass_matrix @ momenta

    def hamiltonian(self, theta, momenta):
        return self.potential(theta) + self.kinetic(momenta)

    def sample_momenta(self, key):
        subkeys = mx.random.split(key, self.diagonal_mass.shape[0])
        return mx.array(
            [
                mx.random.normal(key=subkey, scale=mi.tolist() ** 0.5)
                for subkey, mi in zip(subkeys, self.diagonal_mass)
            ]
        )

    def sample_theta(self, a, b):
        return mx.random.uniform(a, b, shape=a.shape)

    def generate_new_step_traj(self):
        # if self.min_step is None:
        #     min_step = mx.random.uniform(0.01, 0.02)
        # else:
        #     min_step = self.min_step
        # if self.max_step is None:
        #     max_step = mx.random.uniform(0.07, 0.18)
        # else:
        #     max_step = self.max_step
        # if self.max_traj is None:
        #     max_traj = mx.random.uniform(18, 25)
        # else:
        #     max_traj = self.max_traj
        # if self.min_traj is None:
        #     min_traj = mx.random.uniform(1, 18)
        # else:
        #     min_traj = self.min_traj

        # step_size = mx.random.uniform(min_step, max_step)
        # traj_size = int(mx.random.uniform(min_traj, max_traj).tolist())
        step_size = 0.001
        traj_size = 15
        return step_size, traj_size

    def leapfrog(self, theta_init, momenta_init):
        step_size, traj_size = self.generate_new_step_traj()
        momenta = momenta_init - 0.5 * step_size * self.potential(theta_init, grad=True)
        theta = theta_init + step_size * momenta / self.diagonal_mass
        for _ in range(traj_size - 1):
            momenta = momenta - 0.5 * step_size * self.potential(theta, grad=True)
            theta = theta + step_size * momenta / self.diagonal_mass
        momenta = momenta - 0.5 * step_size * self.potential(theta, grad=True)
        # print("potential gradient", self.potential(theta_init, grad=True))
        # print("potential", self.potential(theta_init))
        # print("kinetic", self.kinetic(momenta_init), self.kinetic(momenta))
        # print("current", theta_init, momenta_init)
        # print("proposed", theta, momenta)
        return theta, momenta

    def acceptance_probability(self, theta, theta_proposed, momenta, momenta_proposed):
        H_proposed = self.hamiltonian(theta_proposed, momenta_proposed)
        H_current = self.hamiltonian(theta, momenta)
        # print("Hams:", H_proposed, H_current, mx.exp(-H_proposed) / mx.exp(-H_current))
        return mx.minimum(1.0, mx.exp(-H_proposed) / mx.exp(-H_current))

    def single_step(self, state, step_key, step_key2):
        momenta = self.sample_momenta(step_key)
        state_proposed, momenta_proposed = self.leapfrog(state, momenta)
        prob = self.acceptance_probability(
            state, state_proposed, momenta, momenta_proposed
        )
        rand = mx.random.uniform(
            key=step_key2,
        )
        new_state = mx.where(
            prob > rand,
            state_proposed,
            state,
        )
        return self.log_target_distribution(new_state), new_state

    def step_walker(self, current_state, key):
        prob_keys = mx.random.split(key, 2)
        new_state = self.single_step(current_state, prob_keys[0], prob_keys[1])
        return new_state[0], new_state[1]

    def run(
        self,
        Nsteps,
        key,
        theta_ini,
        Nsave=1000,
        filename="MyChains.npy",
        full_chains=None,
        **kwargs,
    ):
        self.diagonal_mass = kwargs.get("diagonal_mass", mx.ones(theta_ini.shape[1]))
        self.inverse_mass_matrix = mx.diag(1 / self.diagonal_mass)
        self.min_step = kwargs.get("min_step", None)
        self.max_step = kwargs.get("max_step", None)
        self.min_traj = kwargs.get("min_traj", None)
        self.max_traj = kwargs.get("max_traj", None)

        if full_chains is None:
            logLs, chains = self.initialize_chains(theta_ini[0])
        else:
            logLs, chains = (
                list(full_chains[:, :, 0]),
                list(full_chains[:, :, 1:]),
            )
        steps = mx.arange(Nsteps)
        keys = mx.random.split(key, Nsteps)
        for step in tqdm(steps):
            keys_walkers = mx.random.split(keys[step], theta_ini.shape[0])
            new_logLs, new_state = self.step_walker(
                chains[-1],
                keys_walkers[0],
            )
            mx.eval(new_logLs)
            mx.eval(new_state)
            chains.append(new_state)
            logLs.append(new_logLs)
            # print(new_state)

            # if (step + 1) % Nsave == 0:
            #     current_chains = mx.concatenate(
            #         [
            #             mx.array(logLs).reshape(len(chains), theta_ini.shape[0], 1),
            #             mx.array(chains),
            #         ],
            #         axis=-1,
            #     )
            #     np.save(filename, np.array(current_chains))
        # for step in tqdm(steps):
        #     keys_walkers = mx.random.split(keys[step], theta_ini.shape[0])
        #     new_logLs, new_state = mx.vmap(self.step_walker, in_axes=(0, 0))(
        #         chains[-1],
        #         keys_walkers,
        #     )
        #     mx.eval(new_logLs)
        #     mx.eval(new_state)
        #     chains.append(new_state)
        #     logLs.append(new_logLs)

        #     if (step + 1) % Nsave == 0:
        #         current_chains = mx.concatenate(
        #             [
        #                 mx.array(logLs).reshape(len(chains), theta_ini.shape[0], 1),
        #                 mx.array(chains),
        #             ],
        #             axis=-1,
        #         )
        #         np.save(filename, np.array(current_chains))
        print(mx.array(logLs).shape, mx.array(chains).shape)

        # return mx.concatenate(
        #     [
        #         mx.array(logLs).reshape(len(chains), theta_ini.shape[0], 1),
        #         mx.array(chains),
        #     ],
        #     axis=-1,
        # )
        return mx.concatenate(
            [
                mx.array(logLs).reshape(len(chains), 1),
                mx.array(chains),
            ],
            axis=-1,
        ).reshape(len(chains), 1, theta_ini.shape[1] + 1)
        #### HOW TO DEAL WITH GRADIENTS???
