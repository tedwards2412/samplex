import numpy as np
import mlx.core as mx
import matplotlib.pyplot as plt
from samplex.samplex import samplex
from samplex.samplers import MH_Gaussian_sampler
from samplex.plotting import corner_plot

seed = 1234
mx.random.seed(seed)


def initial_condition(a, b):
    return mx.random.uniform(a, b, shape=a.shape)


def uniform_prior(prior_min, prior_max):
    height = 1 / (prior_max - prior_min)
    return lambda a: mx.where(
        (a < prior_max) & (a > prior_min), mx.array([height]), mx.array([0.0])
    )


def log_likelihood(theta, data):
    m, c, b = theta
    x, y, sigma = data
    model = b * x**2 + m * x + c
    residual = y - model
    return sum(-0.5 * (residual**2 / sigma**2))


def log_prior(theta, uniform_priors):
    log_prior = 0.0
    for i, p in enumerate(uniform_priors):
        log_prior = log_prior + mx.log(p(theta[i]))
    return log_prior


def posterior(theta, data, priors):
    return log_likelihood(theta, data) + log_prior(theta, priors)


def generate_data(m_true, c_true, b_true):
    x = mx.linspace(-5, 5, 20)
    err = mx.random.normal(x.shape)  # * 10
    y = b_true * x**2 + m_true * x + c_true + err
    return x, y, err


def make_plots(x, y, err, sam, b_true, m_true, c_true):
    # Plot cornerplot and bestfit

    burned_inchains = np.array(sam.get_chains(remove_burnin=True))

    names = [
        r"$m$",
        r"$c$",
        r"$b$",
    ]
    lims = [
        [1.8, 2.2],
        [2.8, 3.2],
        [0.0, 2.0],
    ]

    fig, axes = corner_plot(
        data=burned_inchains[:, 1:],
        names=names,
        num_bins=[50, 50, 20],
        lims=lims,
    )
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.errorbar(
        x.tolist(),
        y.tolist(),
        yerr=mx.abs(err).tolist(),
        ls="None",
        marker=".",
        markersize=5,
        markeredgewidth=1.5,
        elinewidth=1.5,
        color="black",
        zorder=0,
    )

    plt.plot(
        x.tolist(),
        (b_true * x**2 + m_true * x + c_true).tolist(),
        "-",
        color="k",
        label="truth",
    )
    bestfit = sam.get_bestfit()
    plt.plot(
        x.tolist(),
        (bestfit[3] * x**2 + bestfit[1] * x + bestfit[2]).tolist(),
        "--",
        color="r",
        label="MCMC",
    )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    m_true = 2.0
    c_true = 3.0
    b_true = 1.0
    x, y, err = generate_data(m_true, c_true, b_true)

    m_min, m_max = 0.0, 4.0
    c_min, c_max = 1.0, 5.0
    b_min, b_max = 0.0, 2.0
    priors = [
        uniform_prior(m_min, m_max),
        uniform_prior(c_min, c_max),
        uniform_prior(b_min, b_max),
    ]

    logtarget = lambda theta: posterior(theta, (x, y, err), priors)

    Nwalkers = 32
    Ndim = 3
    Nsteps = 10_000
    cov_matrix = mx.array([0.01, 0.01, 0.01])
    jumping_factor = 1.0

    x0_array = mx.random.uniform(
        mx.array([m_min, c_min, b_min]),
        mx.array([m_max, c_max, b_max]),
        (Nwalkers, Ndim),
    )

    sampler = MH_Gaussian_sampler(logtarget)
    sam = samplex(sampler, Nwalkers)
    sam.run(Nsteps, x0_array, cov_matrix, jumping_factor)
    make_plots(x, y, err, sam, b_true, m_true, c_true)
