import numpy as np
import mlx.core as mx
import matplotlib.pyplot as plt
from samplex.samplex import samplex
from samplex.samplers import emcee_sampler
from samplex.plotting import corner_plot

seed = 1234
mx.random.seed(seed)


def initial_condition(a, b):
    return mx.random.uniform(a, b, shape=a.shape)


def log_target_distribution(theta, data):
    m, c, b = theta
    x, y, sigma = data
    model = b * x**2 + m * x + c
    residual = y - model
    return mx.sum(-0.5 * (residual**2 / sigma**2))


def generate_data(b_true, m_true, c_true):
    x = mx.linspace(-5, 5, 20)
    err = mx.random.normal(x.shape)
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
    # Parameters we are trying to infer
    b_true = 1.0
    m_true = 2.0
    c_true = 3.0
    x, y, err = generate_data(b_true, m_true, c_true)

    logtarget = lambda theta: log_target_distribution(theta, (x, y, err))

    Nwalkers = 32
    Ndim = 3
    Nsteps = 10_000

    theta0_array = initial_condition(mx.zeros((Nwalkers, Ndim)) + 1, 5.0)

    sampler = emcee_sampler(logtarget)
    sam = samplex(sampler, Nwalkers)
    sam.run(Nsteps, theta0_array)
    make_plots(x, y, err, sam, b_true, m_true, c_true)
