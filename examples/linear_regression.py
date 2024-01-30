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


def generate_data():
    # Parameters we are trying to infer
    m_true = 2.0
    c_true = 3.0
    b_true = 1.0

    x = mx.linspace(-5, 5, 100)
    err = mx.random.normal(x.shape)  # * 10
    y = b_true * x**2 + m_true * x + c_true + err

    def log_target_distribution(theta, data):
        m, c, b = theta
        x, y, sigma = data
        model = b * x**2 + m * x + c
        residual = y - model
        return sum(
            -0.5 * (residual**2 / sigma**2)
        )  # + mx.log(mx.sqrt(2 * pi * sigma**2)))

    logtarget = lambda theta: log_target_distribution(theta, (x, y, err))

    Nwalkers = 256
    Ndim = 3
    Nsteps = 5_000
    cov_matrix = mx.array([0.01, 0.01, 0.01])
    jumping_factor = 1.0

    x0_array = initial_condition(mx.zeros((Nwalkers, Ndim)) + 1, 5.0)

    sampler = MH_Gaussian_sampler(logtarget)
    sam = samplex(sampler, Nwalkers)
    result = sam.run(Nsteps, x0_array, cov_matrix, jumping_factor)
    # Should be (Nsteps, Nwalkers, Ndim)
    print(result.shape)

    result_stacked = np.array(sam.get_chain(discard=100, flat=True))

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
        data=result_stacked,
        names=names,
        num_bins=[50, 50, 20],
        lims=lims,
        # interpolation_smoothing=2.0,
        # gaussian_smoothing=0.01
        # fig=None,
        # axes=None,
    )
    plt.show()

    alpha_range = np.linspace(0.1, 1, Nsteps)

    for numwalker in range(Nwalkers):
        plt.scatter(
            result[:, numwalker, 0], result[:, numwalker, 1], s=10, alpha=alpha_range
        )

    plt.show()

    plt.figure(figsize=(10, 5))
    # plt.errorbar(
    #     x.tolist(),
    #     y.tolist(),
    #     yerr=mx.abs(err).tolist(),
    #     fmt=".k",
    #     capsize=0,
    #     alpha=0.5,
    # )
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
    plt.plot(
        x.tolist(),
        (result[-1, 0, 2] * x**2 + result[-1, 0, 0] * x + result[-1, 0, 1]).tolist(),
        "--",
        color="r",
        label="MCMC",
    )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    generate_data()
