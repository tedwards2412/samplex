import numpy as np
import mlx.core as mx
import matplotlib.pyplot as plt
from samplex.samplex import samplex
from samplex.samplers.emcee_sampler import emcee_sampler
from samplex.plotting import corner_plot
import emcee
import time
from multiprocessing import Pool

# import os

# os.environ["OMP_NUM_THREADS"] = "1"

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


def log_target_distribution_np(theta, data):
    m, c, b = theta
    x, y, sigma = data
    model = b * x**2 + m * x + c
    residual = y - model
    return np.sum(-0.5 * (residual**2 / sigma**2))


def generate_data(b_true, m_true, c_true):
    x = mx.linspace(-5, 5, 20)
    err = mx.random.normal(x.shape)
    y = b_true * x**2 + m_true * x + c_true + err
    return x, y, err


def generate_data_np(b_true, m_true, c_true):
    x = np.linspace(-5, 5, 20)
    err = np.random.normal(x.shape)
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


def run_emcee(ndim, nwalkers, Nsteps, p0, data):
    # sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    # sampler.run_mcmc(p0, Nsteps, progress=True)
    # return None

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, logtarget_np, pool=pool, args=[data]
        )
        sampler.run_mcmc(p0, Nsteps, progress=True)
    return None


def logtarget_np(theta, data):
    return log_target_distribution_np(theta, data)


if __name__ == "__main__":
    # Parameters we are trying to infer
    b_true = 1.0
    m_true = 2.0
    c_true = 3.0
    x, y, err = generate_data(b_true, m_true, c_true)
    x_np, y_np, err_np = generate_data_np(b_true, m_true, c_true)

    logtarget = lambda theta: log_target_distribution(theta, (x, y, err))

    Nwalkers = np.linspace(6, 512, 2, dtype=int)
    Nsteps = np.linspace(1000, 5000, 2, dtype=int)
    # Nsteps = 2_000
    Ndim = 3
    # samplex_times = []
    samplex_times_GPU = []
    emcee_times = []

    for nwalkers in Nwalkers:
        samplex_times_GPU_temp = []
        emcee_times_temp = []
        for nsteps in Nsteps:

            # print("Running samplex CPU...")
            theta0_array = initial_condition(mx.zeros((nwalkers, Ndim)) + 1, 5.0)
            sampler = emcee_sampler(logtarget)
            # sam = samplex(sampler, nwalkers, device=mx.cpu)
            # start_time = time.time()
            # sam.run(Nsteps, theta0_array)
            # samplex_time = time.time() - start_time
            # samplex_times.append(samplex_time)

            print("Running samplex GPU...")
            sam = samplex(sampler, nwalkers, device=mx.gpu)
            start_time = time.time()
            sam.run(nsteps, theta0_array)
            samplex_time_GPU = time.time() - start_time
            samplex_times_GPU_temp.append(samplex_time_GPU)

            print("Running emcee...")
            theta0_array_np = np.random.uniform(1, 5, size=(nwalkers, Ndim))
            start_time = time.time()
            run_emcee(Ndim, nwalkers, nsteps, theta0_array_np, (x_np, y_np, err_np))
            emcee_time = time.time() - start_time
            emcee_times_temp.append(emcee_time)

        emcee_times.append(emcee_times_temp)
        samplex_times_GPU.append(samplex_times_GPU_temp)

    samplex_times_GPU_array = np.array(samplex_times_GPU)

    plt.figure(figsize=(10, 5))
    # Create the contour plot
    X, Y = np.meshgrid(Nwalkers, Nsteps)
    contour = plt.contourf(
        X, Y, samplex_times_GPU_array.T, cmap="viridis", levels=20
    )  # Transpose to match dimensions
    plt.colorbar(contour)
    plt.title("Samplex Execution Time (GPU)")
    plt.xlabel("Number of Walkers")
    plt.ylabel("Number of Steps")
    plt.show()

    # plt.plot(Nwalkers, samplex_times_GPU, label="samplex GPU")
    # plt.plot(Nwalkers, emcee_times, label="emcee")
    # plt.xlabel("Number of walkers")
    # plt.ylabel("Time (s)")
    # plt.legend()
    # plt.show()


# if __name__ == "__main__":
#     # Parameters we are trying to infer
#     b_true = 1.0
#     m_true = 2.0
#     c_true = 3.0
#     x, y, err = generate_data(b_true, m_true, c_true)
#     x_np, y_np, err_np = generate_data_np(b_true, m_true, c_true)

#     logtarget = lambda theta: log_target_distribution(theta, (x, y, err))

#     Nwalkers = [2**n for n in range(4, 10)]
#     Ndim = 3
#     Nsteps = 2_000
#     samplex_times = []
#     samplex_times_GPU = []
#     emcee_times = []

#     for nwalkers in Nwalkers:

#         # print("Running samplex CPU...")
#         theta0_array = initial_condition(mx.zeros((nwalkers, Ndim)) + 1, 5.0)
#         sampler = emcee_sampler(logtarget)
#         # sam = samplex(sampler, nwalkers, device=mx.cpu)
#         # start_time = time.time()
#         # sam.run(Nsteps, theta0_array)
#         # samplex_time = time.time() - start_time
#         # samplex_times.append(samplex_time)

#         print("Running samplex GPU...")
#         sam = samplex(sampler, nwalkers, device=mx.gpu)
#         start_time = time.time()
#         sam.run(Nsteps, theta0_array)
#         samplex_time_GPU = time.time() - start_time
#         samplex_times_GPU.append(samplex_time_GPU)

#         print("Running emcee...")
#         theta0_array_np = np.random.uniform(1, 5, size=(nwalkers, Ndim))
#         start_time = time.time()
#         run_emcee(Ndim, nwalkers, Nsteps, theta0_array_np, (x_np, y_np, err_np))
#         emcee_time = time.time() - start_time
#         emcee_times.append(emcee_time)

#     plt.figure(figsize=(10, 5))
#     # plt.plot(Nwalkers, samplex_times, label="samplex CPU")
#     plt.plot(Nwalkers, samplex_times_GPU, label="samplex GPU")
#     plt.plot(Nwalkers, emcee_times, label="emcee")
#     plt.xlabel("Number of walkers")
#     plt.ylabel("Time (s)")
#     plt.legend()
#     plt.show()
