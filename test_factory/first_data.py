import numpy as np
import mlx.core as mx
import matplotlib.pyplot as plt
from walker_ND import samplex
from math import pi


def generate_data():
    # Parameters we are trying to infer
    m_true = 2.0
    c_true = 2.0

    x = mx.linspace(-5, 5, 100)
    err = mx.random.normal(x.shape)
    y = m_true * x + c_true + err

    def logLikelihood(theta, data):
        m, c = theta
        x, y, sigma = data
        # return -0.5 * sum((y - (m * x + c)) ** 2 / sigma**2)
        # return sum(theta ** 2)
        # sigma = mx.array([1])
        # return mx.prod((1 / mx.sqrt(2 * pi * sigma**2)) * mx.exp(-0.5 * (y - (m * x + c))**2 / sigma ** 2))
        return mx.exp(sum(-0.5 * (((y - (m * x + c))**2 / sigma ** 2) + mx.sqrt(2 * pi * sigma**2))))
    
    logL = lambda theta: logLikelihood(theta, (x, y, err))

    Nwalkers = 2
    Ndim = 2
    Nsteps = 1000

    sam = samplex(Nwalkers, Ndim, logL)
    result = sam.run(Nsteps)
    alpha_range = np.linspace(0.1, 1, Nsteps)

    for numwalker in range(Nwalkers):
        # plt.scatter(result[0,numwalker,0], result[0,numwalker,1], s=1000, color="r")
        plt.plot(result[:,numwalker,0], result[:,numwalker,1])

    plt.show()

    print(logL((30.0, 100.0)))
    print(logL((m_true, c_true)))
    print(logL((1.0, 2.0)))

    # plt.figure(figsize=(10, 5))
    # plt.errorbar(
    #     x.tolist(),
    #     y.tolist(),
    #     yerr=mx.abs(err).tolist(),
    #     fmt=".k",
    #     capsize=0,
    #     alpha=0.5,
    # )
    # plt.plot(x.tolist(), (m_true * x + m_true).tolist(), "-", color="k")
    # plt.show()


if __name__ == "__main__":
    generate_data()
