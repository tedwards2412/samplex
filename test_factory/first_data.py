import numpy as np
import mlx.core as mx
import matplotlib.pyplot as plt


def generate_data():
    # Parameters we are trying to infer
    m_true = 2.0
    c_true = 3.0

    x = mx.linspace(-5, 5, 100)
    err = mx.random.normal(x.shape)
    y = m_true * x + c_true + err

    def logLikelihood(theta, data):
        m, c = theta
        x, y, sigma = data
        return -0.5 * sum((y - (m * x + c)) ** 2 / sigma**2)

    print(logLikelihood((3.0, 4.0), (x, y, err)))
    print(logLikelihood((m_true, c_true), (x, y, err)))
    print(logLikelihood((1.0, 2.0), (x, y, err)))

    plt.figure(figsize=(10, 5))
    plt.errorbar(
        x.tolist(),
        y.tolist(),
        yerr=mx.abs(err).tolist(),
        fmt=".k",
        capsize=0,
        alpha=0.5,
    )
    plt.plot(x.tolist(), (m_true * x + m_true).tolist(), "-", color="k")
    plt.show()


if __name__ == "__main__":
    generate_data()