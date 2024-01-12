import numpy as np
import mlx.core as mx
import matplotlib.pyplot as plt
from math import pi
from tqdm import tqdm

mx.random.seed(np.random.randint(0, 1000))


def initial_condition(a, b):
    return mx.random.uniform(a, b, shape=a.shape)


def proposal_distribution(x, y):
    sigma = mx.array([1])
    return (1 / mx.sqrt(2 * pi * sigma**2))[0] * mx.exp(
        -0.5 * sum((y - x)**2) / sigma[0] ** 2
    )


def sample_proposal_distribution(current, key):
    sigma = 1
    return current + sigma * mx.random.normal(key=key, shape=current.shape)


def target_distributon(x):
    sigma = mx.array([1])
    return (1 / mx.sqrt(2 * pi * sigma**2))[0] * mx.exp(-0.5 * sum(x**2) / sigma[0] ** 2)


def acceptance_probability(current, proposal):
    prob = (
        target_distributon(proposal)
        * proposal_distribution(current, proposal)
        / (target_distributon(current) * proposal_distribution(proposal, current))
    )
    return mx.minimum(1.0, prob)


def internal_function(x0, key):
    steps = mx.arange(3000)
    xcurrent = x0
    states = []
    step_key = mx.random.split(key, len(steps))
    step_key2 = mx.random.split(step_key[0], len(steps))
    for step in tqdm(steps):
        states.append(xcurrent)
        xproposal = sample_proposal_distribution(xcurrent, step_key[step])
        prob = acceptance_probability(xcurrent, xproposal)
        rand = mx.random.uniform(key=step_key2[step])
        xcurrent = mx.where(prob > rand, xproposal, xcurrent)
    return states


number_ini = 5
ndim = 2
x0_array = initial_condition(mx.zeros((number_ini, ndim)) - 5, 5.0)
key = mx.random.key(1234)
keys = mx.random.split(key, number_ini)

result = mx.vmap(internal_function, in_axes=(0, 0))(x0_array, keys)
result = np.array(result)
print(result.shape)

for numwalker in range(number_ini):
    plt.scatter(result[:,numwalker,0], result[:,numwalker,1], s=4)

plt.show()
