import numpy as np
import mlx.core as mx
import matplotlib.pyplot as plt

def initial_condition(a, b):
    return mx.random.uniform(a,b, shape=a.shape)

def proposal_distribution(current, key):
    sigma = 1
    return current + sigma * mx.random.normal(key=key)

def internal_function(x0, key):
    steps = mx.arange(1000)
    xcurrent = x0
    states = []
    step_key = mx.random.split(key, len(steps))
    for numstep, step in enumerate(steps):
        states.append(xcurrent)
        xcurrent = proposal_distribution(xcurrent, step_key[numstep])
    return states

number_ini = 100
x0_array = initial_condition(mx.zeros(number_ini), 10.)

key = mx.random.key(1234)
keys = mx.random.split(key, number_ini)

result = mx.vmap(internal_function, in_axes=(0,0))(x0_array, keys)
result = np.array(result).T
for i in range(len(result)):
    plt.plot(result[i])
plt.show()


# def ran(key):
#     return mx.random.normal(key=key)

# key = mx.random.key(1234)
# keys = mx.random.split(key, 3)
# for _ in mx.arange(10):
#     print(ran(keys[0]))