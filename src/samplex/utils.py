import numpy as np
import mlx.core as mx


def get_bestfit(chains):
    flattened_chains = chains.reshape(-1, chains.shape[-1])
    idx = mx.argmin(flattened_chains[:, 0])
    return flattened_chains[idx]
