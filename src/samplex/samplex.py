import numpy as np
import mlx.core as mx


class samplex:
    def __init__(self, Sampler, likelihood, prior):
        self.Sampler = Sampler
        self.likelihood = likelihood
        self.prior = prior
