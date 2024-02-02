import numpy as np
import mlx.core as mx
import samplex.utils as utils


class samplex:
    """
    A class for managing and executing an MCMC sampler using a specified sampler within the mlx.core framework.

    Attributes:
        Nwalkers (int): Number of walkers to use in the sampling process.
        sampler (object): The sampler object responsible for generating samples.
        key (mx.ndarray): A random key for initializing the random number generator.
        keys (mx.ndarray): Array of split random keys for each walker.
        chains (mx.ndarray or None): The chains generated after running the sampler, initially None.
        device (mx.cpu or mx.gpu): The device (CPU/GPU) where computations will be executed.

    Methods:
        run(Nsteps, theta_ini, cov_matrix, jumping_factor): Executes the sampling process.
        get_chain(discard, thin, flat): Retrieves the generated chains, applying thinning and discarding as specified.
        reset(): Resets the chains to None, allowing for a fresh start.
        save_chains(filename): Saves the generated chains to a file.
        get_bestfit(): Computes and returns the best-fit parameters from the generated chains.
    """

    def __init__(self, sampler, Nwalkers, foldername="MyChains", device=mx.cpu):
        """
        Initializes the samplex class with the specified sampler, number of walkers, and device for the mlx.core framework.

        Parameters:
            sampler (object): The sampler object to be used for generating samples.
            Nwalkers (int): Number of walkers to be used in the sampling process.
            device (optional): The device (CPU/GPU) for computation. Defaults to mx.cpu.
        """
        self.Nwalkers = Nwalkers
        self.sampler = sampler

        self.key = mx.random.key(1234)
        self.keys = mx.random.split(self.key, self.Nwalkers)
        self.chains = None
        self.filename = utils.generate_filename(foldername)
        self.logL_threshold = 3.0

        mx.set_default_device(device)

    def run(self, Nsteps, theta_ini, cov_matrix, jumping_factor, Nsave=1000):
        """
        Executes the sampling process with the specified parameters using mlx.core.

        Parameters:
            Nsteps (int): Number of steps for each walker in the sampling process.
            theta_ini (mx.ndarray): Initial parameter values for the walkers.
            cov_matrix (mx.ndarray): Covariance matrix used for the proposal distribution.
            jumping_factor (float): Scaling factor for the proposal distribution.

        Returns:
            mx.ndarray: The generated chains after running the sampling process.
        """
        self.chains = self.sampler.run(
            Nsteps=Nsteps,
            key=self.key,
            theta_ini=theta_ini,
            cov_matrix=cov_matrix,
            jumping_factor=mx.array([jumping_factor]),
            Nsave=Nsave,
            filename=self.filename,
            full_chains=self.chains,
        )

        return self.chains

    def get_chains(self, discard=0, thin=1, flat=True, remove_burnin=False):
        """
        Retrieves the generated chains, with options to discard initial steps, thin the chains, and flatten the result.

        Parameters:
            discard (int, optional): Number of initial steps to discard from each chain. Defaults to 0.
            thin (int, optional): Factor by which to thin the chains. A value of n means every nth sample is retained. Defaults to 1.
            flat (bool, optional): If True, the chains are flattened into a 2D array; otherwise, they are returned as is. Defaults to True.

        Returns:
            mx.ndarray: The processed chains according to the specified parameters.

        Raises:
            ValueError: If no chains have been generated yet.
        """
        if self.chains is None:
            raise ValueError("No chains have been generated yet!")

        if remove_burnin:
            for numw, walker in enumerate(self.chains.transpose(1, 0, 2)):
                minlogL = mx.min(walker[:, 0])
                l = 0
                while walker[l, 0] - minlogL > self.logL_threshold:
                    l += 1
                if l == len(walker):
                    print(f"Walker {numw} is not burned in, removing...")
                    continue
                else:
                    try:
                        burnedin_chains = mx.concatenate(
                            (burnedin_chains, walker[l:, :])
                        )
                    except:
                        burnedin_chains = walker[l:, :]
            return burnedin_chains

        else:
            if flat:
                return self.chains[discard::thin].reshape(-1, self.chains.shape[-1])
            else:
                return self.chains[discard::thin]

    def reset(self):
        """
        Resets the generated chains to None, allowing for a fresh start of the sampling process within mlx.core.
        """
        self.chains = None

    def save_chains(self, filename):
        """
        Saves the generated chains to a file in NumPy's binary format, ensuring compatibility with mlx.core arrays.

        Parameters:
            filename (str): The name of the file to save the chains to.
        """
        np.save(filename, np.array(self.chains))

    def get_bestfit(self):
        """
        Computes and returns the best-fit parameters from the generated chains using a utility function compatible with mlx.core.

        Returns:
            The best-fit parameters derived from the chains.
        """
        return utils.get_bestfit(self.chains)
