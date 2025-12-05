from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import pickle
from env.schemas import OrdersBatch
from src.settings import DATA_PATH

class BaseDemandSampler(ABC):
    def __init__(self, env_config):

        # Set general attributes
        self.n_warehouses = env_config["n_warehouses"]
        self.n_regions = env_config["n_regions"]
        self.n_skus = env_config["n_skus"]
        self.episode_length = env_config["episode_length"]

    def reset(self, episode_length):
        self.episode_length = episode_length

    @abstractmethod
    def sample_timestep(self, timestep):
        raise NotImplementedError

class EmpiricalSampler(BaseDemandSampler):
    def __init__(self, env_config, data_path=None):

        # Initialize base class
        super().__init__(env_config)

        # Set the data path where historical orders are stored
        self.data_path = Path(data_path if data_path is not None else DATA_PATH)

        # TODO: Uncomment when preprocessing is implemented
        # Load the historical orders
        self.historical_orders = self._load_historical_orders(data_path)

        # Get the number of total number of timesteps that are available in the data
        self.timesteps_data = len(self.historical_orders)

        # Initialize the sample window start index
        self._window_start_idx = None

    @staticmethod
    def _load_historical_orders(path):

        # Load the pickle object
        with open(path, "rb") as file:
            historical_orders = pickle.load(file)

        # Check if historical_orders is a list
        if not isinstance(historical_orders, list):
            raise TypeError(f"historical_orders must be list[OrdersBatch], got {type(historical_orders)}")

        # Check if historical_orders contains only OrderBatch elements
        for i, item in enumerate(historical_orders):
            if not isinstance(item, OrdersBatch):
                raise ValueError(f"Element {i} in historical_orders is not OrdersBatch, got {type(item)}")

        return historical_orders

    def reset(self, episode_length):
        """Resets the empirical demand sampler at the beginning of an episode."""

        # Call base reset method
        super().reset(episode_length)

        # Ensure that episode_length does not exceed the available timesteps in the data
        if episode_length > self.timesteps_data:
            raise ValueError(
                f"episode_length must be smaller than the available data of size {self.timesteps_data}, "
                f"got {episode_length}"
            )

        # TODO: should it actually be contiguous or should I maybe choose episode_length integers and shuffle them?
        # Sample the start index for a random contiguous time window of
        # size episode_length that will be used for demand sampling
        self._window_start_idx = np.random.randint(0, self.timesteps_data - self.episode_length + 1)

    def sample_timestep(self, timestep):
        """Samples demand for one timestep according to historical orders."""

        # Ensure a window start index is set
        if self._window_start_idx is None:
            raise ValueError(
                f"window_start_idx must be set, got {self._window_start_idx}. "
                             f"Run reset() before sampling."
            )

        # Determine the current timestep index
        idx = self._window_start_idx + timestep

        # Return the corresponding OrderBatch from historical orders
        return self.historical_orders[idx]

class PoissonSampler(BaseDemandSampler):
    def __init__(self, env_config, lambda_orders, lambda_skus):
        # Initialize base class
        super().__init__(env_config)

        # Set lambda parameters needed for the Poisson distribution
        self.lambda_orders = lambda_orders
        self.lambda_skus = np.asarray(lambda_skus, dtype=np.float64)

    def reset(self, episode_length):
        """Resets the poisson demand sampler at the beginning of an episode."""

        # Call base reset method
        super().reset(episode_length)

    def sample_timestep(self, timestep):
        """Samples demand for one timestep according to a poisson distribution."""

        # Sample number of orders in one timestep
        n_orders = np.random.poisson(self.lambda_orders)

        # Assign each order a random demand region
        order_regions = np.random.randint(0, self.n_regions, size=n_orders)

        # Sample random SKU demand for each order
        orders = np.random.poisson(self.lambda_skus, size=(n_orders, self.n_skus))

        # Remove orders with no SKU demand
        non_zero_orders = orders.sum(axis=1) > 0
        orders = orders[non_zero_orders]
        order_regions = order_regions[non_zero_orders]

        # Return the demand as OrderBatch
        return OrdersBatch(orders=orders, order_regions=order_regions)

def _validate_empirical_sampler(params, env_config):
    """Validates parameters of the empirical sampler for type and shape."""

    # Check if the optional data_path parameter and the default value are path-like
    data_path = params["data_path"] if "data_path" in params else DATA_PATH
    if not isinstance(data_path, (str, Path)):
        raise ValueError(f"specified or default data_path must be path-like / string, got {data_path}")

    # Set data_path as a Path
    data_path = Path(data_path)

    # TODO: uncomment if preprocessing is implemented
    # Check if the specified data file exists
    #if not data_path.is_file():
    #    raise ValueError(
    #        f"Empirical demand_sampler expects historical_orders file at: {data_path}, but no such file was found. "
    #        f"Run preprocessing first."
    #    )

def _validate_poisson_sampler(params, env_config):
    """Validates required parameters of the Poisson sampler for type and shape."""

    # Set values
    n_skus = env_config["n_skus"]
    lambda_orders = np.asarray(params["lambda_orders"])
    lambda_skus = np.asarray(params["lambda_skus"])

    # Check lambda_orders for type and shape
    if not np.issubdtype(lambda_orders.dtype, np.number) or np.any(lambda_orders < 0):
        raise ValueError(f"demand_sampler.lambda_orders must be a non-negative number, got {lambda_orders}.")
    if lambda_orders.shape != ():
        raise ValueError(f"demand_sampler.lambda_orders must have shape scalar (), got {lambda_orders.shape}.")

    # Check lambda_skus for type and shape
    if not np.issubdtype(lambda_skus.dtype, np.number) or np.any(lambda_skus < 0):
        raise ValueError(f"demand_sampler.lambda_skus must be a non-negative number, got {lambda_skus}.")
    if lambda_skus.shape not in [(), (n_skus,)]:
        raise ValueError(
            f"demand_sampler.lambda_skus must have shape scalar () or (n_skus,)={(n_skus,)}, "
            f"got {lambda_skus.shape}."
        )

DEMAND_SAMPLER_REGISTRY = {
    "Empirical": {
        "cls": EmpiricalSampler,
        "required_params": [],
        "optional_params": ["data_path"],
        "validate": _validate_empirical_sampler,
        "desc": "Samples demand from historical orders. If data_path is not provided, defaults to DATA_PATH from src.settings."
    },
    "Poisson": {
        "cls": PoissonSampler,
        "required_params": ["lambda_orders", "lambda_skus"],
        "optional_params": [],
        "validate": _validate_poisson_sampler,
        "desc": "Samples demand from Poisson order number and SKU rates."
    }
}

def build_demand_sampler(env_config):
    from env.components.components import build_env_component
    return build_env_component(component="demand_sampler", registry=DEMAND_SAMPLER_REGISTRY, env_config=env_config)


