from abc import ABC, abstractmethod
import numpy as np


# TODO: add same comments as in demand_sampler.py

class BaseDemandAllocator(ABC):
    def __init__(self, env_config):
        self.n_warehouses = env_config["n_warehouses"]
        self.n_skus = env_config["n_skus"]
        self.shipment_costs = np.asarray(env_config["shipment_costs"])

    @abstractmethod
    def allocate(self):
        raise NotImplementedError

class GreedyDemandAllocator(BaseDemandAllocator):
    def __init__(self, env_config):
        super().__init__(env_config)

    def allocate(self):
        pass


# TODO: add any allocator-specific validation functions here

DEMAND_ALLOCATOR_REGISTRY = {
    "Greedy": {
        "cls": GreedyDemandAllocator,
        "required_params": [],
        "validate": None
    }
}

def build_demand_allocator(env_config):
    from env.components.components import build_env_component
    return build_env_component(component="demand_allocator", registry=DEMAND_ALLOCATOR_REGISTRY, env_config=env_config)



