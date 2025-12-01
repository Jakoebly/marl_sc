
class BaseDemandAllocator(object):
    def __init__(self, env_config):
        pass

    def sample(self):
        raise NotImplementedError

class GreedyDemandAllocator(BaseDemandAllocator):
    def __init__(self, env_config):
        pass

    def sample(self):
        pass

def _validate_lp_allocator():
    pass


DEMAND_ALLOCATOR_REGISTRY = {
    "Greedy": {
        "cls": GreedyDemandAllocator,
        "required_params": [],
        "validate": None
    },
    "LP": {
        "cls": None,
        "required_params": [],
        "validate": _validate_lp_allocator
    }
}

def build_demand_allocator(config):
    from env.components.components import build_env_component
    return build_env_component(registry=DEMAND_ALLOCATOR_REGISTRY, comp_config=config)



