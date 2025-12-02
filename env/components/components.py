from env.components.demand_allocator import DEMAND_ALLOCATOR_REGISTRY
from env.components.demand_sampler import DEMAND_SAMPLER_REGISTRY

# Component name-registry mapping
COMPONENT_REGISTRY = {
    "demand_allocator": DEMAND_ALLOCATOR_REGISTRY,
    "demand_sampler": DEMAND_SAMPLER_REGISTRY,
    # "lead_time_sampler": LEAD_TIME_SAMPLER_REGISTRY,
}

def build_env_component(component, registry, env_config):
    """Build an environment component based on its registry and config."""

    # Set values
    comp_config = env_config[component]
    model_name = comp_config["model"]
    model_specs = registry[model_name]
    cls = model_specs["cls"]

    # Get all arguments that are necessary for class instantiation
    kwargs = {k: v for k, v in comp_config.items() if k != "model"}

    # Return the component class
    return cls(env_config, **kwargs)

