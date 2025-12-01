from env.components.demand_allocator import DEMAND_ALLOCATOR_REGISTRY

# Component name-registry mapping
COMPONENT_REGISTRY = {
    "demand_allocator": DEMAND_ALLOCATOR_REGISTRY,
    # "demand_sampler": DEMAND_SAMPLER_REGISTRY,
    # "lead_time_sampler": LEAD_TIME_SAMPLER_REGISTRY,
}

def build_env_component(registry, comp_config):
    """Build an environment component based on its registry and config."""
    model_name = comp_config["model"]
    model_specs = registry[model_name]
    cls = model_specs["cls"]

    kwargs = {k: v for k, v in comp_config.items() if k != "model"}

    return cls(**kwargs)

