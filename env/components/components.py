
def build_env_component(component_group, component, registry, env_config):
    """Builds an environment component based on its registry and config."""

    # Get component config and values
    comp_config = env_config["components"][component_group][component]
    model_name = comp_config["model"]
    model_specs = registry[model_name]
    cls = model_specs["cls"]

    # Get all arguments that are necessary for class instantiation
    kwargs = {k: v for k, v in comp_config.items() if k != "model"}

    # Return the component class
    return cls(env_config, **kwargs)

def validate_env_component(component_group, component_name, registry, env_config, extra_params=None):
    """Validates an environment component based on its registry and config."""

    # Check if the necessary component section exists
    components = env_config.get("components")
    if components is None:
        raise ValueError("Missing 'components' section in env_config.")

    group_config = components.get(component_group)
    if group_config is None:
        raise ValueError(f"Missing 'components.{component_group}' section in env_config.")

    comp_config = group_config.get(component_name)
    if comp_config is None:
        raise ValueError(
            f"Missing 'components.{component_group}.{component_name}' section in env_config."
        )

    # Validate the registry block corresponding to the component
    comp_config = env_config["components"][component_group][component_name]
    context = f"components.{component_group}.{component_name}"
    validate_registry_block(registry, comp_config, context, env_config, extra_params)

def validate_registry_block(registry, config, context, env_config, extra_params=None):
    """Validates a specific component in a config against its registry block."""

    # Check if model field exists
    model_name = config.get("model")
    if model_name is None:
        raise ValueError(f"{context} config must contain a 'model', got '{model_name}'.")

    # Check if the model is registered
    if model_name not in registry:
        raise ValueError(f"Unknown model '{model_name}' in {context}. Allowed: {list(registry.keys())}")

    model_specs = registry[model_name]

    # Check existence of required parameters
    required = model_specs.get("params", [])
    if extra_params:
        required += extra_params
    params = {name: value for name, value in config.items() if name != "model"}
    missing = [param for param in required if param not in params]
    if missing:
        raise ValueError(
            f"Missing required parameter(s) for {context} model '{model_name}': {missing}"
        )

    # Check if unallowed parameters are present
    allowed = ["model"] + required
    unknown = [param for param in config.keys() if param not in allowed]
    if unknown:
        raise ValueError(
            f"{context}.model '{model_name}' has unknown parameter(s): {unknown}. "
            f"Allowed parameter(s): {allowed}."
        )

    # Validate type and shape of parameters
    validate_model_params = model_specs.get("validate")
    if validate_model_params is not None:
        validate_model_params(params, env_config)








