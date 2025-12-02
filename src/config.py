from typing import Callable, Optional, Any, cast
from unittest import removeResult

import yaml
import numpy as np

from env.components.components import COMPONENT_REGISTRY

def load_config(config_path):
    """Loads and validates a .yaml config file"""

    # Open .yaml file
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # return the validated config
    return _validate_config(config)

def _validate_config(config):
    """Validates configs based on type"""

    # Get the config type (e.g., env, algo, ...)
    config_type = config.get("type")

    # Validate configs by using type-specific helper functions
    # TODO: add other configs
    if config_type == "env":
        return _validate_env_config(config)
    else:
        raise ValueError(f"Unknown config type: {config_type}")

def _validate_env_config(env_config):
    """Validates the environment config"""

    # Validate core fields
    _validate_core_env_fields(env_config)

    # Validate environment components
    for comp, reg in COMPONENT_REGISTRY.items():
        _validate_env_component(component=comp, registry=reg, env_config=env_config)

    return env_config

def _validate_core_env_fields(env_config):
    """Validates the core fields of the environment config for presence and type"""

    # Check required scalar fields for presence and type
    for field in ["n_warehouses", "n_skus","episode_length"]:
        if field not in env_config:
            raise ValueError(f"Missing required field {field} in env_config.")
        value = env_config[field]
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"Field {field} in env_config must be a positive integer, got {value}.")

    n_warehouses = env_config["n_warehouses"]

    # Check initial inventory parameters for type
    if "init_inv_mean" in env_config:
        if not isinstance(env_config["init_inv_mean"], (int,float)):
            raise ValueError(
                f"init_inv_mean in env_config must be a number, "
                f"got {env_config["init_inv_mean"]}."
                )
    if "init_inv_std" in env_config:
        if not isinstance(env_config["init_inv_std"], (int,float)) or env_config["init_inv_std"] < 0:
            raise ValueError(
                f"init_inv_mean in env_config must be a non-negative number, "
                f"got {env_config["init_inv_std"]}."
                )

    # Check cost fields for type and shape
    for cost in ["holding_costs", "lost_sales_costs", "shipment_costs"]:
        cost_array = np.asarray(env_config[cost])
        if not np.issubdtype(cost_array.dtype, np.number) or np.any(cost_array < 0):
            raise ValueError(f"{cost} in env_config must be positive number(s), got {cost_array}")
        if cost == "shipment_costs" and cost_array.shape != (n_warehouses, n_warehouses):
            raise ValueError(
                f"{cost} must have shape scalar () or (n_warehouses, n_warehouses)={(n_warehouses, n_warehouses)}, "
                f" got {cost_array.shape}."
            )
        if cost != "shipment_costs" and cost_array.shape not in [(), (n_warehouses,)]:
            raise ValueError(
                f"{cost} must have shape scalar () or (n_warehouses,)={(n_warehouses, )}, got {cost_array.shape}."
            )

def _validate_env_component(component, registry, env_config):
    """Validates environment components (e.g., demand_sampler, ...) for name and parameters"""

    # Get component-specific config
    comp_config = env_config[component]

    # Make sure the specified component model exists
    model_name = comp_config.get("model")
    if model_name not in registry:
        raise ValueError(
            f"Unknown {component} model '{model_name}'. Allowed: {list(registry.keys())}"
        )

    # Get the specs for the given component model
    model_specs = registry[model_name]

    # Check if all required parameters for the component model are provided
    required_params = model_specs.get("required_params") or []
    print(required_params)
    missing = [param for param in required_params if param not in comp_config]
    if missing:
        raise ValueError(
            f"At least one required parameter for {component} model '{model_name}' is missing. "
            f"Missing parameter(s): {missing}."
        )

    # Check that no other parameters are provided
    allowed = ["model"] + required_params
    unknown = [param for param in comp_config.keys() if param not in allowed]
    if unknown:
        raise ValueError(
            f"{component} model '{model_name}' has unknown parameter(s): {unknown}. "
            f"Allowed parameter(s): {allowed}."
        )

    params = {name: comp_config[name] for name in comp_config}

    # Validate model-specific parameters for shape and type consistency, if present
    validate = cast(Optional[Callable[[dict, dict], None]], model_specs["validate"])
    if validate is not None:
        validate(params, env_config)