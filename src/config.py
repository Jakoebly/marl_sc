import yaml
import numpy as np

def load_config(config_path):
    """Loads and validates a .yaml config file"""

    # Open .yaml file
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Return the validated config
    return _validate_config(config)

def _validate_config(config):
    """Validates configs based on type"""

    # Get the config type (e.g., env, algo, ...)
    config_type = config.get("type")

    # Validate configs by using type-specific helper functions
    # TODO: add other configs
    if config_type is None:
        raise ValueError("Missing required 'type' field in config.")
    elif config_type == "env":
        return _validate_env_config(config)
    else:
        raise ValueError(f"Unknown config type: {config_type}")

def _validate_env_config(env_config):
    """Validates core fields (i.e., general and costs) of the environment config for presence, type, and shape.
    Individual components are validated later at runtime."""

    # Set required fields and values
    required_scalar_fields = ["n_warehouses", "n_regions", "n_skus","episode_length"]
    required_cost_fields = ["holding_costs", "lost_sales_costs", "shipment_costs"]

    # Get general and cost fields and check for presence
    general_fields = env_config.get("general")
    cost_fields = env_config.get("costs")

    if general_fields is None:
        raise ValueError("Missing required 'general' section in env_config.")
    if cost_fields is None:
        raise ValueError("Missing required 'costs' section in env_config.")


    # Check required scalar fields for presence and type
    for field in required_scalar_fields:
        if field not in general_fields:
            raise ValueError(f"Missing required field {field} in env_config.")

        value = general_fields[field]
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"Field {field} in env_config must be a positive integer, got {value}.")

    n_warehouses = general_fields["n_warehouses"]
    n_regions = general_fields["n_regions"]
    n_skus = general_fields["n_skus"]

    # Check initial inventory parameters for type
    if "init_inv_mean" in general_fields:
        if not isinstance(general_fields["init_inv_mean"], (int,float)):
            raise ValueError(
                f"init_inv_mean in env_config must be a number, "
                f"got {general_fields['init_inv_mean']}."
                )
    if "init_inv_std" in general_fields:
        if not isinstance(general_fields["init_inv_std"], (int,float)) or general_fields["init_inv_std"] < 0:
            raise ValueError(
                f"init_inv_std in env_config must be a non-negative number, "
                f"got {general_fields['init_inv_std']}."
                )

    # Check cost fields for presence and type
    for cost in required_cost_fields:

        if cost not in cost_fields:
            raise ValueError(f"Missing required field {cost} in env_config.")

        cost_array = np.asarray(cost_fields[cost])

        if not np.issubdtype(cost_array.dtype, np.number) or np.any(cost_array < 0):
            raise ValueError(f"{cost} in env_config must be positive number(s), got {cost_array}")

        if cost == "holding_costs" and cost_array.shape not in [(), (n_warehouses,)]:
            raise ValueError(
                f"{cost} must have shape scalar () or (n_warehouses,)={(n_warehouses, )}, got {cost_array.shape}."
            )

        if cost == "shipment_costs" and cost_array.shape != (n_warehouses, n_regions):
            raise ValueError(
                f"{cost} must have shape (n_warehouses, n_regions)={(n_warehouses, n_regions)}, "
                f" got {cost_array.shape}."
            )

        if cost == "lost_sales_costs" and cost_array.shape not in [(), (n_skus,)]:
            raise ValueError(
                f"{cost} must have shape scalar () or (n_warehouses,)={(n_skus, )}, got {cost_array.shape}."
            )

    return env_config


