import numpy as np
from scipy.special import softmax
from env.components.components import validate_registry_block

def assign_cheapest(self, shipped_skus, lost_sales_region):
    """Assigns lost sales to the closest warehouse of each region."""

    # Sort warehouses by shipment cost per region
    cheapest_warehouse_per_region = np.argmin(self.shipment_costs, axis=0)

    # Compute total lost sales costs per region
    lost_sales_costs_r = (lost_sales_region * self.lost_sales_costs).sum(axis=1)

    # Initialize array for lost sales costs per warehouse
    lost_sales_costs_w = np.zeros(shape=self.n_warehouses, dtype=np.float64)

    # For each region, its lost sales are assigned in full to the closest warehouse for that region
    for region in range(self.n_regions):
        cheapest_warehouse = cheapest_warehouse_per_region[region]
        lost_sales_costs_w[cheapest_warehouse] += lost_sales_costs_r[region]

    return lost_sales_costs_w

def assign_shipment_weighted(self, shipped_skus, lost_sales_region):
    """Assigns lost sales relative to the number of units that each warehouse shipped to each region."""

    # TODO: Replace shipped_skus with shipment_counts if assignment should be done on an order-level basis

    # Determine the sum of all SKU units that each warehouse shipped to each region
    shipped_units_w = shipped_skus.sum(axis=2)

    # Determine the sum of all SKU units that each region received from all warehouses
    region_totals = shipped_units_w.sum(axis=0)

    # Initialize weight array
    weights = np.zeros_like(shipped_units_w, dtype=np.float64)  # (w,r)

    # Set small epsilon to safeguard division by zero
    eps = 1e-8

    # For each region, its lost sales are assigned based on the number of units that each warehouse
    # shipped to that region
    for region in range(self.n_regions):

        # Check if the current region did receive any shipments
        if region_totals[region] > 0:

            # Assign weights according to the amount of units shipped
            weights[:, region] = shipped_units_w[:, region] / region_totals[region]

        # Revert to fallback method if the current region did not receive any shipments (here: cheapest warehouse)
        else:
            cheapest_warehouse = np.argmin(self.shipment_costs, axis=0)[region]
            weights[:, region] = 0
            weights[cheapest_warehouse, region] = 1

    # Apply the weights to the total lost sales costs per region to assign shares of the costs to warehouses
    lost_sales_costs_r = (lost_sales_region * self.lost_sales_costs).sum(axis=1)
    lost_sales_costs_w = (weights * lost_sales_costs_r[None, :]).sum(axis=1)

    return lost_sales_costs_w

def assign_cost_weighted(self, shipped_skus, lost_sales_region, alpha):
    """Assigns lost sales relative to the shipment costs of each warehouse to each region."""

    # TODO: Replace shipped_skus with shipment_counts if assignment should be done on an order-level basis

    # Compute weight array as a softmax over shipment costs
    weights = softmax(-alpha * self.shipment_costs, axis=0)

    # Apply the weights to the total lost sales costs per region to assign shares of the costs to warehouses
    lost_sales_costs_r = (lost_sales_region * self.lost_sales_costs).sum(axis=1)
    lost_sales_costs_w = (weights * lost_sales_costs_r[None, :]).sum(axis=1)

    return lost_sales_costs_w

def validate_lost_sales_assignment(scope, env_config):
    """Validates the lost sales assignment section in the env_config."""

    ls_config = env_config["components"]["reward"]["reward"].get("lost_sales_assignment")

    # No lost sales assignment if reward is shared
    if scope == "team":
        valid = True
        # Check if ls_config is omitted/set to 'Null' or if it is not a dict
        if ls_config is None or not isinstance(ls_config, dict):
            valid = False
        # Check if ls_config is empty except for model=None
        else:
            if ls_config.get("model") is not None:
                valid = False
            if [key for key in ls_config.keys() if key != "model"]:
                valid = False

        # Raise if ls_config is not valid
        if not valid:
            raise ValueError(
                f"For reward.reward.scope 'team', 'lost_sales_assignment.model' must be set to 'Null' "
                f"with no other parameters present, got {ls_config}."
            )

    # Lost sales assignment if reward is per-agent
    elif scope == "agent":
        # Raise if ls_config is empty or the model is not specified
        if ls_config in (None, {}) or ls_config.get("model") is None:
            raise ValueError(
                "For reward.reward.scope 'agent', lost_sales_assignment model must be specified, "
                f"got {ls_config}."
            )

        # Validate parameters for the specified lost sales method for type and shape
        context = "reward.lost_sales_assignment"
        validate_registry_block(LOST_SALES_ASSIGNMENT_REGISTRY, ls_config, context, env_config)

def _validate_cost_weighted(params, env_config):
    """Validates parameters of the CostWeighted lost sales assignment model for type and shape."""

    alpha = np.asarray(params["alpha"])

    # Check if alpha is a positive number
    if not np.issubdtype(alpha.dtype, np.number) or np.any(alpha < 0):
        raise ValueError(f"shipment_weighted.alpha must be a positive number, got {alpha}.")

# Registry for lost sales assignment models
LOST_SALES_ASSIGNMENT_REGISTRY = {
    "Cheapest": {
        "assignment": assign_cheapest,
        "params": [],
        "validate": None,
        "desc": "Assigns lost sales of a region only to the closest warehouse."
    },
    "ShipmentWeighted": {
        "assignment": assign_shipment_weighted,
        "params": [],
        "validate": None,
        "desc": "Assigns lost sales of a region relative to the number of SKU units shipped to that region."
    },
    "CostWeighted": {
        "assignment": assign_cost_weighted,
        "params": ["alpha"],
        "validate": _validate_cost_weighted,
        "desc": "Assigns lost sales of a region relative to shipping costs to that region."
                "Alpha >= 0 controls how strongly lost sales are assigned to cheaper warehouses."
                "If alpha 0, lost sales are assigned equally among all warehouses. The larger alpha, the more "
                "are lost sales assigned to cheap warehouses only."
    },
}