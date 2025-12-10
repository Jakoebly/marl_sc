from abc import ABC, abstractmethod
import numpy as np

from env.schemas import AllocationResults
from env.components.components import validate_env_component

class BaseDemandAllocator(ABC):
    def __init__(self, env_config):

        # Set general attributes
        general_fields = env_config["general"]
        cost_fields = env_config["costs"]
        self.n_warehouses = general_fields["n_warehouses"]
        self.n_regions = general_fields["n_regions"]
        self.n_skus = general_fields["n_skus"]
        self.shipment_costs = np.asarray(cost_fields["shipment_costs"])

    @abstractmethod
    def allocate(self, inventory, demand):
        raise NotImplementedError

class GreedyAllocator(BaseDemandAllocator):
    def __init__(self, env_config, max_splits, **kwargs):

        # Initialize base class
        super().__init__(env_config)

        # Set maximum number of allowed splits
        self.max_splits = self.n_warehouses - 1 if max_splits == "default" else max_splits

        # Sort warehouses by shipment_costs per region
        self.warehouses_sorted_per_region = np.argsort(self.shipment_costs, axis=0)

    def allocate(self, inventory, demand):
        """Allocates orders greedily by shipment cost with a cap on warehouse splitting per order."""

        # Initialize return arrays
        shipment_counts = np.zeros(shape=(self.n_warehouses, self.n_regions), dtype=np.int64)
        shipped_skus = np.zeros(shape=(self.n_warehouses, self.n_regions, self.n_skus), dtype=np.int64)
        lost_sales_region = np.zeros(shape=(self.n_regions, self.n_skus), dtype=np.int64)

        # Extract orders and corresponding demand regions
        orders = demand.orders
        order_regions = demand.order_regions

        # Copy inventory so that it can be altered in the loop
        inventory = inventory.copy()

        # Define max number of warehouses allowed to fulfill a single order
        max_warehouses_per_order = self.max_splits + 1

        for region in range(self.n_regions):

            # Fetch all orders from the current region
            orders_per_region = orders[order_regions == region]

            # If a region has no orders, move to the next region
            if orders_per_region.size == 0:
                continue

            # Sort warehouses by shipment costs to the current region
            warehouses_sorted = self.warehouses_sorted_per_region[:, region]

            # Loop over all orders associated with the current region
            for order in orders_per_region:

                # Track remaining SKU demand and shipping warehouses for the current order
                remaining = order.copy()
                used_warehouses = 0

                # Loop over warehouses in order of increasing costs
                for warehouse in warehouses_sorted:

                    # If the maximum number of splits is reached, stop assigning demand
                    if used_warehouses >= max_warehouses_per_order:
                        break

                    # Determine the quantity per SKU that the current warehouse can fulfill for the current order
                    fulfillment_qty = np.minimum(remaining, inventory[warehouse])

                    # Only continue if the current warehouse can fulfill something from the current order
                    if np.any(fulfillment_qty > 0):

                        # The current warehouse ships what it can contribute
                        shipped_skus[warehouse, region] += fulfillment_qty
                        shipment_counts[warehouse, region] += 1

                        # Remaining SKU demands and inventory is updated
                        remaining -= fulfillment_qty
                        inventory[warehouse] -= fulfillment_qty
                        used_warehouses += 1

                        # If no SKU demand is left, move to the next order
                        if not np.any(remaining > 0):
                            break

                # Check if there are remaining SKU demands and add them as lost sales for current region
                if np.any(remaining > 0):
                    remaining_mask = remaining > 0
                    lost_sales_region[region, remaining_mask] += remaining[remaining_mask > 0]

        # Return results as AllocationResults
        return AllocationResults(
            shipped_skus=shipped_skus,
            shipment_counts=shipment_counts,
            lost_sales_region=lost_sales_region
        )

def _validate_demand_allocator(env_config):
    """Validates the demand allocator section in the env_config."""

    # Validate the demand allocator component based on its registry and config (no allocator-specific extra fields)
    validate_env_component(
        component_group="processes",
        component_name="demand_allocator",
        registry=DEMAND_ALLOCATOR_REGISTRY,
        env_config=env_config,
        extra_params=None
    )

def _validate_greedy_allocator(params, env_config):
    """Validates parameters of the Greedy allocator for type and shape."""

    # Set values
    n_warehouses = env_config["general"]["n_warehouses"]
    max_splits = params["max_splits"]

    # Return if max_splits has default value
    if max_splits == "default":
        return

    # Check max_splits for type and shape
    max_splits = np.asarray(max_splits)
    if not np.issubdtype(max_splits.dtype, np.integer) or np.any(max_splits < 0) or np.any(max_splits >= n_warehouses):
        raise ValueError(f"demand_allocator.max_splits must be a non-negative integer in range "
                         f"[0, n_warehouses-1]=[0, {n_warehouses-1}], got {max_splits}.")
    if max_splits.shape != ():
        raise ValueError(f"demand_allocator.max_splits must have shape scalar (), got {max_splits.shape}.")


# Registry for allocator models
DEMAND_ALLOCATOR_REGISTRY = {
    "Greedy": {
        "cls": GreedyAllocator,
        "params": ["max_splits"],
        "validate": _validate_greedy_allocator,
        "desc": "Allocates orders greedily by shipment cost with optional max_splits limit. "
                "If max_splits is set to 'default', uses max_splits = n_warehouses - 1 (no splitting cap)"
    }
}

def build_demand_allocator(env_config):
    """Validates and builds the demand allocator specified in the env_config"""

    # Validate demand_allocator section in env_config
    _validate_demand_allocator(env_config)

    # Build demand allocator
    from env.components.components import build_env_component
    return build_env_component(
        component_group="processes",
        component="demand_allocator",
        registry=DEMAND_ALLOCATOR_REGISTRY,
        env_config=env_config
    )



