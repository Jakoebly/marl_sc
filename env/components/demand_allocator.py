from abc import ABC, abstractmethod
import numpy as np

from env.schemas import AllocationResults


# TODO: add same comments as in demand_sampler.py

class BaseDemandAllocator(ABC):
    def __init__(self, env_config):

        # Set general attributes
        self.n_warehouses = env_config["n_warehouses"]
        # TODO: implement n_regions in the env_config including validation
        self.n_regions = self.n_warehouses
        self.n_skus = env_config["n_skus"]
        self.shipment_costs = np.asarray(env_config["shipment_costs"])

    @abstractmethod
    def allocate(self, inventory, demand):
        raise NotImplementedError

class GreedyAllocator(BaseDemandAllocator):
    def __init__(self, env_config, max_splits=None):

        # Initialize base class
        super().__init__(env_config)

        # Set maximum number of allowed splits
        if max_splits is None:
            self.max_splits = self.n_warehouses - 1
        else:
            self.max_splits = max_splits

    def allocate(self, inventory, demand):
        """Allocates orders greedily by shipment cost with a cap on warehouse splitting per order."""

        # Initialize return arrays
        shipment_counts = np.zeros(shape=(self.n_warehouses, self.n_regions), dtype=np.int64)
        shipped_skus = np.zeros(shape=(self.n_warehouses, self.n_regions, self.n_skus), dtype=np.int64)
        lost_sales = np.zeros(shape=(self.n_warehouses, self.n_skus), dtype=np.int64)

        # Extract orders and corresponding demand regions
        orders = demand.orders
        order_regions = demand.order_regions

        # Copy inventory so that it can be altered in the loop
        inventory = inventory.copy()

        # Define max number of warehouses allowed to fulfill a single order
        max_warehouses_per_order = self.max_splits + 1

        for region in range(self.n_regions):

            # Fetch all orders from the given region
            orders_per_region = orders[order_regions == region]

            # If a region has no orders, move to the next region
            if orders_per_region.size == 0:
                continue

            # Sort warehouses by shipment costs to the given region
            warehouses_sorted = np.argsort(self.shipment_costs[:, region], axis=0)

            # Loop over all orders associated with the given region and allocate demands to warehouses
            for order in orders_per_region:
                order = order.copy()

                # Track how many warehouses already ship for the given order
                used_warehouses = 0

                for warehouse in warehouses_sorted:

                    # If the maximum number of splits is reached, stop splitting
                    if used_warehouses >= max_warehouses_per_order:
                        break

                    # Determine the quantity per SKU that the given warehouse can fulfill for the given order
                    fulfillment_qty = np.minimum(order, inventory[warehouse])

                    # The given warehouse cannot fulfill anything from this order, move to the next cheapest warehouse
                    if not np.any(fulfillment_qty > 0):
                        continue

                    # The given warehouse ships what it can contribute
                    shipped_skus[warehouse, region] += fulfillment_qty
                    shipment_counts[warehouse, region] += 1

                    # Remaining SKU demands and inventory is updated
                    order -= fulfillment_qty
                    inventory[warehouse] -= fulfillment_qty
                    used_warehouses += 1

                    # If no SKU demand is left, move to the next order
                    if not np.any(order > 0):
                        break

                # Check if there are remaining SKU demands and add them as lost sales for the cheapest warehouse
                remaining = order > 0
                if np.any(remaining):
                    cheapest_warehouse = warehouses_sorted[0]
                    lost_sales[cheapest_warehouse, remaining] += order[remaining]

        return AllocationResults(
            shipped_skus=shipped_skus,
            shipment_counts=shipment_counts,
            lost_sales=lost_sales
        )

def _validate_greedy_allocator(params, env_config):
    """Validates required parameters of the GreedyMaxSplits allocator for type and shape."""

    # If the optional max_splits parameter is not given, return
    if params == {}:
        return

    # Set values
    n_warehouses = env_config["n_warehouses"]
    max_splits = np.asarray(params["max_splits"])

    # Check max_splits for type and shape
    if not np.issubdtype(max_splits.dtype, np.integer) or np.any(max_splits < 0) or np.any(max_splits >= n_warehouses):
        raise ValueError(f"demand_allocator.max_splits must be a non-negative integer in range "
                         f"[0, n_warehouses-1]=[0, {n_warehouses-1}], got {max_splits}.")
    if max_splits.shape != ():
        raise ValueError(f"demand_allocator.max_splits must have shape scalar (), got {max_splits.shape}.")

DEMAND_ALLOCATOR_REGISTRY = {
    "Greedy": {
        "cls": GreedyAllocator,
        "required_params": [],
        "optional_params": ["max_splits"],
        "validate": _validate_greedy_allocator,
        "desc": "Allocates orders greedily by shipment cost with optional max_splits limit."
    }
}

def build_demand_allocator(env_config):
    from env.components.components import build_env_component
    return build_env_component(component="demand_allocator", registry=DEMAND_ALLOCATOR_REGISTRY, env_config=env_config)



