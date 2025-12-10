from abc import ABC, abstractmethod
import math
import numpy as np
from env.components.components import validate_env_component
from env.components.rewards.lost_sales_assignments import LOST_SALES_ASSIGNMENT_REGISTRY, validate_lost_sales_assignment

class BaseReward(ABC):
    def __init__(self, env_config, **kwargs):

        # Set general attributes
        general_fields = env_config["general"]
        self.n_warehouses = general_fields["n_warehouses"]
        self.n_regions = general_fields["n_regions"]
        self.n_skus = general_fields["n_skus"]

        # Set reward fields
        reward_fields = env_config["components"]["reward"]["reward"]
        self.scope = reward_fields["scope"]
        # TODO: Implement normalize and scale
        self.normalize = reward_fields["normalize"]
        self.scale = reward_fields["scale"]

    @abstractmethod
    def compute(self, inventory, shipped_skus, shipment_counts, lost_sales_region, agents):
        raise NotImplementedError

class InventoryCostReward(BaseReward):
    def __init__(self, env_config, **kwargs):

        # Initialize base class
        super().__init__(env_config)

        # Set cost fields
        cost_fields = env_config["costs"]
        self.holding_costs = np.asarray(cost_fields["holding_costs"])
        self.lost_sales_costs = np.asarray(cost_fields["lost_sales_costs"])
        self.shipment_costs = np.asarray(cost_fields["shipment_costs"])

        # Set inventory cost weights
        reward_fields = env_config["components"]["reward"]["reward"]
        self.cost_weights = reward_fields["cost_weights"]
        for weight, value in self.cost_weights.items():
            setattr(self, weight, value)

        # Set lost sales assignment method
        lost_sales_fields = reward_fields["lost_sales_assignment"]
        self.lost_sales_assignment = lost_sales_fields["model"]
        params = {param: value for param, value in lost_sales_fields.items() if param != "model"}
        self.lost_sales_params = params

    def compute(self, inventory, shipped_skus, shipment_counts, lost_sales_region, agents):

        # Check if reward is shared
        if self.scope == "team":

            # Compute total shipping costs
            shipping_costs = (shipment_counts * self.shipment_costs).sum()

            # Compute total holding costs
            holding_costs = (inventory * self.holding_costs[:, None]).sum()

            # Compute total lost sales costs
            lost_sales_costs = (lost_sales_region * self.lost_sales_costs).sum()

            # Compute total costs by weighting each cost component with its corresponding weight
            total_costs = (
                    self.w_shipment_costs * shipping_costs
                    + self.w_holding_costs * holding_costs
                    + self.w_lost_sales_costs * lost_sales_costs
            )

            # TODO: Check if negative reward is what I need
            # Set total reward as negative total costs
            total_reward = -total_costs.astype(float)

            # Return a dict that assigns each agent the same shared total reward
            return {agent: total_reward for agent in agents}

        # Check if reward is per-agent
        elif self.scope == "agent":

            # Compute shipping costs per warehouse
            shipping_costs = (shipment_counts * self.shipment_costs).sum(axis=1)

            # TODO: if holding costs per sku, remove .sum(axis=1) and validate holding_costs to be of size (n_skus,)
            # Compute holding costs per warehouse
            holding_costs = (inventory * self.holding_costs[:, None]).sum(axis=1)

            # Check for lost sales assignment type and assign correspondingly
            assign_lost_sales = LOST_SALES_ASSIGNMENT_REGISTRY[self.lost_sales_assignment]["assignment"]
            lost_sales_costs = assign_lost_sales(self, shipped_skus, lost_sales_region, **self.lost_sales_params)

            # Compute total costs per agent
            total_costs = (
                    self.w_shipment_costs * shipping_costs
                    + self.w_holding_costs * holding_costs
                    + self.w_lost_sales_costs * lost_sales_costs
            )

            # TODO: Check if negative reward is what I need
            # Set total reward per agent as negative costs
            total_reward = -total_costs.astype(float)

            # Return a dict that assigns each agent its own total reward
            return {agent: total_reward[idx] for idx, agent in enumerate(agents)}

def _validate_rewards(env_config):
    """Validates the reward section in the env_config."""

    # Set additional required fields and values specific to the reward section
    required_rew_fields = ["scope", "normalize", "scale", "lost_sales_assignment"]

    # Validate the reward component based on its registry and config (including reward-specific extra fields)
    validate_env_component(
        component_group="reward",
        component_name="reward",
        registry=REWARD_REGISTRY,
        env_config=env_config,
        extra_params=required_rew_fields
    )

    rew_config = env_config["components"]["reward"]["reward"]

    # Validate scope type
    required_scopes = ["agent", "team"]
    scope = rew_config["scope"]
    if scope not in required_scopes:
        raise ValueError(f"Reward field 'scope' must be in {required_scopes}, got {scope!r}")

    # Check normalize type
    normalize = rew_config["normalize"]
    if not isinstance(normalize, bool):
        raise ValueError(f"Reward field 'normalize' must be type 'bool', got {type(normalize)}.")

    # Check scale type
    scale = rew_config["scale"]
    if not isinstance(scale, (int, float)) or scale <= 0:
        raise ValueError(f"Reward field 'scale' must be a positive number, got {scale}.")

    # Check the lost_sales_assignment subsection
    validate_lost_sales_assignment(scope, env_config)

def _validate_inventory_costs_reward(params, env_config):
    """Validates parameters of the InventoryCost reward for type and shape."""

    # Set required and provided cost weights
    required_cost_weights = [f"w_{cost}" for cost in env_config["costs"]]
    cost_weights = params["cost_weights"]

    # Check existence of required cost weights
    missing = [cost_weight for cost_weight in required_cost_weights if cost_weight not in cost_weights]
    if missing:
        raise ValueError(
            f"Missing required parameter(s) for reward model 'InventoryCost': {missing}"
        )

    # Check if unallowed cost weights are present
    unknown = [param for param in cost_weights.keys() if param not in required_cost_weights]
    if unknown:
        raise ValueError(
            f"Reward model 'InventoryCost' has unknown parameter(s): {unknown}. "
            f"Allowed parameter(s): {required_cost_weights}."
        )

    # Check cost weights for type
    for cost_weight, value in cost_weights.items():
        if not isinstance(value, (int, float)) or value < 0 or value > 1:
            raise ValueError(f"Reward field {cost_weight} must be a positive number in range [0, 1], got {value}.")

    # Check if cost weights sum to one
    if not math.isclose(sum(cost_weights.values()), 1.0, abs_tol=0.0001):
        raise ValueError(f"Values for reward fields [{cost_weights}] must sum to 1 (±0.0001), got sum {sum(cost_weights.values())}")


# Registry for reward models
REWARD_REGISTRY = {
    "InventoryCosts": {
        "cls": InventoryCostReward,
        "params": ["cost_weights"],
        "validate": _validate_inventory_costs_reward,
        "desc": "Computes rewards from an inventory costs perspective (i.e., holding, stockout, ...) "
                "and weights the different costs according to ."
    }
}

def build_reward_function(env_config):
    _validate_rewards(env_config)
    from env.components.components import build_env_component
    return build_env_component(
        component_group="reward",
        component="reward",
        registry=REWARD_REGISTRY,
        env_config=env_config
    )











