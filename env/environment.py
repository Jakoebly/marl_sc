from __future__ import annotations

from copy import copy

import numpy as np

from pettingzoo.utils import ParallelEnv
from gymnasium import spaces

from env.components.processes.demand_sampler import build_demand_sampler
from env.components.processes.demand_allocator import build_demand_allocator
# from components.lead_time_sampler import build_lead_time_sampler
from env.components.rewards.rewards import build_reward_function

class InventoryEnv(ParallelEnv):

    metadata = {"render_modes": ["human"], "name": "multi_warehouse_inventory_v0"}

    def __init__(self, env_config):

        # General attributes
        self.n_warehouses = env_config["general"]["n_warehouses"]
        self.n_skus = env_config["general"]["n_skus"]
        self.episode_length = env_config["general"].get("episode_length", 365)

        # TODO: Add lead_time_sampler
        # External components
        self.demand_sampler = build_demand_sampler(env_config)
        self.demand_allocator = build_demand_allocator(env_config)
        # self.lead_time_sampler = build_lead_time_sampler(env_config)
        self.reward_function = build_reward_function(env_config)

        # TODO: specify reward and cost parameters
        # Costs
        # TODO: add to config and validation logic
        self.holding_costs = env_config["costs"].get("holding_costs", 1.0)
        self.lost_sales_costs = env_config["costs"].get("lost_sales_costs", 2.0)
        self.shipment_costs = env_config["costs"].get("shipment_costs", 0.0)

        # TODO: decide whether to keep initial inventory as normal distribution
        # Initial inventory distribution parameters
        self.init_inv_mean = env_config["general"].get("init_inv_mean", 10)
        self.init_inv_std = env_config["general"].get("init_inv_std", 2)

        # Agent information
        self.possible_agents = [f"warehouse_{i}" for i in range(self.n_warehouses)]
        self.agent_ids = list(range(len(self.possible_agents)))
        self.agent_id_mapping = dict(zip(self.possible_agents, self.agent_ids))
        self.observation_spaces = self._build_observation_spaces()
        self.action_spaces = self._build_action_spaces()

        # TODO: adjust when defining more state vars
        # State information
        self.inventory = None
        self.pipeline = None
        self.pipeline_total = None
        self.timestep  = None
        self.terminated = False


    def reset(self, seed=None, options=None):

        self.agents = copy(self.possible_agents)
        self.timestep = 0
        self.terminated = False

        # TODO: Adjust initial inventory method if necessary
        # Set inventory to an initial value based on a specified normal distribution
        self.inventory = np.maximum(
            np.random.normal(
                loc=self.init_inv_mean,
                scale=self.init_inv_std,
                size=(self.n_warehouses, self.n_skus)
            ),
            0
        ).astype(np.int64)

        # Reset pipeline
        self.pipeline = np.zeros((self.n_warehouses, self.n_skus, self.episode_length), dtype=np.int64)
        self.pipeline_total = np.zeros((self.n_warehouses, self.n_skus), dtype=np.int64)

        # Reset demand sampler
        self.demand_sampler.reset(self.episode_length)

        # Get initial observations
        obs = self._get_observations()

        # Dummy infos for Parallel to AEC conversion
        infos = {agent: {} for agent in self.agents}

        return obs, infos

    def step(self, actions):

        """
        # TODO: Adjust decision sequence if necessary.
        Decision sequence per period t:
            1. Each warehouse chooses reorder quantities (actions) based on obs of t-1.
            2. Arrivals from previous orders based on realized lead times.
            3. Demand arrives as a batch of orders (per region / SKU).
            4. LP allocator decides how much each warehouse ships, and lost sales.
            5. Shipments are applied and inventories updated.
        """

        # TODO: Adjust terminal behaviour, evtl. run _terminal_step()  on the bottom when terminated is set and
        # check termination in outside loop
        #if self.terminated:
        #    return self._terminal_step()
#
        ## Apply actions
        #self._apply_actions(actions)
#
        ## Shipments arrive
        #self._apply_arrivals()

        # Sample demand
        order_batch = self.demand_sampler.sample_timestep(self.timestep) # OrderBatch()

        # Allocate and ship demand
        alloc_results = self.demand_allocator.allocate(inventory=self.inventory, demand=order_batch)
        shipped_skus = alloc_results.shipped_skus
        shipped_per_wh = shipped_skus.sum(axis=1)
        shipment_counts = alloc_results.shipment_counts
        lost_sales_region = alloc_results.lost_sales_region

        # Apply shipments
        self.inventory = np.maximum(self.inventory - shipped_per_wh, 0)

        # Compute rewards
        rewards = self.reward_function.compute(
            inventory=self.inventory,
            shipped_skus=shipped_skus,
            shipment_counts=shipment_counts,
            lost_sales_region=lost_sales_region,
            agents=self.agents
        )

        self.timestep += 1
        if self.timestep >= self.episode_length:
            self.terminated = True

        obs = self._get_observations()

        terminations = {agent: self.terminated for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return obs, rewards, terminations, truncations, infos

    def state(self):
        pass

    def _build_observation_spaces(self):
        """
        # TODO: Adjust observation space
        Observation per warehouse:
          - current inventory per SKU (R^K)
          - pipeline (total outstanding per SKU) (R^K)
          - time step (scalar, normalized)
        """

        # TODO: Adjust low and high for actual observation space
        # TODO: Add feature config if applicable
        low = np.concatenate([
            np.zeros(self.n_skus, dtype=np.float32),  # inv >= 0
            np.zeros(self.n_skus, dtype=np.float32),  # pipeline >= 0
            np.array([0], dtype=np.float32)  # normalized time step in [0, 1]
        ])
        high = np.concatenate([
            np.full(self.n_skus, 1e6, dtype=np.float32),  # inv >= 0
            np.full(self.n_skus, 1e6, dtype=np.float32),  # pipeline >= 0
            np.array([1], dtype=np.float32)  # normalized time step in [0, 1]
        ])

        box = spaces.Box(low=low, high=high, dtype=np.float32)

        return {agent: box for agent in self.possible_agents}

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def _build_action_spaces(self):
        """
        # TODO: Adjust observation space if necessary
        Action per warehouse:
          - reorder quantities per SKU (non-negative).
        """

        # TODO: Adjust low and high for actual action space if necessary
        low = np.zeros(self.n_skus, dtype=np.float32)
        high = np.full(self.n_skus, 1e6, dtype=np.float32)

        box = spaces.Box(low=low, high=high, dtype=np.float32)

        return {agent: box for agent in self.possible_agents}

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _get_observations(self):

        # TODO: Add more observations
        timestep_norm = np.array([self.timestep / max(self.episode_length, 1)]).astype(np.float32)

        obs = {}

        for agent_id, agent in enumerate(self.agents):
            inv = self.inventory[agent_id]
            pipe = self.pipeline_total[agent_id]
            obs[agent] = np.concatenate([inv, pipe, timestep_norm]).astype(np.float32)
        return obs

    def _terminal_step(self):
        pass

    def _apply_arrivals(self):
        arrivals = self.pipeline[..., self.timestep]
        self.inventory += arrivals
        self.pipeline[..., self.timestep] = 0
        self.pipeline_total -= arrivals

    def _apply_actions(self, actions):

        # actions: Dict[str, array[K]]

        for agent_id, agent in enumerate(self.agents):
            reorder_qty = np.array(actions[agent], dtype=np.int64) # [K], already assumes non-negativity
            # TODO: adjust lead time sampling
            #  --> there should be one lt dist per SKU, if dists are different per SKU, then handled in lead time model
            lead_times = self.lead_time_sampler.sample(n_skus=self.n_skus) # [K]
            self.pipeline[agent_id, np.arange(self.n_skus), self.timestep + lead_times] += reorder_qty
            self.pipeline_total[agent_id] += reorder_qty

    def render(self):

        print(f"t={self.timestep}")
        for agent_id, agent in enumerate(self.agents):
            print(f"{agent}: inventory={self.inventory[agent_id]}")

    def close(self):
        pass


