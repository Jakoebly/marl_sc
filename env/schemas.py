from dataclasses import dataclass
import numpy as np

@dataclass
class OrdersBatch:
    orders: np.ndarray # [O_t, K] with O_t = number of orders in time t
    order_regions: np.ndarray # [O_t]

@dataclass
class AllocationResults:
    shipped_skus: np.ndarray
    shipment_counts: np.ndarray
    lost_sales_region: np.ndarray
