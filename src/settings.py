from pathlib import Path

SEED = 42

ROOT = Path(__file__).parent.parent

CONFIG_DIR = ROOT / "configs"
ENV_CFG_PATH = CONFIG_DIR / "env" / "base_env.yaml" # TODO: Move ENV_CONFIG_DIR determination somewhere more dynamic

DATA_PATH = ROOT / "data" / "processed" / "historical_orders.pkl"

