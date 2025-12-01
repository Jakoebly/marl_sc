from src.utils import set_seed
from src.config import load_config
from src.settings import SEED, ENV_CFG_PATH

def main():

    # Set seeds
    set_seed(SEED)
    env_cfg = load_config(ENV_CFG_PATH)
    print("Success")


if __name__ == "__main__":
    main()
