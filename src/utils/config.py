"""
IntelliCrash — Configuration Loader Utility.

Central config loading from YAML with dot-access support.
"""

import os
import yaml
from pathlib import Path


# Project root is the parent of the src/ directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(config_path: str = None) -> dict:
    """Load YAML configuration file.

    Args:
        config_path: Optional path to config file. Defaults to configs/config.yaml.

    Returns:
        dict: Configuration dictionary.
    """
    path = Path(config_path) if config_path else CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # Resolve relative paths against project root
    if "paths" in cfg:
        for key, val in cfg["paths"].items():
            if isinstance(val, str) and not os.path.isabs(val):
                cfg["paths"][key] = str(PROJECT_ROOT / val)

    return cfg


def get_config() -> dict:
    """Singleton-style config loader — loads once and caches."""
    if not hasattr(get_config, "_cache"):
        get_config._cache = load_config()
    return get_config._cache
