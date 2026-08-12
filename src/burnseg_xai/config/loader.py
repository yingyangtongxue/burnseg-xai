import os

import yaml

from .schema import ProjectConfig

# Path fields that may be overridden from the environment (or a local .env
# file) instead of being hardcoded in a tracked YAML config. This keeps
# machine-specific storage layouts out of version control while the repo
# itself ships only relative, portable defaults.
_ENV_PATH_OVERRIDES = {
    "dataset_root": "BURNSEG_DATASET_ROOT",
    "output_root": "BURNSEG_OUTPUT_ROOT",
    "kml_dir": "BURNSEG_KML_DIR",
    "checkpoint_dir": "BURNSEG_CHECKPOINT_DIR",
    "mlflow_tracking_uri": "BURNSEG_MLFLOW_TRACKING_URI",
}


def _load_dotenv():
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)


def load_config(path: str = None) -> ProjectConfig:
    _load_dotenv()
    cfg = ProjectConfig()

    if path is not None:
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Handle both flat keys and nested sections (training.batch_size etc.)
        for key, value in data.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if hasattr(cfg, sub_key):
                        setattr(cfg, sub_key, sub_value)
            elif hasattr(cfg, key):
                setattr(cfg, key, value)

    for field_name, env_var in _ENV_PATH_OVERRIDES.items():
        env_value = os.environ.get(env_var)
        if env_value:
            setattr(cfg, field_name, env_value)

    return cfg
