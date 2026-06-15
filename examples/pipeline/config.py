"""Config loading: a base yaml merged with CLI dotlist overrides (OmegaConf).
Usage: --config path.yaml model.ssl=vicreg optim.epochs=50"""
import os

from omegaconf import OmegaConf

_DEFAULT = os.path.join(os.path.dirname(__file__), "configs", "base.yaml")


def load_config(path=None, overrides=None):
    cfg = OmegaConf.load(path or _DEFAULT)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg
