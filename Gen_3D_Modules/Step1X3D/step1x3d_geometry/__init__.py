import sys
import os
import logging

logging.basicConfig(level=logging.WARNING)

current_dir = os.path.dirname(__file__)
step1x3d_parent = os.path.abspath(os.path.join(current_dir, ".."))
if step1x3d_parent not in sys.path:
    sys.path.insert(0, step1x3d_parent)

import importlib

__modules__ = {}

def register(name):
    def decorator(cls):
        if name in __modules__:
            import logging
            logging.getLogger(__name__).warning(f"Module {name} already registered, skipping.")
        else:
            __modules__[name] = cls
        return cls
    return decorator

def find(name):
    if name in __modules__:
        return __modules__[name]
    else:
        try:
            module_string = ".".join(name.split(".")[:-1])
            cls_name = name.split(".")[-1]
            module = importlib.import_module(module_string, package=None)
            return getattr(module, cls_name)
        except Exception as e:
            raise ValueError(f"Module {name} not found!")


###  grammar sugar for logging utilities  ###
import logging

logger = logging.getLogger("pytorch_lightning")

from pytorch_lightning.utilities.rank_zero import (
    rank_zero_debug,
    rank_zero_info,
    rank_zero_only,
)

debug = rank_zero_debug
info = rank_zero_info


@rank_zero_only
def warn(*args, **kwargs):
    logger.warn(*args, **kwargs)


from . import data, models, systems
