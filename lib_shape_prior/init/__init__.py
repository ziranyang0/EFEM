import os
from .config_utils import load_config, get_spec_with_default
from .pre_config import parse_cmd_params, merge_cmd2cfg
from .post_config import post_config
from torch import autograd
import torch
import numpy as np
import random

def get_cfg():
    """
    init priority: command line > init file > default or base init
    """
    project_root = os.getcwd()

    # parse cmd args
    cmd = parse_cmd_params()
    if cmd.enable_anomaly:
        autograd.detect_anomaly()
    # load from init file and default file
    config_fn = os.path.join(project_root, cmd.config_fn)
    cfg = load_config(config_fn, default_path=os.path.join(project_root, "init/default.yaml"))

    # merge cmd to cfg
    cfg = merge_cmd2cfg(cmd, cfg)
    cfg["root"] = project_root

    # startup
    cfg = post_config(cfg, interactive=not cmd.no_interaction)

    return cfg

import argparse
def dev_get_cfg(category = "mugs"):
    assert category in ["mugs", "chairs", "kit4cates"]
    project_root = os.getcwd()
    # cmd = parse_cmd_params()
    cmd = argparse.Namespace(config=f'/home/ziran/se3/EFEM/lib_shape_prior/configs/{category}.yaml', f=True,
                             gpu=None, batch_size=32, 
                             num_workers=-1, debug=False,
                             anomaly=False, debug_logging_flag=False,
                             resume=None, no_interaction=True,
                             enable_anomaly=False,)

    
    config_fn = os.path.join(project_root, f"/home/ziran/se3/EFEM/lib_shape_prior/configs/{category}.yaml")
    cfg = load_config(config_fn, default_path=os.path.join(project_root, "/home/ziran/se3/EFEM/lib_shape_prior/init/default.yaml"))

    # merge cmd to cfg
    cfg = merge_cmd2cfg(cmd, cfg)
    cfg["root"] = project_root

    # startup
    cfg = post_config(cfg, interactive=not cmd.no_interaction)

    return cfg
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
