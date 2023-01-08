import inspect
import logging
import os
import time
from datetime import datetime
from functools import reduce
from operator import getitem
from pathlib import Path

from helpers.fit.utils import read_json, write_json


class ConfigParser:
    def __init__(self, args, options='', timestamp=True):
        # parse default and custom cli options
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        args = args.parse_args()

        if args.device:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        msg_no_cfg = "Configuration file need to be specified."
        assert args.config is not None, msg_no_cfg
        self.cfg_fname = Path(args.config)
        config = read_json(self.cfg_fname)

        # load config file and apply custom cli options
        self._config = _update_config(config, options, args)

    def initialize(self, name, module,  *args, index=None, **kwargs):
        """
        finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding keyword args given as 'args'.
        """
        if index is None:
            module_name = self[name]['type']
            module_args = dict(self[name]['args'])
            assert all(k not in module_args for k in kwargs), 'Overwriting kwargs given in config file is not allowed'
            module_args.update(kwargs)
        else:
            module_name = self[name][index]['type']
            module_args = dict(self[name][index]['args'])

        # if parameter not in config subdict, then check if it's in global config.
        signature = inspect.signature(getattr(module, module_name).__init__)
        print(module_name)
        for param in signature.parameters.keys():
            if param not in module_args and param in self.config:
                module_args[param] = self[param]

        return getattr(module, module_name)(*args, **module_args)

    def __getitem__(self, name):
        return self.config[name]

    # setting read-only attributes
    @property
    def config(self):
        return self._config

# helper functions used to update config dict with custom cli options
def _update_config(config, options, args):
    for opt in options:
        value = getattr(args, _get_opt_name(opt.flags))
        if value is not None:
            _set_by_path(config, opt.target, value)
    return config


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')


def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
