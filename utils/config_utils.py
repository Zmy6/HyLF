#!/usr/bin/env python

from omegaconf import DictConfig


def apply_to_matching_config_keys(config, key_fragment, callback):
    for key in list(config.keys()):
        value = config[key]
        if isinstance(value, DictConfig):
            apply_to_matching_config_keys(value, key_fragment, callback)
        if key_fragment in key:
            callback(config, key)
