import importlib
import logging
import os
from copy import deepcopy
from os import path as osp

from basicsr.utils.registry import ARCH_REGISTRY

__all__ = ['build_network']

_ARCHS_IMPORTED = False


def _ensure_arch_modules_imported():
    """Lazy import arch modules for registry.

    In upstream BasicSR, importing `basicsr.archs` scans and imports all `*_arch.py`
    modules eagerly to populate the registry. That adds import overhead and may
    pull in extra dependencies in inference-only scenarios.
    Here we make it lazy: only scan/import when `build_network` is actually called.
    """
    global _ARCHS_IMPORTED
    if _ARCHS_IMPORTED:
        return
    arch_folder = osp.dirname(osp.abspath(__file__))
    arch_filenames = []
    for name in os.listdir(arch_folder):
        if name.endswith("_arch.py"):
            arch_filenames.append(osp.splitext(name)[0])
    for file_name in arch_filenames:
        importlib.import_module(f'basicsr.archs.{file_name}')
    _ARCHS_IMPORTED = True


def build_network(opt):
    _ensure_arch_modules_imported()
    opt = deepcopy(opt)
    network_type = opt.pop('type')
    net = ARCH_REGISTRY.get(network_type)(**opt)
    logging.getLogger('basicsr').info(f'Network [{net.__class__.__name__}] is created.')
    return net
