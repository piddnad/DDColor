"""BasicSR (vendored)

This repo's inference scripts only need a small subset under `basicsr.archs...`.
Upstream BasicSR's `basicsr/__init__.py` often does `import *` from archs/data/losses/metrics/models/train/utils,
which pulls in many training-only dependencies during inference import.

We keep this `__init__` lightweight to avoid import-time side effects.
Training code should explicitly import the needed submodules.
"""

# flake8: noqa
try:
    from .version import __gitsha__, __version__  # type: ignore
except Exception:
    __gitsha__ = None
    __version__ = None
