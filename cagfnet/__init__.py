"""CAGF-Net: Cross-Attention Gated Fusion of CT imaging and audiometric signals
for cholesteatoma detection, with a reproducible hybrid simulation benchmark."""

__version__ = "1.0.0"

# On Windows, torch must be imported before pandas/scipy load their own OpenMP/MKL DLLs,
# otherwise c10.dll can fail to initialise (WinError 1114).
import torch  # noqa: E402,F401
