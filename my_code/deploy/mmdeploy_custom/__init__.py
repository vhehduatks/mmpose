"""
MMDeploy Custom Model Registration for EgoPose

This module registers custom EgoPose models with MMDeploy for deployment.
It provides model rewriters and symbolic functions for ONNX export.

Usage:
    # Import this module before running MMDeploy conversion
    import my_code.deploy.mmdeploy_custom
"""

from .model_rewriters import *
from .deploy_configs import *

__all__ = ['EgoPoseWrapper', 'get_deploy_config']
