"""Extended-CAM and the PASCAL VOC 2007 pointing game."""

from .api import extended_cam, grad_cam, grad_cam_pp
from .timm_adapter import TimmClassifier

__all__ = ["TimmClassifier", "extended_cam", "grad_cam", "grad_cam_pp"]
