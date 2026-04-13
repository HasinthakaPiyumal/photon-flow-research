"""PhotonFlow: Butterfly-Structured Flow Matching for Optical Neural Network Hardware."""

from photonflow.activation import SaturableAbsorber
from photonflow.normalization import DivisivePowerNorm
from photonflow.noise import PhotonicNoise
from photonflow.model import MonarchLayer, PhotonFlowBlock, PhotonFlowModel
from photonflow.train import CFMLoss, euler_sample, Trainer, EMA

__all__ = [
    "SaturableAbsorber",
    "DivisivePowerNorm",
    "PhotonicNoise",
    "MonarchLayer",
    "PhotonFlowBlock",
    "PhotonFlowModel",
    "CFMLoss",
    "euler_sample",
    "Trainer",
    "EMA",
]
