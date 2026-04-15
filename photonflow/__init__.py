"""PhotonFlow: strictly photon-native CFM for MZI silicon-photonic hardware."""

from photonflow.activation import SaturableAbsorber
from photonflow.normalization import DivisivePowerNorm
from photonflow.noise import PhotonicNoise
from photonflow.layers import MonarchLinear, PPLNSigmoid
from photonflow.time_embed import WavelengthCodedTime
from photonflow.model import MonarchLayer, PhotonFlowBlock, PhotonFlowModel
from photonflow.sampler import OpticalSampler
from photonflow.train import CFMLoss, Trainer, EMA

__all__ = [
    # Photonic primitives
    "SaturableAbsorber",
    "PPLNSigmoid",
    "DivisivePowerNorm",
    "PhotonicNoise",
    "MonarchLayer",
    "MonarchLinear",
    "WavelengthCodedTime",
    # Network building blocks
    "PhotonFlowBlock",
    "PhotonFlowModel",
    # Photon-native sampler (replaces euler_sample)
    "OpticalSampler",
    # Training utilities (digital off-chip)
    "CFMLoss",
    "Trainer",
    "EMA",
]
