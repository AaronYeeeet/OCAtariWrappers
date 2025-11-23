from .masked_dqn import *
from .frostbite import *
from .saliency_guided_wrapper import GradientSaliencyWrapper
from .sarfa_wrapper import SARFAWrapper

# aliales to match the names in the paper
from .masked_dqn import (
    BinaryMaskWrapper as BinaryMasksWrapper,
    ObjectTypeMaskWrapper as ClassMasksWrapper,
    PixelMaskWrapper as ObjectMasksWrapper,
    ObjectTypeMaskPlanesWrapper as PlanesWrapper
)