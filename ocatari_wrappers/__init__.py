from .masked_dqn import *
from .frostbite import *
from .saliency_guided_wrapper import GradientSaliencyWrapper

# aliales to match the names in the paper
from .masked_dqn import (
    BinaryMaskWrapper as BinaryMasksWrapper,
    ObjectTypeMaskWrapper as ClassMasksWrapper,
    PixelMaskWrapper as ObjectMasksWrapper,
    ObjectTypeMaskPlanesWrapper as PlanesWrapper
)