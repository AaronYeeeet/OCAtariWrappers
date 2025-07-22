from .masked_dqn import *
from .frostbite import *

# aliales to match the names in the paper
from .masked_dqn import (
    BinaryMaskWrapper as BinaryMaskWrapper,
    ObjectTypeMaskWrapper as ClassMaskWrapper,
    PixelMaskWrapper as ObjectMaskWrapper,
    ObjectTypeMaskPlanesWrapper as PlanesWrapper
)