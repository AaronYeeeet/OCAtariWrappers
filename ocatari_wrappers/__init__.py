from .masked_dqn import *
from .frostbite import *
from .sarfa_wrapper import SarfaSaliencyWrapper
from .sarfa_dual import SarfaDualWrapperFive, SarfaDualWrapperEight

# aliales to match the names in the paper
from .masked_dqn import (
    BinaryMaskWrapper as BinaryMasksWrapper,
    ObjectTypeMaskWrapper as ClassMasksWrapper,
    PixelMaskWrapper as ObjectMasksWrapper,
    ObjectTypeMaskPlanesWrapper as PlanesWrapper
)