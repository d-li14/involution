from glob import glob
import os

from torch import ops

_LIB_PATH = glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'involution.*.so'))[0]
ops.load_library(_LIB_PATH)

from .involution2d import Involution2d
