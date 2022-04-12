import os

PATH = os.path.dirname(__file__)
with open(os.path.join(PATH, "VERSION"), "r") as rf:
        __version__ = rf.read().strip()

# user-facing modules
# -------------------

from . import vision
from . import nn
from . import utils
from . import adversarial
