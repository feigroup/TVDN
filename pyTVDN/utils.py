import sys
import numpy as np
from numpy.linalg import inv
from pathlib import Path
import pickle


def in_notebook():
    """
    Return True if the module is runing in Ipython kernel
    """
    return "ipykernel" in sys.modules



