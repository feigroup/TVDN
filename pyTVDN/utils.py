import sys
import numpy as np
from numpy.linalg import inv
from pathlib import Path
from scipy.stats import iqr
import pickle


# bw.nrd0 fn in R
def pybwnrd0(x):
    hi = np.std(x, ddof=1)
    lo = np.min((hi, iqr(x)/1.34))
    eps = 1e-10
    if np.abs(lo-0) <= eps:
        if np.abs(hi-0) > eps:
            lo = hi
        elif np.abs(x[0]-0) > eps:
            lo = x[0]
        else:
            lo = 1
    rev = 0.9 * lo * len(x)**(-0.2)
    return rev    

def in_notebook():
    """
    Return True if the module is runing in Ipython kernel
    """
    return "ipykernel" in sys.modules



