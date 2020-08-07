import sys


def in_notebook():
    """
    Return True if the module is runing in Ipython kernel
    """
    return "ipykernel" in sys.modules
