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



# The function to generate on simulation data
def GenSingleData(time, cpts, U=None, Vs=None):
    """
    Input:
        time: The sampling time points, n.
        cpts: The ground truth change point set.
        U: The ground truth eigen vctors, d x d.  If None, loading default U and Vs
        Vs: The matrix whose column is the eigen values for each segment, r*(num of cpts + 1).
    return:
        Xmat, A dataset without errors
        Ymat, A dataset with errors
    """
        
        
    if U is None:
        filePath = Path(__file__).parent
        with open(filePath/"data/SimuEigen.pkl", "rb") as f:
            SimuEigen = pickle.load(f)
            U = SimuEigen["U"]
            Vs = SimuEigen["Vs"]
            
    d, _ = U.shape
    n = len(time)
    r, _ = Vs.shape
    tStep = np.diff(time)[0]
    VsTmp = Vs
    Vs = np.zeros((d, len(cpts)+1), dtype=np.complex)
    Vs[:r, :] = VsTmp
    cpts = np.array(cpts) - 1
    
    k = 0
    errV = U.dot(np.diag(Vs[:, k])/10).dot(inv(U)) * np.random.binomial(1, 0.1, (d, d))
    
    XmatCur = np.random.normal(loc=0, scale=tStep/8, size=d)
    dXmatCur = U.dot(np.diag(Vs[:, k])).dot(inv(U)).dot(XmatCur)*tStep
    YmatCur = XmatCur + errV.dot(np.random.normal(loc=0, scale=tStep/8, size=d))
    XmatList = [XmatCur]
    dXmatList = [dXmatCur]
    YmatList = [YmatCur]
    for i in range(1, n):
        XmatCur = XmatList[-1] + dXmatList[-1]
        YmatCur = XmatCur + errV.dot(np.random.normal(loc=0, scale=tStep/8, size=d))
        dXmatCur = U.dot(np.diag(Vs[:, k])).dot(inv(U)).dot(XmatCur)*tStep
        XmatList.append(XmatCur)
        dXmatList.append(dXmatCur)
        YmatList.append(YmatCur)
        if i in cpts:
            k += 1
    Xmat = np.array(XmatList, dtype=np.complex).T
    Ymat = np.array(YmatList, dtype=np.complex).T
    return Xmat.real, Ymat.real
        
    
def GenSimuData(nSim, time, cpts, U=None, Vs=None):
    """
    Input:
        nSim: The number of dataset to generate
        U: The ground truth eigen vctors, d x d. 
        Vs: The matrix whose column is the eigen values for each segment, (num of cpts + 1) x r. 
        time: The sampling time points, n.
        cpts: The ground truth change point set.
    return:
        Xmats, A list of dataset without errors
        Ymats, A list of dataset with errors
    """
    Xmats = []
    Ymats = []
    for i in range(nSim):
        Xmat, Ymat = GenSingleData(time, cpts, U, Vs)
        Xmats.append(Xmat)
        Ymats.append(Ymat)
    return Xmats, Ymats
    


