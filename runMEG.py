import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import rpy2.robjects as robj
from easydict import EasyDict as edict
from tqdm import tqdm
import sys
import time as Time
import pickle
from scipy.io import loadmat
from scipy.signal import decimate, detrend
import seaborn as sns
from pathlib import Path
sys.path.append("/home/huaqingj/MyResearch/dynamicR2py")
from utils_copy import *
from pprint import pprint

resDir = Path("./results")
dataDir = Path("./data")

filname = Path("subj1.mat")
MEG = loadmat(dataDir/filname)["DK_timecourse"]


#Rate when doing decimate 
rate = 10 
#MEG = detrend(MEG) # The data are already detrended
#MEG = decimate(MEG, q=rate)
MEGlist = []
for i in range(MEG.shape[0]):
    MEGlist.append(decimate_R(MEG[i, :], rate))
MEG = np.array(MEGlist)


kappa = 2.65 # 2.95 MEG2, 2.65 MEG1
r = 10
Lmin = 60
# Down sample rate when estimating A matrix
downrate = 20 
MaxM = 19
lamb = 1e-4
fct = 0.5
d, n = MEG.shape
time = np.linspace(0, 2, n)
paras = {"kappa":kappa, "Lmin":Lmin, "r":r, "downrate":downrate, "MaxM":MaxM, "lamb":lamb, "rate":rate}
pprint(paras)


resFil = resDir/f"{filname.stem}_Rank{r}.pkl"


if not resFil.exists():
    t0 = Time()
    dXmat, Xmat = GetBsplienEst(MEG, time, lamb=lamb)
    Amat = GetAmat(dXmat, Xmat, time, downrate, fct=fct)
    midRes = GetNewEst(dXmat, Xmat, Amat, r=r, is_full=True)
    ndXmat, nXmat = midRes.ndXmat, midRes.nXmat
    finalRes = EGenDy(ndXmat, nXmat, kappa=kappa, Lmin=Lmin, MaxM=MaxM, diag=True)
    tc = Time()
    MEGRes = edict()
    MEGRes.PostMEG = MEG
    MEGRes.midRes = midRes
    MEGRes.finalRes = finalRes
    MEGRes.Amat = Amat
    MEGRes.dXmat = dXmat
    MEGRes.Xmat = Xmat
    MEGRes.paras = paras
    print(f"The running time is {tc-t0:.3f}.")
    with open(resFil, "wb") as f:
        pickle.dump(MEGRes, f)
else:
    with open(resFil, "rb") as f:
        MEGRes = pickle.load(f)
    

RecMEGfil = resDir/f"{filname.stem}_Rank{r}_Rec.pkl"
if not RecMEGfil.exists():
    t0 = Time()
    finalRes = MEGRes.finalRes
    midRes = MEGRes.midRes
    tStep = np.diff(time)[0]
    ecpts = finalRes.mbic_ecpts
    ndXmat = midRes.ndXmat
    nXmat = midRes.nXmat
    kpidxs = midRes.kpidxs
    eigVecs = midRes.eigVecs
    RecRes = ReconXmat(ecpts, ndXmat, nXmat, kpidxs, eigVecs, MEG, tStep, r=r, is_full=True) 
    tc = Time()
    print(f"The consumed time is {tc-t0:.2f}s.")
    with open(RecMEGfil, "wb") as f:
        pickle.dump(RecRes, f)
else:
    with open(RecMEGfil, "rb") as f:
        RecRes = pickle.load(f)
RecMEG = RecRes.EstXmatReal


RecMEGAllfil = resDir/f"{filname.stem}_Rank{r}_RecAll.pkl"
RecMEGAll = []
if not RecMEGAllfil.exists():
    for numchg in range(20):
        print(f"Current number of change point is {numchg}.")
        finalRes = MEGRes.finalRes
        midRes = MEGRes.midRes
        t0 = Time()
        tStep = np.diff(time)[0]
        ndXmat = midRes.ndXmat
        nXmat = midRes.nXmat
        kpidxs = midRes.kpidxs
        eigVecs = midRes.eigVecs
        if numchg == 0:
            RecResCur = ReconXmat([], ndXmat, nXmat, kpidxs, eigVecs, MEG, tStep, r=r, is_full=True) 
        else:
            RecResCur = ReconXmat(finalRes.chgMat[numchg-1, :numchg], ndXmat, nXmat, kpidxs, eigVecs, MEG, tStep, r=r, is_full=True) 
        RecMEGAll.append(RecResCur)
    with open(RecMEGAllfil, "wb") as f:
        pickle.dump(RecMEGAll, f)
else:
    with open(RecMEGAllfil, "rb") as f:
        RecMEGAll = pickle.load(f)
