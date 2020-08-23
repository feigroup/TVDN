from easydict import EasyDict as edict
from scipy.signal import detrend
import numpy as np
from .Rfuns import decimate_R
from .TVDNclass import TVDNDetect
from .TVDNutils import ReconXmatCV
from .utils import in_notebook
if in_notebook():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

# Tuning the kappa w.r.t MSE based on CV
def TVDNKappaTuningCV(kappas, Ymat, numFold=10, numTimes=None, randomSel=False, wh=None, dataType=None, saveDir=None, **paras):
    """
    Input:
        kappas: The range of kappas to tune
        Ymat: The data matrix, d x n
        numFold: The number of folds for cross validation
        numTimes: The number of times to do CV, if None, numTimes=numFold
        randomSel: Whether randomly selecting the deleted points or not
        wh: The window size for screening step. If None, no screening step
        dataType: real data type, fMRI or MEG
        saveDir: Dir to save the results, if not specified, not save
        paras: Other parameters. There are default values but you may specify these parameters manually.
            Inlcuding:
            r: The rank for the detection
            kappa: The parameter of penalty in MBIC
            Lmin: The minimal length between 2 change points
            MaxM: int, maximal number of change point 
            lamb: The smooth parameter for B-spline
            downRate: The downsample factor, determine how many Ai matrix to contribute to estimate the eigen values/vectors.
            decimateRate: Mainly for MEG data. The rate to decimate from MEG data.
            T: The time course
            is_detrend: Whether detrend data or not
            fct: The factor to adjust h when estimating A matrix
            fName:  The file name when saving the results
            plotfct: The factor to adjust the time course when plotting
            freq: The parameter used drawing the eigen values plots
    Return:
       Optimal Kappa in kappas
    """
    paras = edict(paras)
    kappas = np.array(kappas)
    kappaCur = kappas[0]
    paras["kappa"] = kappaCur
    
    if dataType is not None:
        dataType = dataType.lower()
    else:
        dataType = dataType
    
    if dataType == "meg":
        is_detrend = False
        decimateRate = 10
        fName = "MEG"
    elif dataType == "fmri":
        is_detrend = False
        decimateRate = None
        fName = "fMRI"
    else:
        is_detrend = False
        decimateRate = None
        fName = "others"

    if "is_detrend" in paras.keys():
        is_detrend = paras["is_detrend"]
    if is_detrend:
        Ymat = detrend(Ymat)

    if "decimateRate" in paras.keys():
        decimateRate = paras["decimateRate"]
    if decimateRate is not None:
        YmatList = []
        for i in range(Ymat.shape[0]):
            YmatList.append(decimate_R(Ymat[i, :], decimateRate))
        Ymat = np.array(YmatList)
    
    if "fName" in paras.keys():
        fName = paras["fName"]

    paras["is_detrend"] = False
    paras["decimateRate"] = None

    adjFct = numFold/(numFold-1)
    d, n = Ymat.shape

    MSEssKappa = []

    if numTimes is None:
        numTimes = numFold
    
    for k in tqdm(range(numTimes),  desc="Cross Validation"):
        #print("="*100)
        #print(f"The {k+1}th/{numTimes} cross validation.")
        paras["fName"] = f"{fName}_{k+1}th"
        if not randomSel:
            idxs = np.arange(n, step=numFold)+k
            idxs = idxs[idxs<n]
        else:
            idxs = np.sort(np.random.choice(n, size=int(n/numFold), replace=False))

        Ymatk = np.delete(Ymat, idxs, axis=1)
        Ymatidxs = Ymat[:, idxs]
        detection = TVDNDetect(Ymat=Ymatk, dataType=dataType, saveDir=saveDir, showProgress=False, **paras)
        r = detection.paras.r
        MaxM = detection.paras.MaxM
        if wh is not None:
            detection.Screening(wh=wh)
        detection()

        U0 = detection.finalRes.U0
        _, nc = detection.midRes.nXmat.shape
        Us = []
        for kappac in kappas:
            Us.append(U0 + 2*r*np.log(nc)**kappac* (np.arange(1, MaxM+2)))
        Us = np.array(Us)
        numchgs = Us.argmin(axis=1)
                
        midRes = detection.midRes
        time = np.linspace(0, detection.paras.T, n)
        tStep = np.diff(time)[0]
        r = detection.paras.r
        ndXmat = midRes.ndXmat
        nXmat = midRes.nXmat
        kpidxs = midRes.kpidxs
        eigVecs = midRes.eigVecs
        MSEs = []
        for numchg in range(MaxM+1):
            if numchg == 0:
                RecYmatCV = ReconXmatCV([], ndXmat, nXmat, kpidxs, eigVecs, Ymat, tStep, r=r, adjFct=adjFct, nFull=n, is_full=False) 
            else:
                RecYmatCV = ReconXmatCV(detection.finalRes.chgMat[numchg-1, :numchg], ndXmat, nXmat, kpidxs, eigVecs, Ymat, tStep, 
                                        r=r, adjFct=adjFct, nFull=n, is_full=False) 
            RecYmatCVidxs = RecYmatCV[:, idxs]
            #MSE = np.mean((RecYmatCVidxs-Ymatidxs)**2)
            MSE = np.sqrt(np.sum((RecYmatCVidxs-Ymatidxs)**2)/np.sum(Ymatidxs**2))
            MSEs.append(MSE)
        MSEsKappa = [MSEs[i] for i in numchgs]
        MSEssKappa.append(MSEsKappa)

    MSEssKappaArr = np.array(MSEssKappa)
    MSEssAv = MSEssKappaArr.mean(axis=0)
    bestKappaIdx = np.argmin(MSEssAv)
    bestKappa = kappas[bestKappaIdx]

    return bestKappa

