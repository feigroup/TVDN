from easydict import EasyDict as edict
from .TVDNclass import TVDNDetect
import numpy as np

# Tuning the kappa w.r.t MSE based on CV
def TVDNKappaTuningCV(kappas, Ymat, numFold=10, wh=None, dataType=None, saveDir=None, **paras):
    """
    Input:
        kappas: The range of kappas to tune
        Ymat: The data matrix, d x n
        numFold: The number of folds for cross validation
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
        A dict containing:
            1. Optimal rank in the given ranks
            2. Optimal kappa in the given kappas
            3. The detection object under the optimal rank and kappa
            4. The minimal MSE in the given ranks and kappas
    """
    kappas = np.array(kappas)
    kappaCur = kappas[0]
    paras["kappa"] = kappaCur

    adjFct = numFold/(numFold-1)
    d, n = Ymat.shape
    MaxM = paras.MaxM

    CVRecXmat = np.zeros((d, n))
    CVRecXmats = [CVRecXmat] * len(kappas)
    
    for k in range(numFold):
        idxs = np.arange(n, step=numFold)+k
        idxs = idxs[idxs<n]
        Ymatk = np.delete(Ymat, idxs, axis=1)
        detection = TVDNDetect(Ymat=Ymatk, dataType=dataType, saveDir=saveDir, **paras)
        if wh is not None:
            detection.Screening(wh=wh)
        detection()

        U0 = detection.finalRes.U0
        rAct, _ = detection.midRes.nXmat.shape
        Us = []
        for kappac in kappas:
            Us.append(U0 + 2*rAct*np.log(n)**kappac* (np.arange(1, MaxM+2)))
        Us = np.array(Us)
        numchgs = Us.argmin(axis=1)
                
        for numchg in numchgs:
            if numchg == 0:
                pass
            else:
                pass
