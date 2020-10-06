from .TVDNclass import TVDNDetect
from easydict import EasyDict as edict
import numpy as np


def TVDNRankTuning(ranks, kappas, Ymat, dataType=None, saveDir=None, showProgress=False, **paras):
    """
    Input:
        ranks: The range of ranks to tune
        kappas: The range of kappas to tune
        Ymat: The data matrix, d x n
        dataType: real data type, fMRI or MEG
        saveDir: Dir to save the results, if not specified, not save
        paras: Other parameters. There are default values but you may specify these parameters manually.
            Inlcuding:
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

    try:
        len(kappas)
    except:
        kappas = [kappas]
    kappas = np.array(kappas)

    kappaCur = kappas[0]
    paras["kappa"] = kappaCur

    MSEs = []
    optKappasCur = []
    detections = []
    for rank in ranks:
        if showProgress:
            print("="*50)
            print(f"The current rank is {rank}.")
        paras["r"] = rank
        detection = TVDNDetect(Ymat=Ymat, dataType=dataType, saveDir=saveDir, showProgress=showProgress, **paras)
        detection()
        if len(kappas) == 1:
            MSE = detection.GetCurMSE()
            optKappaCur = kappaCur
        else:
            detection.TuningKappa(kappas)
            MSEKappas = [detection.MSEs[i] for i in detection.numchgs]

            #optMSECur = np.min(detection.MSEs)
            #optNum = np.argmin(detection.MSEs)

            optKappaCur = kappas[MSEKappas==np.min(MSEKappas)]
            optKappaOptNumChg = detection.numchgs[np.argmin(MSEKappas)]
            MSE = detection.MSEs[optKappaOptNumChg]

        MSEs.append(MSE)
        optKappasCur.append(optKappaCur)
        detections.append(detection)

        if showProgress:
            print("="*50)

    optRank = ranks[np.argmin(MSEs)]
    optKappa = optKappasCur[np.argmin(MSEs)]
    optDetect = detections[np.argmin(MSEs)]
    if len(kappas) != 1:
        optDetect.UpdateEcpts()

    Res = edict()
    Res.minErr = np.min(MSEs)
    Res.optRank = optRank
    Res.optKappa = optKappa
    Res.DetectObj = optDetect
    return Res
