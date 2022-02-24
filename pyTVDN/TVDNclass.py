import numpy as np
from pprint import pprint
from pathlib import Path
from scipy.signal import detrend
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from easydict import EasyDict as edict
from prettytable import PrettyTable
import warnings
from .TVDNutils import *
from .Rfuns import decimate_R
# from .utils import in_notebook
#if in_notebook():
#    from tqdm import tqdm_notebook as tqdm
#else:
from tqdm import tqdm

class TVDNDetect:
    def __init__(self, Ymat, smoothType="Bspline", dataType=None, saveDir=None, showProgress=True, **paras):
        """
        Input:
            Ymat: The data matrix, d x n
            dataType: real data type, fMRI or MEG
            saveDir: Dir to save the results, if not specified, not save
            paras: Other parameters. There are default valuesi but you may specify these parameters manually.
               Inlcuding:
                    kappa: The parameter of penalty in MBIC
                    Lmin: The minimal length between 2 change points
                    r: The rank of A matrix, in most cases, r=rAct. If we have non-complex singular values, r < rAct
                       If r is decimal, the rank is the number of eigen values which account for 100r % of the total variance
                       If r is integer, the r in algorithm can be r + 1 if r breaks the conjugate eigval pairs. 
                    MaxM: int, maximal number of change point 
                    lamb: The smooth parameter for B-spline
                    downRate: The downsample factor, determine how many Ai matrix to contribute to estimate the eigen values/vectors.
                    decimateRate: Mainly for MEG data. The rate to decimate from MEG data, reduce the resolution
                    T: The time course
                    is_detrend: Whether detrend data or not
                    fct: The factor to adjust h when estimating A matrix
                    fName:  The file name when saving the results
                    freq: The frequency of the data sequences, the parameter used drawing the eigen values plots
                    nKnots: number of knots for Bspline
        """
        self.Ymat = Ymat
        self.paras = edict()
        if dataType is not None:
            self.dataType = dataType.lower()
        else:
            self.dataType = dataType

        if smoothType is not None:
            self.smoothType = smoothType.lower()
        else:
            self.smoothType = smoothType


        if self.dataType == "meg":
            self.paras.kappa = 2.65
            self.paras.Lmin = 60
            self.paras.r = 6
            self.paras.MaxM = 19
            self.paras.lamb = 1e-4
            self.paras.downRate = 20
            self.paras.decimateRate = 10
            self.paras.T = 2
            self.paras.is_detrend = False
            self.paras.fct = 0.5
            self.paras.fName = "MEG"
            self.paras.freq = 60
            self.paras.nbasis = 10
            self.paras.nKnots = None
        elif self.dataType == "fmri":
            self.paras.kappa = 2.65
            self.paras.Lmin = 4
            self.paras.r = 6
            self.paras.MaxM = 10
            self.paras.lamb = 1e-4
            self.paras.downRate = 4
            self.paras.decimateRate = None
            self.paras.T = 2
            self.paras.is_detrend = False
            self.paras.fct = 0.5
            self.paras.fName = "fMRI"
            self.paras.freq = 0.5
            self.paras.nbasis = 10
            self.paras.nKnots = None
        else:
            self.paras.kappa = 2.65
            self.paras.Lmin = 4
            self.paras.r = 6
            self.paras.MaxM = 19
            self.paras.lamb = 1e-6
            self.paras.downRate = 4
            self.paras.decimateRate = None
            self.paras.T = 2
            self.paras.is_detrend = False
            self.paras.fct = 1
            self.paras.fName = "simu"
            self.paras.freq = 180
            self.paras.nbasis = 10
            self.paras.nKnots = None
        keys = list(self.paras.keys())
        for key in paras.keys():
            self.paras[key] = paras[key]
        if showProgress:
            print("The parameters for detection are:")
            pprint(self.paras)
            if isinstance(self.paras.r, int):
                print(f"The rank can be {self.paras.r+1} if {self.paras.r} breaks a eigval pair.")
        
        if saveDir is not None:
            self.saveDir = Path(saveDir)
            if not self.saveDir.exists():
                self.saveDir.mkdir()
        else:
            self.saveDir = saveDir
            
        self.showProgress = showProgress
        self.nYmat = None
        self.Xmat = None
        self.dXmat = None
        self.time = None
        self.midRes = None
        self.nXmat = None
        self.ndXmat = None
        self.Amat = None
        self.finalRes = None
        self.RecYmatAll = None
        self.RecResCur = None
        self.numchgs = None
        self.ecpts = None
        self.canpts = None
        self.curEigVecs = None
        self.curEigVals = None
    
    # Data preprocessing, including detrend and decimate
    def _Preprocess(self):
        nYmat = self.Ymat

        # Decimate the data first
        decimateRate = self.paras.decimateRate
        if decimateRate is not None:
            nYmatList = []
            # I use the decimate function in R to reproduce the results by the R version code
            # It is OK to use decimate function in python
            for i in range(nYmat.shape[0]):
                nYmatList.append(decimate_R(nYmat[i, :], decimateRate))
            nYmat = np.array(nYmatList)

        # Then Detrend the data
        is_detrend = self.paras.is_detrend
        if is_detrend:
            nYmat = detrend(nYmat)
            
        self.nYmat = nYmat
        _, n = self.nYmat.shape
        acTime = n / self.paras.freq
        self.ptime = np.linspace(0, acTime, n) 
        self.time = np.linspace(0, self.paras.T, n)
    
    def SmoothEst(self):
        if self.nYmat is None:
            self._Preprocess()
        _, n = self.nYmat.shape
        acTime = n / self.paras.freq
        self.ptime = np.linspace(0, acTime, n) 
        self.time = np.linspace(0, self.paras.T, n)
        if self.smoothType == "bspline":
            self.dXmat, self.Xmat = GetBsplineEst(self.nYmat, self.time, lamb=self.paras.lamb, nKnots=self.paras.nKnots)
        elif self.smoothType == "fourier":
            self.dXmat, self.Xmat = GetFourierEst(self.nYmat, self.time, nbasis=self.paras.nbasis)
    
    def GetAmat(self):
        downRate = self.paras.downRate
        fct = self.paras.fct
        if self.dXmat is None:
            self.SmoothEst()
            
        if self.saveDir is None:
            self.Amat = GetAmat(self.dXmat, self.Xmat, self.time, downRate, fct=fct)
        else:
            saveAmatPath = self.saveDir/f"{self.paras.fName}_Amat.pkl"
            if not saveAmatPath.exists():
                self.Amat = GetAmat(self.dXmat, self.Xmat, self.time, downRate, fct=fct)
                with open(saveAmatPath, "wb") as f:
                    pickle.dump(self.Amat, f)
            else:
                with open(saveAmatPath, "rb") as f:
                    self.Amat = pickle.load(f)
                
    
    
    def GetNewData(self):
        if self.Amat is None:
            self.GetAmat()

        eigVals, eigVecs = np.linalg.eig(self.Amat)
        if self.paras.r is None:
            rSel = np.where(np.cumsum(np.abs(eigVals))/np.sum(np.abs(eigVals)) >0.8)[0][0] + 1
            self.paras.r = rSel
        elif self.paras.r < 1:
            rSel = np.where(np.cumsum(np.abs(eigVals))/np.sum(np.abs(eigVals)) >self.paras.r)[0][0] + 1
            self.paras.r = rSel
        
        # if breaking conjugate eigval pair, add r with 1
        if (eigVals[self.paras.r-1].imag + eigVals[self.paras.r].imag ) == 0:
            self.paras.r = self.paras.r + 1

        r = self.paras.r
        
        self.midRes = GetNewEst(self.dXmat, self.Xmat, self.Amat, r=r, is_full=True)
        self.ndXmat, self.nXmat = self.midRes.ndXmat, self.midRes.nXmat
        
    # Get the scanning stats at index k
    def GetScanStats(self, k, wh):
        lidx = k - wh + 1
        uidx = k + wh + 1

        pndXmatA = self.ndXmat[:, lidx:uidx]
        pnXmatA = self.nXmat[:, lidx:uidx]
        GamkA = GetGammak(pndXmatA, pnXmatA)
        nlogA = GetNlogk(pndXmatA, pnXmatA, GamkA)

        pndXmatL = self.ndXmat[:, lidx:(k+1)]
        pnXmatL = self.nXmat[:, lidx:(k+1)]
        GamkL = GetGammak(pndXmatL, pnXmatL)
        nlogL = GetNlogk(pndXmatL, pnXmatL, GamkL)

        pndXmatR = self.ndXmat[:, (k+1):uidx]
        pnXmatR = self.nXmat[:, (k+1):uidx]
        GamkR = GetGammak(pndXmatR, pnXmatR)
        nlogR = GetNlogk(pndXmatR, pnXmatR, GamkR)

        return nlogR + nlogL - nlogA


    # Obtain the candidate point set via screening
    def Screening(self, wh=10):
        """
        Input:
            wh: screening window size
        """
        if self.midRes is None:
            self.GetNewData()
        _, n = self.ndXmat.shape
        scanStats = []
        if self.showProgress:
            for k in tqdm(range(n), desc="Screening"):
                if k < (wh-1):
                    scanStats.append(np.inf)
                elif k >= (n-wh):
                    scanStats.append(np.inf)
                else:
                    scanStats.append(self.GetScanStats(k, wh))
        else:
            for k in range(n):
                if k < (wh-1):
                    scanStats.append(np.inf)
                elif k >= (n-wh):
                    scanStats.append(np.inf)
                else:
                    scanStats.append(self.GetScanStats(k, wh))

        self.canpts = []
        for idx, scanStat in enumerate(scanStats):
            if (idx >= (wh-1)) and (idx < (n-wh)):
                lidx = idx - wh + 1
                uidx = idx + wh + 1
                if scanStat == np.min(scanStats[lidx:uidx]):
                    self.canpts.append(idx+1) # adjust the change point such that the starting point is from 1 not 0


        
    
    def __call__(self):
        kappa = self.paras.kappa
        Lmin = self.paras.Lmin
        MaxM = self.paras.MaxM

        if self.saveDir is not None:
            # Update the rank
            if self.Amat is None:
                self.GetAmat()
            eigVals, eigVecs = np.linalg.eig(self.Amat)
            if self.paras.r is None:
                rSel = np.where(np.cumsum(np.abs(eigVals))/np.sum(np.abs(eigVals)) >0.8)[0][0] + 1
                self.paras.r = rSel
            elif self.paras.r < 1:
                rSel = np.where(np.cumsum(np.abs(eigVals))/np.sum(np.abs(eigVals)) >self.paras.r)[0][0] + 1
                self.paras.r = rSel
        
            # if breaking conjugate eigval pair, add r with 1
            if (eigVals[self.paras.r-1].imag + eigVals[self.paras.r].imag ) == 0:
                self.paras.r = self.paras.r + 1

            saveResPath = self.saveDir/f"{self.paras.fName}_Rank{self.paras.r}.pkl"
            if not saveResPath.exists():
                if self.midRes is None:
                    self.GetNewData()
                self.finalRes = EGenDy(self.ndXmat, self.nXmat, r=self.paras.r, canpts=self.canpts, kappa=kappa, Lmin=Lmin, MaxM=MaxM, is_full=True, showProgress=self.showProgress)
                self.ecpts = self.finalRes.mbic_ecpts
                print(f"Save Main Results at {saveResPath}.")
                MainResults = edict()
                MainResults.nYmat = self.nYmat
                MainResults.Xmat = self.Xmat
                MainResults.Ymat = self.Ymat
                MainResults.midRes = self.midRes
                MainResults.finalRes = self.finalRes
                MainResults.Amat = self.Amat
                MainResults.paras = self.paras
                MainResults.ptime = self.ptime
                MainResults.canpts = self.canpts
                with open(saveResPath, "wb") as f:
                    pickle.dump(MainResults, f)
            else:
                warnings.warn("As loading the saved results, kappa will be ignored", UserWarning)
                with open(saveResPath, "rb") as f:
                    MainResults = pickle.load(f)
                    self.finalRes = MainResults.finalRes
                    self.ecpts = self.finalRes.mbic_ecpts
                    self.nYmat = MainResults.nYmat
                    self.Ymat = MainResults.Ymat
                    self.Xmat = MainResults.Xmat
                    self.midRes = MainResults.midRes
                    self.Amat = MainResults.Amat
                    self.ptime = MainResults.ptime
                
        else:
            if self.midRes is None:
                self.GetNewData()
            self.finalRes = EGenDy(self.ndXmat, self.nXmat, r=self.paras.r, canpts=self.canpts, kappa=kappa, Lmin=Lmin, MaxM=MaxM, is_full=True, showProgress=self.showProgress)
            self.ecpts = self.finalRes.mbic_ecpts
        self.GetRecResCur()
            
    # Plot the change point detection results
    def PlotEcpts(self, saveFigPath=None, GT=None):
        assert self.finalRes is not None, "Run main function first!"
        d, n = self.nYmat.shape
        acTime = n / self.paras.freq
        ajfct = n/acTime
        plt.figure(figsize=[10, 5])
        for i in range(d):
            plt.plot(self.ptime, self.nYmat[i, :], "-")

        if GT is not None:
            for j, cpt in enumerate(GT):
                if j == 0:
                    plt.axvline(cpt/ajfct, color="blue", linestyle="-", label="Ground truth")
                else:
                    plt.axvline(cpt/ajfct, color="blue", linestyle="-")

        for j, ecpt in enumerate(self.ecpts):
            if j == 0:
                plt.axvline(ecpt/ajfct, color="red", linestyle="--", label="Estimate")
            else:
                plt.axvline(ecpt/ajfct, color="red", linestyle="--")

            plt.legend(loc="upper left")
        if saveFigPath is None:
            plt.show() 
        else:
            plt.savefig(saveFigPath)
        

    # Plot the change point detection results
    def PlotEcptsFull(self, saveFigPath=None, GT=None):
        assert self.finalRes is not None, "Run main function first!"
        d, n = self.Ymat.shape
        acTime = n / self.paras.freq/ self.paras.decimateRate
        ajfct = n/acTime
        plt.figure(figsize=[10, 5])
        cptime = np.linspace(0, acTime, n)
        for i in range(d):
            plt.plot(cptime, self.Ymat[i, :], "-")

        if GT is not None:
            for j, cpt in enumerate(GT):
                if j == 0:
                    plt.axvline(cpt/ajfct, color="blue", linestyle="-", label="Ground truth")
                else:
                    plt.axvline(cpt/ajfct, color="blue", linestyle="-")

        for j, ecpt in enumerate(self.ecpts):
            if j == 0:
                plt.axvline(ecpt*self.paras.decimateRate/ajfct, color="red", linestyle="--", label="Estimate")
            else:
                plt.axvline(ecpt*self.paras.decimateRate/ajfct, color="red", linestyle="--")

            plt.legend(loc="upper left")
        if saveFigPath is None:
            plt.show() 
        else:
            plt.savefig(saveFigPath)
    
    # Plot reconstructed Ymat curve 
    def PlotRecCurve(self, idxs=None, bestK=None, quantiles=None, saveFigPath=None, is_smoothCurve=False):
        """
        idxs: The indices of the sequences to plot 
        bestK: The best K fitted curves to plot according to the errors
        quantiles: The fitted sequences to plot according to the quantiles of errors.
        (priority: idxs > bestK > quantiles)_
        """
        assert self.finalRes is not None, "Run main function first!"
        if idxs is not None and (bestK is not None or quantiles is not None):
            warnings.warn("idxs is provided, so bestK or quantiles will be ignored", UserWarning)
        if idxs is None and bestK is not None and quantiles is not None:
            warnings.warn("bestK is provided, so quantiles will be ignored", UserWarning)
        if self.RecResCur is None:
            self.GetRecResCur()
            
        # or detrend version
        RecYmatCur = self.RecResCur.EstXmatRealOrg
        d, n = self.nYmat.shape
        if idxs is not None:
            assert d>=np.max(idxs) & np.min(idxs)>=0, "Wrong index!"
        else:
            diff = RecYmatCur - self.nYmat
            errs2 = np.sum(diff**2, axis=1)/np.sum(self.nYmat**2, axis=1)
            errs = np.sqrt(errs2)
            argidxs = np.argsort(errs)
            if quantiles is None and bestK is None:
                qidxs = np.quantile(np.arange(d), [0, 0.25, 0.5, 0.75, 1]).astype(np.int)
            elif bestK is not None:
                qidxs = argidxs[:bestK]
            else:
                qidxs = np.quantile(np.arange(d), quantiles).astype(np.int)
            idxs = argidxs[qidxs]
        if self.showProgress:
            print(f"The plot indices are {idxs}.")
        
        
        numSubPlot = len(idxs)
        numRow = ((numSubPlot-1) // 3) + 1
        
        plt.figure(figsize=[15, 5*numRow])

        for i, idx, in enumerate(idxs):
            plt.subplot(numRow, 3, i+1)
            plt.plot(self.ptime, self.nYmat[idx, :], "-", label="Observed")
            plt.plot(self.ptime, RecYmatCur[idx, :], "-.", label="Reconstructed")
            if is_smoothCurve:
                if self.Xmat is None:
                    self.SmoothEst()
                plt.plot(self.ptime, self.Xmat[idx, :], "r--", label=f"{self.smoothType} Estimator")
            #plt.legend()
            plt.legend(loc="upper left")
        if saveFigPath is None:
            plt.show() 
        else:
            plt.savefig(saveFigPath)

        return idxs
    
    # Plot the eigen value curve
    def PlotEigenCurve(self, saveFigPath=None):
        assert self.finalRes is not None, "Run main function first!"
        if self.RecResCur is None:
            self.GetRecResCur()
        freq = self.paras.freq
        numChgCur = len(self.ecpts)
        LamMs = self.RecResCur.LamMs
        rAct, n = LamMs.shape
        pltIdxs = np.arange(1, rAct)[np.diff(np.abs(LamMs), axis=0).astype(np.bool).all(axis=1)] 
        pltIdxs = np.concatenate([[0], pltIdxs])
        acTime = n / self.paras.freq
        ReLamMs = freq*LamMs.real/(acTime/self.paras.T)
        ImLamMs = freq*LamMs.imag/((2*np.pi)*(acTime/self.paras.T))
        cols = sns.color_palette("Paired", len(pltIdxs))
        
        plt.figure(figsize=[10, 5])
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.subplot(121)
        for i0, i in enumerate(pltIdxs):
            labs = f"$\\lambda_{i0+1}$"
            plt.plot(self.ptime, np.abs(ReLamMs[i, :]), label=labs, 
                     color=cols[i0], linewidth=2)
        plt.ylabel("Change of growth/decay constant")
        plt.xlabel("Time")
        #_ = plt.legend()
        _ = plt.legend(loc="upper left")
        
        plt.subplot(122)
        for i0, i in enumerate(pltIdxs):
            labs = f"$\\lambda_{i0+1}$"
            plt.plot(self.ptime, np.abs(ImLamMs[i, :]), label=labs, 
                     color=cols[i0], linewidth=2)
        plt.ylabel("Change of frequencyy")
        plt.xlabel("Time")
        # _ = plt.legend()
        _ = plt.legend(loc="upper left")
        if saveFigPath is None:
            plt.show() 
        else:
            plt.savefig(saveFigPath)

    def GetCurMSE(self):
        assert self.finalRes is not None, "Run main function first!"
        if self.RecResCur is None:
            self.GetRecResCur()
        RecYmatCur = self.RecResCur.EstXmatRealOrg
        #deltaT = np.diff(self.time)[0]
        #MSE = np.sqrt(np.sum((RecYmatCur-self.Xmat)**2)/np.sum(self.Xmat**2))
        MSE = np.sqrt(np.sum((RecYmatCur-self.nYmat)**2)/np.sum(self.nYmat**2))
        #MSE = np.mean((RecYmatCur-self.nYmat)**2)
        return MSE


    def GetRecResCur(self):
        numchg = len(self.ecpts)
        assert self.finalRes is not None, "Run main function first!"
        if self.RecYmatAll is not None:
            self.RecResCur = self.RecYmatAll[numchg]
        elif self.saveDir is not None:
            RecYmatAllPath = self.saveDir/f"{self.paras.fName}_Rank{self.paras.r}_RecAll.pkl"
            if RecYmatAllPath.exists():
                with open(RecYmatAllPath, "rb") as f:
                    self.RecYmatAll = pickle.load(f)
                self.RecResCur = self.RecYmatAll[numchg]
            else:
                MaxM = self.paras.MaxM
                r = self.paras.r
                finalRes = self.finalRes
                midRes = self.midRes
                _, n = midRes.nXmat.shape
                time = np.linspace(0, self.paras.T, n)
                tStep = np.diff(time)[0]
                ndXmat = midRes.ndXmat
                nXmat = midRes.nXmat
                kpidxs = midRes.kpidxs
                eigVecs = midRes.eigVecs
                self.RecResCur = ReconXmat(finalRes.chgMat[numchg-1, :numchg], ndXmat, nXmat, kpidxs, eigVecs, self.nYmat, tStep, r=r, is_full=True) 
        else:
            MaxM = self.paras.MaxM
            r = self.paras.r
            finalRes = self.finalRes
            midRes = self.midRes
            _, n = midRes.nXmat.shape
            time = np.linspace(0, self.paras.T, n)
            tStep = np.diff(time)[0]
            ndXmat = midRes.ndXmat
            nXmat = midRes.nXmat
            kpidxs = midRes.kpidxs
            eigVecs = midRes.eigVecs
            self.RecResCur = ReconXmat(finalRes.chgMat[numchg-1, :numchg], ndXmat, nXmat, kpidxs, eigVecs, self.nYmat, tStep, r=r, is_full=True) 
    
    def GetRecYmats(self):
        if self.RecYmatAll is None:
            RecYmatAll = []
            MaxM = self.paras.MaxM
            r = self.paras.r
            finalRes = self.finalRes
            midRes = self.midRes
            _, n = midRes.nXmat.shape
            time = np.linspace(0, self.paras.T, n)
            tStep = np.diff(time)[0]
            ndXmat = midRes.ndXmat
            nXmat = midRes.nXmat
            kpidxs = midRes.kpidxs
            eigVecs = midRes.eigVecs
            if self.showProgress:
                pbar = tqdm(range(MaxM+1))
                for numchg in pbar:
                    pbar.set_description(f"Kappa Tuning")
        #            print(f"Current number of change point is {numchg}.")
                    if numchg == 0:
                        RecResCur = ReconXmat([], ndXmat, nXmat, kpidxs, eigVecs, self.nYmat, tStep, r=r, is_full=True) 
                    else:
                        RecResCur = ReconXmat(finalRes.chgMat[numchg-1, :numchg], ndXmat, nXmat, kpidxs, eigVecs, self.nYmat, tStep, r=r, is_full=True) 
                    RecYmatAll.append(RecResCur)
            else:
                for numchg in range(MaxM+1):
                    if numchg == 0:
                        RecResCur = ReconXmat([], ndXmat, nXmat, kpidxs, eigVecs, self.nYmat, tStep, r=r, is_full=True) 
                    else:
                        RecResCur = ReconXmat(finalRes.chgMat[numchg-1, :numchg], ndXmat, nXmat, kpidxs, eigVecs, self.nYmat, tStep, r=r, is_full=True) 
                    RecYmatAll.append(RecResCur)
            self.RecYmatAll = RecYmatAll
    
    
    # Tuning the kappa parameters w.r.t MSE error
    def TuningKappa(self, kappas):
        assert self.finalRes is not None, "Run main function first!"
        
        MaxM = self.paras.MaxM
        U0 = self.finalRes.U0 
        rAct, n = self.midRes.nXmat.shape
        r = self.paras.r
        Us = []
        for kappac in kappas:
            Us.append(U0 + 2*r*np.log(n)**kappac* (np.arange(1, MaxM+2)))
        Us = np.array(Us)
        numchgs = Us.argmin(axis=1)
        self.numchgs = numchgs
        time = np.linspace(0, self.paras.T, n)
        
        if self.saveDir is not None:
            RecYmatAllPath = self.saveDir/f"{self.paras.fName}_Rank{self.paras.r}_RecAll.pkl"
            if not RecYmatAllPath.exists():
                self.GetRecYmats()
                with open(RecYmatAllPath, "wb") as f:
                    pickle.dump(self.RecYmatAll, f)
            else:
                with open(RecYmatAllPath, "rb") as f:
                    self.RecYmatAll = pickle.load(f)
        else:
            self.GetRecYmats()
            
        MSEs = []
        for i in range(MaxM+1):
            RecYmatCur = self.RecYmatAll[i].EstXmatRealOrg
            #deltaT = np.diff(self.time)[0]
            MSE = np.sqrt(np.sum((RecYmatCur-self.Xmat)**2)/np.sum(self.Xmat**2))
            #MSE = np.sqrt(np.sum((RecYmatCur-self.nYmat)**2)/np.sum(self.nYmat**2))
            #MSE = np.mean((RecYmatCur-self.nYmat)**2)
            MSEs.append(MSE)
        self.MSEs = MSEs
        
        self.optNumChg = np.argmin(MSEs)
        
        MSEsKappa = [MSEs[i] for i in numchgs]
        self.optKappa = kappas[np.argmin(MSEsKappa)]
        self.optKappaOptNumChg = numchgs[np.argmin(MSEsKappa)]
        self.kappas = kappas

    def PlotKappaErrCurve(self):
        assert self.MSEs is not None, "Run the TuningKappa first!"
        MSEs = np.array(self.MSEs)
        numchgs = np.array(self.numchgs)
        plt.figure(figsize=[15, 5])

        plt.subplot(131)
        plt.plot(self.kappas, MSEs[numchgs], ".-")
        plt.ylabel("Error")
        _ = plt.xlabel("Kappa")

        plt.subplot(132)
        plt.plot(self.kappas, numchgs, ".-")
        plt.ylabel("Num of Change points")
        _ = plt.xlabel("Kappa")

        plt.subplot(133)
        plt.plot(MSEs, ".-")
        plt.xlabel("Num of Change points")
        _ = plt.ylabel("Error")
        plt.show()

    

    def UpdateEcpts(self, numChg=None):
        assert self.finalRes is not None, "Run main function first!"
        if numChg is None:
            assert self.RecYmatAll is not None, "Run TuningKappa function first!"
            numChg = self.optKappaOptNumChg
        if numChg == 0:
            self.ecpts = np.array([])
        else:
            self.ecpts = self.finalRes.chgMat[numChg-1, :numChg]
        self.GetRecResCur()
    

    def __str__(self):
        tb = PrettyTable(["Num of CPTs", "Estiamted CPTs", "MSE", "Rank"])
        if self.finalRes is None:
            print("Run main function fisrt!")
            tb.add_row([None, None, None, self.paras.r])
        else:
            MSE = self.GetCurMSE()
            tb.add_row([len(self.ecpts), self.ecpts, MSE, self.paras.r])
        return tb.__str__()
    
    def GetFeatures(self):
        """
        obtain the eigvals and eigvectors for current ecpts
        """
        if self.RecResCur is None:
            self.GetRecResCur()
        Ur = self.midRes.eigVecs[:, :self.paras.r]
            
        lamMs = []
        for idx, ecpt in enumerate(np.concatenate([[0], self.ecpts])):
            lamM = self.RecResCur.LamMs[:, int(ecpt)]
            lamMs.append(lamM)
        
        self.curEigVecs = Ur
        self.curEigVals = lamMs
