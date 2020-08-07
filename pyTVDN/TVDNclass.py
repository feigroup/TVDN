import numpy as np
from pprint import pprint
from pathlib import Path
from scipy.signal import detrend
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from easydict import EasyDict as edict
from .TVDNutils import *
from .Rfuns import decimate_R

class TVDNDetect:
    def __init__(self, Ymat, dataType=None, saveDir=None, **paras):
        """
        Ymat: The data matrix, d x n
        dataType: real data type, fMRI or MEG
        saveDir: Dir to save the results, if not specified, not save
        paras: Other parameters. There are default parameters but you may specify these parameters manually.
           Inlcuding:
                kappa: The parameter of penalty
                Lmin: The minimal length between 2 change points
                r: The rank setted beforehand, in most cases, r=rAct. If we have non-complex singular values, r < rAct
                MaxM: int, maximal number of change point 
                lamb: The smooth parameter for B-spline
                downRate: The downsample factor, determine how many Ai matrix to contribute to estimate the eigen values/vectors.
                decimateRate: Mainly for MEG data. The rate to decimate from MEG data.
                T: The time course
                is_detrend: Whether detrend data or not
                fct: The factor to ajust h when estimating A matrix
                fName:  The file name when saving the results
                plotfct: The factor to ajust the time course when plotting
                freq: The parameter used drawing the eigen values plots
        """
        self.Ymat = Ymat
        self.paras = edict()
        self.dataType = dataType.lower()
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
            self.paras.plotfct = 30
            self.paras.freq = 30
        elif self.dataType == "fmri":
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
            self.paras.fName = "fMRI"
            self.paras.plotfct = 1
            self.paras.freq = 180
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
            self.paras.plotfct = 1
            self.paras.freq = 180
        keys = list(self.paras.keys())
        for key in paras.keys():
            self.paras[key] = paras[key]
        print("The parameters for detection are:")
        pprint(self.paras)
        
        if saveDir is not None:
            self.saveDir = Path(saveDir)
            if not self.saveDir.exists():
                self.saveDir.mkdir()
        else:
            self.saveDir = saveDir
            
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
    
    # Data preprocessing, including detrend and decimate
    def _preprocess(self):
        # Detrend the data
        is_detrend = self.paras.is_detrend
        if is_detrend:
            nYmat = detrend(self.Ymat)
        else:
            nYmat = self.Ymat
            
        # Decimate the data
        decimateRate = self.paras.decimateRate
        if decimateRate is not None:
            nYmatList = []
            for i in range(nYmat.shape[0]):
                nYmatList.append(decimate_R(nYmat[i, :], decimateRate))
            self.nYmat = np.array(nYmatList)
        else:
            self.nYmat = nYmat
        _, n = self.nYmat.shape
        self.ptime = np.linspace(0, self.paras.T, n) * self.paras.plotfct
    
    def GetBsplineEst(self):
        if self.nYmat is None:
            self._preprocess()
        d, n = self.nYmat.shape
        T = self.paras.T
        lamb = self.paras.lamb
        self.time = np.linspace(0, T, n)
        self.dXmat, self.Xmat = GetBsplineEst(self.nYmat, self.time, lamb=lamb)
    
    def GetAmat(self):
        downRate = self.paras.downRate
        fct = self.paras.fct
        if self.dXmat is None:
            self.GetBsplineEst()
        self.Amat = GetAmat(self.dXmat, self.Xmat, self.time, downRate, fct=fct)
    
    
    def GetNewData(self):
        r = self.paras.r
        
        if self.Amat is None:
            self.GetAmat()
        
        self.midRes = GetNewEst(self.dXmat, self.Xmat, self.Amat, r=r, is_full=True)
        self.ndXmat, self.nXmat = self.midRes.ndXmat, self.midRes.nXmat
    
    def __call__(self):
        kappa = self.paras.kappa
        Lmin = self.paras.Lmin
        MaxM = self.paras.MaxM
        
        if self.saveDir is not None:
            saveResPath = self.saveDir/f"{self.paras.fName}_Rank{self.paras.r}.pkl"
            if not saveResPath.exists():
                if self.midRes is None:
                    self.GetNewData()
                self.finalRes = EGenDy(self.ndXmat, self.nXmat, kappa=kappa, Lmin=Lmin, MaxM=MaxM, is_full=True)
                self.ecpts = self.finalRes.mbic_ecpts
                print(f"Save Main Results at {saveResPath}.")
                MainResults = edict()
                MainResults.nYmat = self.nYmat
                MainResults.Ymat = self.Ymat
                MainResults.midRes = self.midRes
                MainResults.finalRes = self.finalRes
                MainResults.Amat = self.Amat
                MainResults.paras = self.paras
                MainResults.ptime = self.ptime
                with open(saveResPath, "wb") as f:
                    pickle.dump(MainResults, f)
            else:
                with open(saveResPath, "rb") as f:
                    MainResults = pickle.load(f)
                    self.finalRes = MainResults.finalRes
                    self.ecpts = self.finalRes.mbic_ecpts
                    self.nYmat = MainResults.nYmat
                    self.Ymat = MainResults.Ymat
                    self.midRes = MainResults.midRes
                    self.Amat = MainResults.Amat
                    self.ptime = MainResults.ptime
                
        else:
            if self.midRes is None:
                self.GetNewData()
            self.finalRes = EGenDy(self.ndXmat, self.nXmat, kappa=kappa, Lmin=Lmin, MaxM=MaxM, is_full=True)
            self.ecpts = self.finalRes.mbic_ecpts
            
    # Plot the change point detection results
    def PlotECPTs(self, saveFigPath=None):
        assert self.finalRes is not None, "Run main function first!"
        d, n = self.nYmat.shape
        ajfct = n/(self.paras.plotfct*self.paras.T)
        plt.figure(figsize=[10, 5])
        for i in range(d):
            plt.plot(self.ptime, self.nYmat[i, :], "-")
        for ecpt in self.ecpts:
            plt.axvline(ecpt/ajfct, color="black", linestyle="-.")
        if saveFigPath is None:
            plt.show() 
        else:
            plt.savefig(saveFigPath)
        
    
    # Plot reconstructed Ymat curve 
    def PlotRecCurve(self, idxs, saveFigPath=None):
        """
        idxs: The indices of the sequences to plot 
        """
        assert self.finalRes is not None, "Run main function first!"
        if self.RecResCur is None:
            self.__GetRecResCur()
        d, n = self.nYmat.shape
        numChgCur = len(self.ecpts)
        RecYmatCur = self.RecResCur.EstXmatReal
        assert d>=np.max(idxs) & np.min(idxs)>=0, "Wrong index!"
        
        
        numSubPlot = len(idxs)
        numRow = ((numSubPlot-1) // 3) + 1
        
        plt.figure(figsize=[15, 5*numRow])

        for i, idx, in enumerate(idxs):
            plt.subplot(numRow, 3, i+1)
            plt.plot(self.ptime, self.nYmat[idx, :], label="Observed")
            plt.plot(self.ptime, RecYmatCur[idx, :], label="Estimated")
            plt.legend()
        if saveFigPath is None:
            plt.show() 
        else:
            plt.savefig(saveFigPath)
    
    # Plot the eigen value curve
    def PlotEigenCurve(self, saveFigPath=None):
        assert self.finalRes is not None, "Run main function first!"
        if self.RecResCur is None:
            self.__GetRecResCur()
        freq = self.paras.freq
        numChgCur = len(self.ecpts)
        LamMs = self.RecResCur.LamMs
        ReLamMs = LamMs.real*freq/30 
        ImLamMs = LamMs.imag*freq /(30*2*np.pi)
        cols = sns.color_palette("Paired", ReLamMs.shape[0])
        
        plt.figure(figsize=[20,10])
        plt.subplot(121)
        for i in range(ReLamMs.shape[0]):
            plt.plot(self.ptime, ReLamMs[i, :], label=f"Lam {i+1}", 
                     color=cols[i], linewidth=2)
        plt.ylabel("change of growth/decay constant")
        plt.xlabel("time")
        _ = plt.legend()
        
        plt.subplot(122)
        for i in range(ReLamMs.shape[0]):
            plt.plot(self.ptime, ImLamMs[i, :], label=f"Lam {i+1}", 
                     color=cols[i], linewidth=2)
        plt.ylabel("change of growth/decay constant")
        plt.xlabel("time")
        _ = plt.legend()
        if saveFigPath is None:
            plt.show() 
        else:
            plt.savefig(saveFigPath)


    def __GetRecResCur(self):
        numchg = len(self.ecpts)
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
    
    def __GetRecYmats(self):
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
            for numchg in range(MaxM+1):
                print(f"Current number of change point is {numchg}.")
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
        Us = []
        for kappac in kappas:
            Us.append(U0 + 2*rAct*np.log(n)**kappac* (np.arange(1, MaxM+2)))
        Us = np.array(Us)
        numchgs = Us.argmin(axis=1)
        time = np.linspace(0, self.paras.T, n)
        
        if self.saveDir is not None:
            RecYmatAllPath = self.saveDir/f"{self.paras.fName}_Rank{self.paras.r}_RecAll.pkl"
            if not RecYmatAllPath.exists():
                self.__GetRecYmats()
                with open(RecYmatAllPath, "wb") as f:
                    pickle.dump(self.RecYmatAll, f)
            else:
                with open(RecYmatAllPath, "rb") as f:
                    self.RecYmatAll = pickle.load(f)
        else:
            self.__GetRecYmats()
            
        MSEs = []
        for i in range(MaxM+1):
            RecYmatCur = self.RecYmatAll[i].EstXmatReal
            MSE = np.mean((RecYmatCur-self.nYmat)**2)
            MSEs.append(MSE)
        self.MSEs = MSEs
        
        self.optNumChg = np.argmin(MSEs)
        
        MSEsKappa = [MSEs[i] for i in numchgs]
        self.optKappa = kappas[np.argmin(MSEsKappa)]
        self.optKappaOptNumChg = numchgs[np.argmin(MSEsKappa)]
    
    def updateEcpts(self, numChg=None):
        assert self.finalRes is not None, "Run main function first!"
        if numChg is None:
            assert self.RecYmatAll is not None, "Run TuningKappa function first!"
            numChg = self.optNumChg
        if numChg == 0:
            self.ecpts = []
        else:
            self.ecpts = self.finalRes.chgMat[numChg-1, :numChg]
        self.__GetRecResCur()
    
