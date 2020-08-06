""" 
This is the python script to implement change point detection
All the functions when involving change point index use Python index method (i.e., start from 0 but not 1)
The input and output of the functions also use R index method (i.e., start from 1 not but 0)
"""

# import needed packages
import numpy as np
from pprint import pprint
from pathlib import Path
from scipy.stats import multivariate_normal as mnorm
from scipy.signal import decimate, detrend
# for general inverse of matrix
from numpy.linalg import pinv, svd, inv
import seaborn as sns
from tqdm import  tqdm
#from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
# to use functions in R languge
import rpy2.robjects as robj
import pickle
from easydict import EasyDict as edict


# Two functions  to use spline functions in R language
def bw_nrd0_R(time, fct=1):
    bw_nrd0 = robj.r["bw.nrd0"]
    time_r = robj.FloatVector(time)
    return np.array(bw_nrd0(time_r))[0]*fct

def smooth_spline_R(x, y, lamb):
    smooth_spline_f = robj.r["smooth.spline"]
    x_r = robj.FloatVector(x)
    y_r = robj.FloatVector(y)
    args = {"x": x_r, "y": y_r, "lambda": lamb}
    spline = smooth_spline_f(**args)
    ysp = np.array(robj.r['predict'](spline, deriv=0).rx2('y'))
    ysp_dev1 = np.array(robj.r['predict'](spline, deriv=1).rx2('y'))
    return {"yhat": ysp, "ydevhat": ysp_dev1}

# The decimate function in R signal packages
def decimate_R(seq, q):
    robj.r("library(signal)")
    decimate_R_f = robj.r("decimate")
    seq_R = robj.FloatVector(seq)
    return np.array(decimate_R_f(seq_R, q))

# Function to obtain the Bspline estimate of Xmat and dXmat, d x n
def GetBsplineEst(Ymat, time, lamb=1e-6):
    """
    Input:
        Ymat: The observed data matrix, d x n
        time: A list of time points of length n
    return:
        The estimated Xmat and dXmat, both are d x n
    """
    d, n = Ymat.shape
    Xmatlist = []
    dXmatlist = []
    for i in range(d):
        spres = smooth_spline_R(x=time, y=Ymat[i, :], lamb=lamb)
        Xmatlist.append(spres["yhat"])
        dXmatlist.append(spres["ydevhat"])
    Xmat = np.array(Xmatlist)
    dXmat = np.array(dXmatlist)
    return dXmat, Xmat

# Function to obtain the sum of Ai matrix
def GetAmat(dXmat, Xmat, time, downrate=1, fct=1):
    """
    Input: 
        dXmat: The first derivative of Xmat, d x n matrix
        Xmat: Xmat, d x n matrix
        time: A list of time points with length n
        downrate: The downrate factor, determine how many Ai matrix to be summed
    Return:
        A d x d matrix, it is sum of n/downrate  Ai matrix
    """
    h = bw_nrd0_R(time, fct=fct)
    d, n = Xmat.shape
    Amat = np.zeros((d, d))
    for idx, s in enumerate(time[::downrate]):
        t_diff = time - s
        kernels = 1/np.sqrt(2*np.pi) * np.exp(-t_diff**2/2/h**2) # normal_pdf(x/h)
        kernelroot = kernels ** (1/2)
        kerdXmat = kernelroot[:, np.newaxis] * (dXmat.T) # n x d
        kerXmat = kernelroot[:, np.newaxis] * (Xmat.T) # n x d
        M = kerXmat.T.dot(kerXmat)/n
        XY = kerdXmat.T.dot(kerXmat)/n
        U, S, VT = np.linalg.svd(M)
        # Num of singular values to keep
        # r = np.argmax(np.cumsum(S)/np.sum(S) > 0.999) + 1 # For simulation
        r = np.argmax(np.cumsum(S)/np.sum(S) >= 0.999) + 1 # For real data
        invM = U[:, :r].dot(np.diag(1/S[:r])).dot(VT[:r, :])
        Amat = Amat + XY.dot(invM)
    return Amat

# Function to obtain the New Xmat and dXmat to do change point detection
def GetNewEst(dXmat, Xmat, Amat, r, is_full=False):
    """
    Input: 
        dXmat, Estimated first derivative of X(t), d x n
        Xmat, Estimated of X(t), d x n
        Amat: The A matrix to to eigendecomposition, d x d
        r: The number of eigen values to keep
        is_full: Where return full outputs or not
    Return: 
        nXmat, ndXmat, rAct x n 
    """
    _, n = dXmat.shape
    eigVals, eigVecs = np.linalg.eig(Amat)
    eigValsfull = np.concatenate([[np.Inf], eigVals])
    kpidxs = np.where(np.diff(np.abs(eigValsfull))[:r] != 0)[0]
    eigVecsInv = np.linalg.inv(eigVecs)
    tXmat = eigVecsInv[kpidxs, :].dot(Xmat)
    tdXmat = eigVecsInv[kpidxs, :].dot(dXmat)
    nrow, _ = tXmat.shape
    nXmat = np.zeros((2*nrow, n))
    ndXmat = np.zeros((2*nrow, n))
    for j in range(nrow):
        nXmat[2*j, :] = tXmat[j, :].real
        nXmat[2*j+1, :] = tXmat[j, :].imag
        ndXmat[2*j, :] = tdXmat[j, :].real
        ndXmat[2*j+1, :] = tdXmat[j, :].imag
    if is_full:
        return edict({"ndXmat":ndXmat, "nXmat":nXmat, "kpidxs":kpidxs, "eigVecs":eigVecs, "eigVals":eigVals, "r": r})
    else:
        return ndXmat, nXmat



# Function to calculate the  Gamma_k matrix during dynamic programming
def GetGammak(pndXmat, pnXmat):
    """
    Input: 
        pndXmat: part of ndXmat, rAct x (j-i)
        pnXmat: part of nXmat, rAct x (j-i)
    Return:
        Gamma matrix, rAct x rAct
    """
    rAct, _ = pndXmat.shape
    GamMat = np.zeros((rAct, rAct))
    for i in range(int(rAct/2)):
        tY = pndXmat[(2*i):(2*i+2) , :]
        tX = pnXmat[(2*i):(2*i+2) , :]
        corY = tY.dot(tX.T) # 2 x 2
        corX = np.trace(tX.dot(tX.T))
        a = np.trace(corY)/corX
        b = (corY[1, 0] - corY[0, 1])/corX
        GamMat[2*i, 2*i] = a
        GamMat[2*i+1, 2*i+1] = a
        GamMat[2*i, 2*i+1] = -b
        GamMat[2*i+1, 2*i] = b
    return GamMat

# Function to calculate the negative log likelihood during dynamic programming
def GetNlogk(pndXmat, pnXmat, Gamk):
    """
    Input: 
        pndXmat: part of ndXmat, rAct x (j-i)
        pnXmat: part of nXmat, rAct x (j-i)
        Gamk: Gamma matrix, rAct x rAct
    Return:
        The Negative log likelihood
    """
    _, nj = pndXmat.shape
    resd = pndXmat - Gamk.dot(pnXmat)
    SigMat = resd.dot(resd.T)/nj
    U, S, VT = np.linalg.svd(SigMat)
    kpidx = np.where(S > (S[0]*1.490116e-8))[0]
    newResd = (U[:, kpidx].T.dot(resd)).T
    meanV = np.zeros(newResd.shape[1])
    Nloglike = - mnorm.logpdf(newResd, mean=meanV, cov=np.diag(S[kpidx])).sum()
    return Nloglike


# Effcient dynamic programming to optimize the MBIC, 
def EGenDy(ndXmat, nXmat, kappa, Lmin=None, canpts=None, MaxM=None, Taget="min", is_full=False, Ms=None):
    """
    Input:
    ndXmat: array, rAct x n. n is length of sequence. 
    nXmat: array, rAct x n. n is length of sequence. 
    kappa: The parameter of penalty
    Lmin: The minimal length between 2 change points
    canpts: candidate point set. list or array,  index should be from 1
    MaxM: int, maximal number of change point 
    Ms: the list containing prespecified number of change points.
       When Ms=None, it means using MBIC to determine the number of change points
    is_full: Where return full outputs or not

    Return:
        change point set with index starting from 1
        chgMat: A matrix containing the change points for each number of change point
        U0: MBIC without penalty
        U:  MBIC  for each number of change point
    """
    def _nloglk(i, j):
        length = j - i + 1
        pndXmat = ndXmat[:, i:(j+1)]
        pnXmat = nXmat[:, i:(j+1)]
        Gamk = GetGammak(pndXmat, pnXmat)
        if length >= Lmin:
            return GetNlogk(pndXmat, pnXmat, Gamk)
        else:
            return decon 

    rAct, n = nXmat.shape
    if Lmin is None:
        Lmin = rAct
        
    if Taget == "min":
        tagf = np.min
        tagargf = np.argmin
        decon = np.inf
    else:
        tagf = np.max
        tagargf = np.argmax
        decon = -np.inf

    if Ms is not None:
        Ms = sorted(Ms)
    if canpts is None:
        canpts = np.arange(n-1)
    else:
        canpts = np.array(canpts) - 1
    M0 = len(canpts) # number of change point in candidate point set

    if MaxM is None:
        MaxM = M0 
    if not (Ms is None or len(Ms)==0):
        MaxM = Ms[-1] if Ms[-1]>=MaxM else MaxM
    canpts_full = np.concatenate(([-1], canpts, [n-1]))
    canpts_full2 = canpts_full[1:]
    canpts_full1 = canpts_full[:-1] + 1 # small

    Hmat = np.zeros((M0+1, M0+1)) + decon

    # create a matrix 
    for ii in tqdm(range(M0+1), desc="Main Loop"):
        for jj in range(ii, M0+1):
            iidx, jjdx = canpts_full1[ii],  canpts_full2[jj]
            Hmat[ii, jj]  = _nloglk(iidx, jjdx)

    # vector contains results for each number of change point
    U = np.zeros(MaxM+1) 
    U[0] = Hmat[0, -1]
    D = Hmat[:, -1]
    # contain the location of candidate points  (in python idx)
    Pos = np.zeros((M0+1, MaxM)) + decon
    Pos[M0, :] = np.ones(MaxM) * M0
    tau_mat = np.zeros((MaxM, MaxM)) + decon
    for k in range(MaxM):
        for j in range(M0): # n = M0 + 1
            dist = Hmat[j, j:-1] + D[(j+1):]
            #print(dist)
            D[j] = np.min(dist)
            Pos[j, 0] = np.argmin(dist) + j + 1
            if k > 0:
                Pos[j, 1:(k+1)] = Pos[int(Pos[j, 0]), 0:k]
        U[k+1] = D[0]
        tau_mat[k, 0:(k+1)] = Pos[0, 0:(k+1)] - 1
    U0 = U 
    U = U + 2*rAct*np.log(n)**kappa* (np.arange(1, MaxM+2))
    chgMat = np.zeros(tau_mat.shape) + np.inf
    for iii in range(chgMat.shape[0]):
        idx = tau_mat[iii,: ]
        idx = np.array(idx[idx<np.inf], dtype=np.int)
        chgMat[iii, :(iii+1)]= np.array(canpts)[idx] + 1 
    
    mbic_numchg = np.argmin(U[:(MaxM+1)])
    if mbic_numchg == 0:
        mbic_ecpts = np.array([])
    else:
        idx = tau_mat[int(mbic_numchg-1),: ]
        idx = np.array(idx[idx<np.inf], dtype=np.int)
        mbic_ecpts = np.array(canpts)[idx] + 1
        
    if Ms is None or len(Ms)==0:
        if not is_full:
            return edict({"U":U, "mbic_ecpts": mbic_ecpts})
        else:
            return edict({"U":U, "mbic_ecpts": mbic_ecpts, "chgMat": chgMat, "U0":U0})
    else:
        ecptss = []
        for numchg in Ms:
            if numchg == 0:
                ecpts = np.array([])
            else:
                idx = tau_mat[int(numchg-1),: ]
                idx = np.array(idx[idx<np.inf], dtype=np.int)
                ecpts = np.array(canpts)[idx] + 1
            ecptss.append(ecpts)
        if not is_full:
            return edict({"U":U, "ecptss": ecptss, "mbic_ecpts": mbic_ecpts})
        else:
            return edict({"U":U, "ecptss": ecptss, "mbic_ecpts": mbic_ecpts, "chgMat": chgMat, "U0":U0})


# Reconstruct Xmat from results
def ReconXmat(ecpts, ndXmat, nXmat, kpidxs, eigVecs, Ymat, tStep, r, is_full=False):
    """
    Input: 
        ecpts: Estimated change points, 
        ndXmat: a rAct x n matrix
        nXmat: a rAct x n matrix
        kpidxs: The intermedian output when calculating ndXmat, nXmat
        eigVecs: The matrix of eigen vectors of A matrix, d x d
        Ymat: The matrix to construct, d x n 
        tStep: The time step
        r: The rank setted beforehand, in most cases, r=rAct. If we have non-complex singular values, r < rAct
        if_full: Where outputing full info or not

    Return:
        Estimated Xmat, d x n
    """
    rAct, n = ndXmat.shape
    d, _ = Ymat.shape
    ecptsfull = np.concatenate(([0], ecpts, [n])) - 1
    ecptsfull = ecptsfull.astype(np.int)
    numchgfull = len(ecptsfull)

    ResegS = np.zeros((numchgfull-1, r), dtype=np.complex)
    for  itr in range(numchgfull-1):
        lower = ecptsfull[itr] + 1
        upper = ecptsfull[itr+1] + 1
        Ycur = ndXmat[:, lower:upper]
        Xcur = nXmat[:, lower:upper]
        lams = np.zeros(r, dtype=np.complex) + np.inf
        for j in range(int(rAct/2)):
            tY = Ycur[(2*j):(2*j+2), :]
            tX = Xcur[(2*j):(2*j+2), :]
            corY = tY.dot(tX.T)
            corX = np.trace(tX.dot(tX.T))
            a = np.trace(corY)/corX
            b = (corY[1, 0] - corY[0, 1])/corX
            lams[kpidxs[j]] = a + b*1j
        tmpIdx = np.where(lams==np.inf)[0]
        lams[tmpIdx] = np.conjugate(lams[tmpIdx-1])
        ResegS[itr, :] = lams
    
    LamMs = np.zeros((r, n), dtype=np.complex)
    LamMs[:, 0] = ResegS[0, :]
    for itr in range(1, numchgfull):
        lower = ecptsfull[itr-1] + 1
        upper = ecptsfull[itr] + 1
        LamMs[:, lower:upper] = ResegS[itr-1, ].reshape(-1, 1)
    
    EstXmat = np.zeros((d, n), dtype=np.complex)
    EstXmat[:, 0] = Ymat[:, 0]
    for i in range(1, n):
        mTerm = np.diag(LamMs[:, i])
        rTerm = np.linalg.inv(eigVecs)[:r, :].dot(EstXmat[:, i-1])
        EstXmat[:, i] = eigVecs[:, :r].dot(mTerm).dot(rTerm) * tStep + EstXmat[:,i-1]
    if is_full:
        ReDict = edict()
        ReDict.EstXmatReal = EstXmat.real
        ReDict.LamMs = LamMs
        return ReDict
    else:
        return EstXmat.real


class TVDNDetect:
    def __init__(self, Ymat, dataType=None, saveDir=None, **paras):
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
        keys = list(self.paras.keys())
        for key in paras.keys():
            self.paras[key] = paras[key]
        print("The parameters for detection are:")
        pprint(self.paras)
        
        if saveDir is not None:
            self.saveDir = Path(saveDir)
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
                
        else:
            if self.midRes is None:
                self.GetNewData()
            self.finalRes = EGenDy(self.ndXmat, self.nXmat, kappa=kappa, Lmin=Lmin, MaxM=MaxM, is_full=True)
            self.ecpts = self.finalRes.mbic_ecpts
            
    def PlotECPTs(self):
        assert self.finalRes is not None, "Run main function first!"
        d, n = self.nYmat.shape
        ajfct = n/(self.paras.plotfct*self.paras.T)
        ptime = np.linspace(0, self.paras.T, n) * self.paras.plotfct
        plt.figure(figsize=[10, 5])
        for i in range(d):
            plt.plot(ptime, self.nYmat[i, :], "-")
        for ecpt in self.ecpts:
            plt.axvline(ecpt/ajfct, color="black", linestyle="-.")
        plt.show() 
        
    
    def PlotRecCurve(self, idxs):
        assert self.finalRes is not None, "Run main function first!"
        assert self.RecYmatAll is not None, "Run TuningKappa function first!"
        numChgCur = len(self.ecpts)
        RecYmatCur = self.RecYmatAll[numChgCur].EstXmatReal
        d, n = RecYmatCur.shape
        assert d>=np.max(idxs) & np.min(idxs)>=0, "Wrong index!"
        
        ajfct = n/(self.paras.plotfct*self.paras.T)
        ptime = np.linspace(0, self.paras.T, n) * self.paras.plotfct
        
        numSubPlot = len(idxs)
        numRow = ((numSubPlot-1) // 3) + 1
        
        plt.figure(figsize=[15, 5*numRow])

        for i, idx, in enumerate(idxs):
            plt.subplot(numRow, 3, i+1)
            plt.plot(ptime, self.nYmat[idx, :], label="Observed")
            plt.plot(ptime, RecYmatCur[idx, :], label="Estimated")
            plt.legend()
        plt.show()
    
    def PlotEigenCurve(self):
        assert self.finalRes is not None, "Run main function first!"
        assert self.RecYmatAll is not None, "Run TuningKappa function first!"
        freq = self.paras.freq
        numChgCur = len(self.ecpts)
        LamMs = self.RecYmatAll[numChgCur].LamMs
        ReLamMs = LamMs.real*freq/30 
        ImLamMs = LamMs.imag*freq /(30*2*np.pi)
        cols = sns.color_palette("Paired", ReLamMs.shape[0])
        _, n = LamMs.shape
        ajfct = n/(self.paras.plotfct*self.paras.T)
        ptime = np.linspace(0, self.paras.T, n) * self.paras.plotfct
        
        plt.figure(figsize=[20,10])
        plt.subplot(121)
        for i in range(ReLamMs.shape[0]):
            plt.plot(ptime, ReLamMs[i, :], label=f"Lam {i+1}", 
                     color=cols[i], linewidth=2)
        plt.ylabel("change of growth/decay constant")
        plt.xlabel("time")
        _ = plt.legend()
        
        plt.subplot(122)
        for i in range(ReLamMs.shape[0]):
            plt.plot(ptime, ImLamMs[i, :], label=f"Lam {i+1}", 
                     color=cols[i], linewidth=2)
        plt.ylabel("change of growth/decay constant")
        plt.xlabel("time")
        _ = plt.legend()
        plt.show()

    
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
        assert self.RecYmatAll is not None, "Run TuningKappa function first!"
        if numChg is None:
            numChg = self.optNumChg
        if numChg == 0:
            self.ecpts = []
        else:
            self.ecpts = self.finalRes.chgMat[numChg-1, :numChg]
    
