""" 
Some utils functions  to implement TVDN detection method.
All the functions when involving change point index use Python index method (i.e., start from 0 but not 1)
The input and output of the functions use R index method (i.e., start from 1 but not 0)
"""

# import needed packages
import numpy as np
from scipy.signal import detrend
from numpy.linalg import inv, svd
from scipy.stats import multivariate_normal as mnorm
from easydict import EasyDict as edict
from .Rfuns import smooth_spline_R, fourier_reg_R
from .utils import pybwnrd0
# from .utils import in_notebook
#if in_notebook():
#    from tqdm import tqdm_notebook as tqdm
#else:
from tqdm import tqdm


# Function to obtain the Bspline estimate of Xmat and dXmat, d x n
def GetBsplineEst(Ymat, time, lamb=1e-6, nKnots=None):
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
        spres = smooth_spline_R(x=time, y=Ymat[i, :], lamb=lamb, nKnots=nKnots)
        Xmatlist.append(spres["yhat"])
        dXmatlist.append(spres["ydevhat"])
    Xmat = np.array(Xmatlist)
    dXmat = np.array(dXmatlist)
    return dXmat, Xmat

# Function to obtain the fourier basis estimate of Xmat and dXmat, d x n
def GetFourierEst(Ymat, time, nbasis=10):
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
        spres = fourier_reg_R(x=time, y=Ymat[i, :], nbasis=nbasis)
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
    h = pybwnrd0(time)*fct
    d, n = Xmat.shape
    Amat = np.zeros((d, d))
    for idx, s in enumerate(time[::downrate]):
        t_diff = time - s
        kernels = 1/np.sqrt(2*np.pi) * np.exp(-t_diff**2/2/h**2) # normal_pdf(x/h)
        kernelroot = kernels ** (1/2)
        kerdXmat = kernelroot[:, np.newaxis] * (dXmat.T) # n x d
        kerXmat = kernelroot[:, np.newaxis] * (Xmat.T) # n x d
        M = kerXmat.T.dot(kerXmat)/n
        XY = kerdXmat.T.dot(kerXmat)/n # it is Y\trans x X , formula is Amat = Y\trans X (X\trans X)^{-1}
        U, S, VT = svd(M)
        # Num of singular values to keep
        # r = np.argmax(np.cumsum(S)/np.sum(S) > 0.999) + 1 # For simulation
        r = np.argmax(np.cumsum(S)/np.sum(S) >= 0.999) + 1 # For real data
        invM = U[:, :r].dot(np.diag(1/S[:r])).dot(VT[:r, :]) # M is symmetric
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
    # sort the eigvs and eigvecs
    sidx = np.argsort(-np.abs(eigVals))
    eigVals = eigVals[sidx]
    eigVecs = eigVecs[:, sidx]

    eigValsfull = np.concatenate([[np.Inf], eigVals])
    kpidxs = np.where(np.diff(np.abs(eigValsfull))[:r] != 0)[0]
    eigVecsInv = inv(eigVecs)
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
    U, S, VT = svd(SigMat)
    kpidx = np.where(S > (S[0]*1.490116e-8))[0]
    newResd = (U[:, kpidx].T.dot(resd)).T
    meanV = np.zeros(newResd.shape[1])
    Nloglike = - mnorm.logpdf(newResd, mean=meanV, cov=np.diag(S[kpidx])).sum()
    return Nloglike


# Effcient dynamic programming to optimize the MBIC, 
def EGenDy(ndXmat, nXmat, kappa, r, Lmin=None, canpts=None, MaxM=None, Taget="min", is_full=False, Ms=None, showProgress=True):
    """
    Input:
    ndXmat: array, rAct x n. n is length of sequence. 
    nXmat: array, rAct x n. n is length of sequence. 
    kappa: The parameter of penalty
    r: The rank of detection
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

    if (MaxM is None) or (MaxM>M0):
        MaxM = M0 
    if not (Ms is None or len(Ms)==0):
        MaxM = Ms[-1] if Ms[-1]>=MaxM else MaxM
    canpts_full = np.concatenate(([-1], canpts, [n-1]))
    canpts_full2 = canpts_full[1:]
    canpts_full1 = canpts_full[:-1] + 1 # small

    Hmat = np.zeros((M0+1, M0+1)) + decon

    # create a matrix 
    if showProgress:
        for ii in tqdm(range(M0+1), desc="Dynamic Programming"):
            for jj in range(ii, M0+1):
                iidx, jjdx = canpts_full1[ii],  canpts_full2[jj]
                Hmat[ii, jj]  = _nloglk(iidx, jjdx)
    else:
        for ii in range(M0+1):
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
    U = U + 2*r*np.log(n)**kappa* (np.arange(1, MaxM+2))
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
    invEigVecsr = inv(eigVecs)[:r, :]
    eigVecsr = eigVecs[:, :r]
    for i in range(1, n):
        mTerm = np.diag(LamMs[:, i])
        rTerm = invEigVecsr.dot(EstXmat[:, i-1])
        EstXmat[:, i] = eigVecsr.dot(mTerm).dot(rTerm) * tStep + EstXmat[:,i-1]
    if is_full:
        ReDict = edict()
        ReDict.EstXmatReal = detrend(EstXmat.real)
        ReDict.EstXmatRealOrg = EstXmat.real
        ReDict.EstXmatImag = EstXmat.imag # the imag part here should be very small.
        ReDict.LamMs = LamMs
        return ReDict
    else:
        return detrend(EstXmat.real)


# Reconstruct Xmat from results for CV
# i.e., the full Xmat is longer than the Ymatk used in detection
def ReconXmatCV(ecpts, ndXmat, nXmat, kpidxs, eigVecs, Ymat, tStep, r, adjFct, nFull, is_full=False):
    """
    Input: 
        ecpts: Estimated change points, 
        ndXmat: a rAct x n matrix
        nXmat: a rAct x n matrix
        kpidxs: The intermedian output when calculating ndXmat, nXmat
        eigVecs: The matrix of eigen vectors of A matrix, d x d
        Ymat: The matrix to construct, d x n 
        tStep: The time step, which is based on the length of reconstructed Xmat
        r: The rank setted beforehand, in most cases, r=rAct. If we have non-complex singular values, r < rAct
        adjFct: The factor to adjust the change points
        nFull: The full length of the sequence
        if_full: Where outputing full info or not

    Return:
        Estimated Xmat, d x n
    """
    rAct, n = ndXmat.shape
    d, _ = Ymat.shape
    ecptsfull = np.concatenate(([0], ecpts, [n])) - 1
    ecptsfull = ecptsfull.astype(np.int)

    if len(ecpts) == 0:
        adjEcptsFull = np.concatenate(([0], ecpts, [nFull])) - 1
    else:
        adjEcpts = np.round(ecpts * adjFct, 0)
        adjEcptsFull = np.concatenate(([0], adjEcpts, [nFull])) - 1
    # The adjusted locations of change points
    adjEcptsFull = adjEcptsFull.astype(np.int)
    
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
    
    LamMs = np.zeros((r, nFull), dtype=np.complex)
    LamMs[:, 0] = ResegS[0, :]
    for itr in range(1, numchgfull):
        lower = adjEcptsFull[itr-1] + 1
        upper = adjEcptsFull[itr] + 1
        LamMs[:, lower:upper] = ResegS[itr-1, ].reshape(-1, 1)
    
    EstXmat = np.zeros((d, nFull), dtype=np.complex)
    EstXmat[:, 0] = Ymat[:, 0]
    invEigVecsr = inv(eigVecs)[:r, :]
    eigVecsr = eigVecs[:, :r]
    for i in range(1, nFull):
        mTerm = np.diag(LamMs[:, i])
        rTerm = invEigVecsr.dot(EstXmat[:, i-1])
        EstXmat[:, i] = eigVecsr.dot(mTerm).dot(rTerm) * tStep + EstXmat[:,i-1]
    if is_full:
        ReDict = edict()
        ReDict.EstXmatReal = detrend(EstXmat.real)
        ReDict.EstXmatRealOrg = EstXmat.real
        ReDict.EstXmatImag = EstXmat.imag
        ReDict.LamMs = LamMs
        return ReDict
    else:
        return detrend(EstXmat.real)
