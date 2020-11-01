import numpy as np
import rpy2.robjects as robj
from easydict import EasyDict as edict

timeLims = edict()
timeLims.st02 = [35, 95]
timeLims.st03 = [20, 80]

# load the gt for MEG--Eye data
def txt2Time(txtF):   
    with open(txtF, "r") as f:
        data = f.readlines() 
    data = data[1:]
    data = [i.strip().split("(") for i in data]
    data = [float(i[0]) for i in data if len(i)>1]
    return data

# Time to change points
def time2pts(ts, lims, Up=7200):
    ts = np.array(ts)
    timeC = 60
    ts = ts[ts>=lims[0]]
    ts = ts[ts<=lims[1]]
    ts = ts - lims[0]
    cpts = ts*Up/timeC
    cpts = cpts.astype(np.int)
    
    res = edict()
    res.ts = ts
    res.cpts = cpts
    return res


def py2Rmat(Mat):
    Mat = np.array(Mat)
    nrow, _ = Mat.shape
    rVec = robj.FloatVector(Mat.ravel())
    rMat = robj.r.matrix(rVec, nrow=nrow, byrow=True)
    return rMat

def py2Rvec(vec):
    vec = np.array(vec)
    vec = vec.astype(np.float)
    return robj.FloatVector(vec)


# Obtain the weighted U from the detection obj
def obtainAbswU(DetObj):
    eigVecs = DetObj.midRes.eigVecs[:, :DetObj.paras.r]
    kpidxs = np.concatenate([[0], DetObj.ecpts]).astype(np.int)
    eigVals = DetObj.RecResCur.LamMs[:, kpidxs]
    wU = eigVecs.dot(eigVals)
    return np.abs(wU)