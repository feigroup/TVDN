# Three functions  call from  R language

import rpy2.robjects as robj
import numpy as np
from pathlib import Path

def bw_nrd0_R(time, fct=1):
    bw_nrd0 = robj.r["bw.nrd0"]
    time_r = robj.FloatVector(time)
    return np.array(bw_nrd0(time_r))[0]*fct

def smooth_spline_R(x, y, lamb, nKnots=None):
    smooth_spline_f = robj.r["smooth.spline"]
    x_r = robj.FloatVector(x)
    y_r = robj.FloatVector(y)
    if nKnots is None:
        args = {"x": x_r, "y": y_r, "lambda": lamb}
    else:
        args = {"x": x_r, "y": y_r, "lambda": lamb, "nknots":nKnots}
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


def fourier_reg_R(x, y, nbasis=10):
    filePath = Path(__file__).parent
    robj.r.source(str(filePath/"Rfuns.R"))
    x_r = robj.FloatVector(x)
    y_r = robj.FloatVector(y)
    res = robj.r.fourier_reg(x_r, y_r, nbasis)
    yhat = np.array(res.rx2("yhat"))
    ydevhat = np.array(res.rx2("dyhat"))
    return {"yhat":yhat, "ydevhat":ydevhat}
