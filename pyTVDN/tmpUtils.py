import numpy as np
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