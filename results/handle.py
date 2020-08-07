import pickle
import numpy as np
from easydict import EasyDict as edict
from pathlib import Path 
from pprint import pprint

rootDir = Path(".")
fils = list(rootDir.glob("*Rank10.pkl"))

for fil in fils:
    with open(fil, "rb") as f:
        data = pickle.load(f)
    data.Ymat = None
    data.nYmat = data.pop("PostMEG")
    data.paras.downRate = data.paras.downrate
    data.paras.decimateRate = data.paras.rate
    data.paras.fct = 0.5
    data.paras.freq = 30
    data.paras.plotfct = 30
    data.paras.is_detrend = False
    data.paras.T = 2
    data.paras.fName = fil.stem.split("_")[0]
    data.paras.pop("downrate")
    data.ptime = np.linspace(0, data.paras.T, 3600)*30
    
    with open(fil, "wb") as f:
        pickle.dump(data, f)


