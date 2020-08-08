# The Time-varying Dynamic Network Model for Extracting the Dyanmic Resting State Functional Connectivity


This is repo is to implement a time-varying dynamic network (TVDN) method to 
extract the resting state functional connectivity (RSFC) from both 
functional magnetic resonance (fMRI) and the magnetoencephalography (MEG) imaging.


## Installation

- Enviroment:
  - Python 3
  - R
 
```
git clone https://github.com/JINhuaqing/TVDN.git
```
For dependencies in Python, run

```bash

pip install -r requirements.txt
```

`signal` pakcage in R is also needed

```R

install.packages("signal")
```

##  Examples

You may find demo for MEG and fMRI data in demo folder.

```python

# import the class for detection
from pyTVDN import TVDNDetect
from pathlib import Path
import numpy as np


# Construnct the detection object
Detection = TVDNDetect(Ymat=Ymat, saveDir="../results", dataType="MEG", fName="subj2", r=8, kappa=2.95, freq=60)
# Ymat, d x n maitrx, where d is the number of sequences
# saveDir, the path to save the results. If not specified, the results will not be saved
# dataType, "MEG" or "fMRI". Different dataTypes have different default parameters. You may leave it blank
# All other parameters have default values, but you can still specify here.
# For the meaning of the other parameters, you can refer to the source code


# Run detection
Detection()

# You can tuning the kappa, the parameters for MBIC penalty term
kappas = np.linspace(2.5, 3, 100)
Detection.TuningKappa(kappas)

# You can specify the number of change points you want via provide the argument `numChg`, then the `UpdateEcpts` will update the current estimated change point set accordingly
Detection.UpdateEcpts(numChg=12)
# If You don't specify the number of change points you want, then the `UpdateEcpts` will update the current estimated change point set based on optimal kappa values
Detection.UpdateEcpts()

# Plot the detection results
Detection.PlotEcpts(saveFigPath="detectionResults.jpg")
# save figure if you specify the `saveFigPath`

# Plot the Reconstruncted Ymat.
Detection.PlotRecCurve(idxs=[43, 45, 59], saveFigPath="recCurve.jpg")
# idxs: The indices of sequence to plot

# Plot the eigen values curve
Detection.PlotEigenCurve()
# save figure if you specify the `saveFigPath`

```
