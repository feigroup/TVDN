# The Time-varying Dynamic Network Model for Extracting the Dyanmic Resting State Functional Connectivity


This is repo is to implement a time-varying dynamic network (TVDN) method to 
extract the resting state functional connectivity (RSFC) from both 
functional magnetic resonance (fMRI) and the magnetoencephalography (MEG) imaging.

It contains the simulation code.

**If you encounter any problems when using the repo, be free to contact us (<jinhuaqing@connect.hku.hk>)**


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

You may find demos for MEG and fMRI data in **demo** folder.

```python

# import the class for detection
from pyTVDN import TVDNDetect
from pathlib import Path
import numpy as np


# Construnct the detection object
Detection = TVDNDetect(Ymat=Ymat, saveDir="../results", dataType="MEG", fName="subj2", r=8, kappa=2.95, freq=60)
# Ymat, d x n maitrx, where d is the number of sequences
# r: the number of rank used for detection
#    if r is decimal, the rank is the number of eigen values such that account for 100r% of the variance.
#    if r is None, r=0.8
# saveDir, the path to save the results. If not specified, the results will not be saved
# dataType, "MEG" or "fMRI". Different dataTypes have different default parameters. You may leave it blank
# All other parameters have default values, but you can still specify here.
# For the meaning of the other parameters, you can refer to the source code


# When n is large, the detection would take a while
# To reduce the computaion burden, we provide a screening step 
# to obain the candidate point set
# Screening is optional
Detection.Screening(wh=10)
# wh: the screening window size

# Run detection
Detection()

# You can tune kappa, the parameters for MBIC penalty term kappa by the reconstructed errors
# However, it does not always work very well
kappas = np.linspace(2.5, 3, 100)
Detection.TuningKappa(kappas)

# You can specify the number of change points you want via provide the argument `numChg`, then the `UpdateEcpts` will update the current estimated change point set accordingly
Detection.UpdateEcpts(numChg=12)
# If You don't specify the number of change points you want, then the `UpdateEcpts` will update the current estimated change point set based on optimal kappa values by TuningKappas function
Detection.UpdateEcpts()

# Plot the detection results
Detection.PlotEcpts(saveFigPath="detectionResults.jpg")
# save figure if you specify the `saveFigPath`
# You can specify the GT parameter to draw the ground truth for comparison

# Plot the reconstruncted Ymat.
Detection.PlotRecCurve(idxs=[43, 45, 59], saveFigPath="recCurve.jpg")
# idxs: The indices of sequences to plot

# Plot the eigen values curve
Detection.PlotEigenCurve()
# save figure if you specify the `saveFigPath`

# print the results here
print(Detection)

```


