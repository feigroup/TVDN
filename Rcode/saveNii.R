library(magrittr)
source("utils.R")

library(oro.nifti)
library(RNifti)
library(neurobase)

Rlibpath = "../necessary files"
MNI2mm_brain_1 = readnii(file.path(Rlibpath,"MNI152_T1_in_mask_2mm.nii.gz"))
AALmask = readnii(fname=file.path(Rlibpath,"AAL_MNI_2mm.nii"))
region_list = read.table(file=file.path(Rlibpath,"RegionList.txt")) 
AAL <- readNifti(file.path(Rlibpath, 'AAL.nii'))
idx <- 149
filName <- paste0("../realdata/midRess/fMRI", idx, "wU.txt")
weightedU <- read.table(filName)

for (em in 1:ncol(weightedU)){
    outPut <- innOut(weightedU[, em],region_list, AAL)
    writeNifti(outPut, file = paste0("./brainsPlot/", 'fMRI', idx, "_", em, '.nii'))
}

