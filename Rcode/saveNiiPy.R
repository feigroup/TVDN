setwd("~/MyResearch/TVDN/Rcode")
library(magrittr)
source("utils.R")

library(RNifti)
library(oro.nifti)
library(neurobase)

Rlibpath = "../necessary files"
region_list = read.table(file=file.path(Rlibpath,"RegionList.txt")) 
AAL <- readNifti(file.path(Rlibpath, 'AAL.nii'))
weightedU <- read.table(filName)

for (em in 1:ncol(weightedU)){
    outPut <- innOut(weightedU[, em],region_list, AAL)
    writeNifti(outPut, file = paste0(prefixOut, idx, "_", em, '.nii'))
}



