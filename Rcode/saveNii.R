setwd("C:/Users/Dell/Documents/ProjectCode/TVDN/Rcode")
library(magrittr)
source("utils.R")

library(RNifti)
library(oro.nifti)
library(neurobase)

Rlibpath = "../necessary files"
MNI2mm_brain_1 = readnii(file.path(Rlibpath,"MNI152_T1_in_mask_2mm.nii.gz"))
AALmask = readnii(fname=file.path(Rlibpath,"AAL_MNI_2mm.nii"))
region_list = read.table(file=file.path(Rlibpath,"RegionList.txt")) 
AAL <- readNifti(file.path(Rlibpath, 'AAL.nii'))
idx <- 122
filName <- paste0("../results/fMRIHPFs_rankAdap/fMRI", idx, "wU.txt")
#filName <- paste0("./fMRI", idx, "wU.txt")
weightedU <- read.table(filName)
dim(weightedU)

for (em in 1:ncol(weightedU)){
    outPut <- innOut(weightedU[, em],region_list, AAL)
    writeNifti(outPut, file = paste0("./brainsPlot/", 'fMRIM8_', idx, "_", em, '.nii'))
}



# save the nii for canonical networks
fcR = read.csv('../necessary files/AALICA.csv')
fcR = fcR[1:90, ] # 90 x 7
nams <- names(fcR)
for (i in 1:7){
    name <- nams[i]
    outPut <- innOut(fcR[, i], region_list, AAL)
    writeNifti(outPut, file=paste0("./brainsPlot/", name, ".nii"))
}


# The correlation between wU of selected data and fcR 
idxs <- c(5, 15, 58, 70, 149, 230)
for (idx in idxs){
    filName <- paste0("../realdata/midRess/fMRI", idx, "wU.txt")
    saveName <- paste0("../realdata/midRess/fMRI", idx, "corrMax.txt")
    weightedU <- read.table(filName)
    corrs <- corF.fMRI(weightedU, fcR)
    midxs <- apply(corrs, 1, which.max)
    mnames <- nams[midxs]
    write.table(rbind(midxs, mnames), saveName)
}


# The nii file of U with largest correlation with Default network
name <- "Default"
dat <- read.table("../results/max_Default_U.txt")
names(dat) <- NULL
dat <- as.matrix(dat)
dat.corr <- read.table("../results/max_Default_U_corrV.txt")
names(dat.corr) <- NULL
dat.corr <- unlist(dat.corr)


mU <- colMeans(dat)
outPut <- innOut(mU, region_list, AAL)
writeNifti(outPut, file=paste0("./brainsPlot/", "max_meanU", name, ".nii"))

minU <- dat[which.min(dat.corr), ]
maxU <- dat[which.max(dat.corr), ]

outPut <- innOut(minU, region_list, AAL)
writeNifti(outPut, file=paste0("./brainsPlot/", "max_minU", name, ".nii"))
outPut <- innOut(maxU, region_list, AAL)
writeNifti(outPut, file=paste0("./brainsPlot/", "max_maxU", name, ".nii"))


