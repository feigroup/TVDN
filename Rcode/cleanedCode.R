# 1. decimate (if necessary)
# 2. detrend
# 3. HPF

rm(list=ls())
setwd("C:/Users/Dell/Documents/ProjectCode/TVDN/Rcode")
setwd("C:/Users/JINHU/Documents/ProjectCode/TVDN/Rcode")
source("utils.R")
library(R.matlab)
library(magrittr)
library(ggplot2)
library(reshape2)
# Load dataset
wsize  = 10
chgcor = chgPCA = chgDMD = vector('list')
fMRIs = readMat('../data/fMRI_samples.mat')
datamatrix = vector('list')
nsim <- 243
for(k in 1:243){
     fname <- paste0("../Rcode/to_R/fMRI_", k-1, ".txt")
     datamatrix[[k]] <- read.table(fname)
     #datamatrix[[k]] =  fMRIs$clean.subjects[[k * 4]][1:90, ]
}


for(k in 1:nsim){
   
    print(k)
 
    rank <- 6
    downseq =datamatrix[[k]]
    seqslide = seq(2, ncol(downseq) - wsize, 4) 
    # Get the vectorized results
    corres  = segCorr(downseq, wsize, seqslide)
    PCAres  = segPCA(downseq, wsize, seqslide, rank=rank)
    DMDres = DMD(downseq, wsize, seqslide, rank=rank)
    # Obtain the change points by k-means
    chgcor[[k]] <- chgF(corres, seqslide);chgcor[[k]]
    chgPCA[[k]] <- chgF(PCAres, seqslide)
    chgDMD[[k]] <- chgF(DMDres, seqslide)
    
}


# Obtain the non-vectorized results for PCA and DMD
PCAress = DMDress = vector('list')
for(k in 1:nsim){
   
    print(k)
    rank <- 6 
    downseq =datamatrix[[k]]
    #time =  seq(0, 2, length.out = ncol(downseq))
    seqslide = seq(2, ncol(downseq) - wsize, 4)

    PCAress[[k]]  = segPCAOrg(downseq, wsize, seqslide, rank=rank)
    DMDress[[k]] = DMDOrg(downseq, wsize, seqslide, rank=rank)
    
}



# fMRI 
fcR = read.csv('../necessary files/AALICA.csv')
fcR = fcR[1:90, ] # 90 x 7
segcorrDMDs <- NULL
segcorrPCAs <- NULL

for(k in 1:nsim){
    segcorrDMD <- corF.fMRI(Mod(DMDress[[k]]), fcR) 
    segcorrDMDs <- rbind(segcorrDMDs, segcorrDMD)
    segcorrPCA <- corF.fMRI(PCAress[[k]], fcR)# no Mod?
    segcorrPCAs <- rbind(segcorrPCAs, segcorrPCA)
}

segcorrTVDNs <- abs(read.table("../Rcode/allCorrwU.txt"))
segcorrPCAs <- abs(segcorrPCAs)
segcorrDMDs <- abs(segcorrDMDs)
                                                                                                   
apply(segcorrTVDNs, 1, max)[1:10]
apply(segcorrDMDs, 1, max)[1:10]
apply(segcorrPCAs, 1, max)[1:10]
plotRes <- data.frame(Correlations = c(apply(segcorrTVDNs, 1, max), apply(segcorrDMDs, 1, max), apply(segcorrPCAs, 1, max)), 
                 Names = c(rep('TVDN', nrow(segcorrTVDNs)), rep('DMD', nrow(segcorrDMDs)), rep('PCA', nrow(segcorrPCAs))), 
                 Methods = c(rep(1, nrow(segcorrTVDNs)), rep(2, nrow(segcorrDMDs)), rep(3, nrow(segcorrPCAs))))
p <- ggplot(plotRes, aes(x=Names, y=Correlations, color=Methods)) + 
    geom_violin(trim=T) +  theme(legend.position="none")

p
#ggsave(paste('figures/corr_fMRI_15', '.pdf', sep = '' ))




# Extract the nii file for TVDN method given the abs vaule of weighted U
library(oro.nifti)
library(RNifti)
library(neurobase)

Rlibpath = "../necessary files"
MNI2mm_brain_1 = readnii(file.path(Rlibpath,"MNI152_T1_in_mask_2mm.nii.gz"))
AALmask = readnii(fname=file.path(Rlibpath,"AAL_MNI_2mm.nii"))
region_list = read.table(file=file.path(Rlibpath,"RegionList.txt")) 
AAL <- readNifti(file.path(Rlibpath, 'AAL.nii'))
weightedU <- read.table("../Rcode/weightedUM.txt")

for (em in 1:ncol(weightedU)){
    outPut <- innOut(weightedU[, em],region_list, AAL)
    writeNifti(outPut, file = paste('sp_2_mode_', em, '.nii', sep = ''))
}



## Uncleaned
# Draw the estimated change points 
{
time1 = seq(1, 360, length.out = ncol(datamatrix[[1]]))#meg = 60 fMRI = 360
jpeg(paste('figures/corwsize_', wsize, '_',  sp, '.jpg', sep = ''))
Xmat = datamatrix[[1]]
chgpoint = chgcor[[1]]
plot(Xmat[1, ]~time1, col = 1, type = 'l', lwd = 0.8, ylab = 'Signal', xlab = 'Time'
     , ylim = range(Xmat), xlim = range(time1))
for(i in 1:nrow(Xmat)){
   lines (Xmat[i, ]~time1, col = i, lwd = 0.8, ylab = 'Signal', xlab = 'Time')
}
abline(v = time1[chgpoint], lty = 2, lwd = 3)
dev.off()


jpeg(paste('figures/DMDwsize_', wsize, '_',  sp, '.jpg', sep = ''))
 Xmat = datamatrix[[1]]
chgpoint = chgDMD[[1]]
plot(Xmat[1, ]~time1, col = 1, type = 'l', lwd = 0.8, ylab = 'Signal', xlab = 'Time', ylim = range(Xmat), xlim = range(time1))
for(i in 1:nrow(Xmat)){
   lines (Xmat[i, ]~time1, col = i, lwd = 0.8, ylab = 'Signal', xlab = 'Time')
}
abline(v = time1[chgpoint], lty = 2, lwd = 3)
dev.off()

jpeg(paste('figures/PCAwsize_', wsize, '_',  sp, '.jpg', sep = ''))
Xmat = datamatrix[[1]]
chgpoint = chgPCA[[1]]
plot(Xmat[1, ]~time1, col = 1, type = 'l', lwd = 0.8, ylab = 'Signal', xlab = 'Time', ylim = range(Xmat), xlim = range(time1))
for(i in 1:nrow(Xmat)){
   lines (Xmat[i, ]~time1, col = i, lwd = 0.8, ylab = 'Signal', xlab = 'Time')
}
abline(v = time1[chgpoint], lty = 2, lwd = 3)
dev.off()
}


### MEG correlation
{
load('../necessary files/fcR.rdata')
fcR = fcR[-c(1, 5, 37, 42), ]
fcR[is.nan(fcR)] = 0
segU =Mod(DMDress[[1]])
names = c("Default", "Dorsal_Attention", "Frontoparietal", "Limbic", "Somatomotor", "Ventral_Attention", "Visual")
rank =7
fcR[1:34, ] = apply(fcR[1:34, ], 2, minmax)
fcR[35:68, ] = apply(fcR[35:68, ], 2, minmax)


ncorr = vector('list', ncol(segU))
for(i in 1:ncol(segU)){
    temp = segU[ ,  i]
    # What is the shape of segU, if it is of dim 2, then the following 4 lines seem problematic
    dix = which(duplicated(t(temp)))
    if(length(dix) >0){
        temp = temp[, -dix]
        }
    LH = minmax(temp[1:34])
    RH = minmax(temp[35:68])
   ncorr[[i]] = abs(cor(c(LH, RH), fcR))
}
segcorrDMD = do.call(rbind, ncorr)
# The correlation of TVDN is computed on the U
res = data.frame(Correlations = c(apply(segcorr, 1, max), apply(segcorrDMD, 1, max), apply(segcorrPCA, 1, max)), 
                 Names = c(rep('TVDN', nrow(segcorr)), rep('DMD', nrow(segcorrDMD)), rep('PCA', nrow(segcorrPCA))), 
                 Methods = c(rep(1, nrow(segcorr)), rep(2, nrow(segcorrDMD)), rep(3, nrow(segcorrPCA))))
p <- ggplot(res, aes(x=Names, y=Correlations, color=Methods)) + 
    geom_violin(trim=T)
p
ggsave(paste('figures/corr_MEG2', '.pdf', sep = '' ))
}
