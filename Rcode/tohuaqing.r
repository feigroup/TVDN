segcorr= function(downseq, wsize, seqw){
    tix = 1:length(time)
    
    PCAU = NULL
    for(itr in 1:length(seqw)){
        ix = itr : (itr + wsize -1)
        temp = as.vector((cor(t( downseq[, ix]))))
        #rank = sum(cumsum(temp$d)/sum(temp$d)<= perrank) + 1
        #temp1 = temp$u[, 1:rank, drop = F] %*%  diag((sign(temp$u[1, 1:rank])), rank, rank)
        PCAU = cbind(PCAU,  temp)
 }
    return(PCAU)
}
segPCA = function(downseq, wsize, seqw){
    tix = 1:length(time)
    
    PCAU = NULL
    for(itr in 1:length(seqw)){
        ix = itr : (itr + wsize -1)
        temp = svd(cov(t( downseq[, ix])))
      #  rank = wsize-1#sum(cumsum(temp$d)/sum(temp$d)<= perrank) + 1
        temp1 = as.vector(temp$u[, 1:rank, drop = F] %*%  diag((sign(temp$u[1, 1:rank])), rank, rank))
        PCAU = cbind(PCAU,  temp1)
 }
    return(PCAU)
}

segPCA1 = function(downseq, wsize, seqw){
    tix = 1:length(time)
    
    PCAU = NULL
    for(itr in 1:length(seqw)){
        ix = itr : (itr + wsize -1)
        temp = svd(cov(t( downseq[, ix])))
      #  rank = wsize-1#sum(cumsum(temp$d)/sum(temp$d)<= perrank) + 1
        temp1 = (temp$u[, 1:rank, drop = F] %*%  diag((sign(temp$u[1, 1:rank])), rank, rank))
        PCAU = cbind(PCAU,  temp1)
 }
    return(PCAU)
}

DMD= function(downseq, wsize, seqw){
    tix = 1:length(time)
    PCAU = NULL
    lambda = vector('list')
    for(itr in 1:length(seqw)){
        ix = itr : (itr + wsize -1)
        Xprim = downseq[, ix][, -1]
        X = downseq[, ix][, -length(ix)]
        svdX = svd(X)
       # rank = wsize-1#sum(cumsum(svdX$d)/sum(svdX$d)<= perrank) + 1
        sigma = diag(svdX$d)
        Ahat = diag(svdX$d^(-1/2)) %*% t(svdX$u) %*%  Xprim %*% svdX$v %*% diag(svdX$d^(1/2))
        eigres = eigen(Ahat)
        eigmode = Xprim %*% svdX$v %*% diag(svdX$d^(-1/2)) %*% eigres$vector
        temp1 = eigmode[, 1:rank, drop = F] %*%   diag((sign(Re(eigmode)[1, 1:rank])), rank, rank)
	temp1 = as.vector(Mod(temp1))
        PCAU = cbind(PCAU,  temp1)
        
        
    }
    return(PCAU)
}


DMD1= function(downseq, wsize, seqw){
    tix = 1:length(time)
    PCAU = NULL
    lambda = vector('list')
    for(itr in 1:length(seqw)){
        ix = itr : (itr + wsize -1)
        Xprim = downseq[, ix][, -1]
        X = downseq[, ix][, -length(ix)]
        svdX = svd(X)
       # rank = wsize-1#sum(cumsum(svdX$d)/sum(svdX$d)<= perrank) + 1
        sigma = diag(svdX$d)
        Ahat = diag(svdX$d^(-1/2)) %*% t(svdX$u) %*%  Xprim %*% svdX$v %*% diag(svdX$d^(1/2))
        eigres = eigen(Ahat)
        eigmode = Xprim %*% svdX$v %*% diag(svdX$d^(-1/2)) %*% eigres$vector
        temp1 = eigmode[, 1:rank, drop = F] %*%   diag((sign(Re(eigmode)[1, 1:rank])), rank, rank)
	
        PCAU = cbind(PCAU,  temp1)
        
        
    }
    return(PCAU)
}
#####
sfreq =0.5
perrank = 0.95
slidecorr = vector('list', nsim)
mnum = rep(NA, nsim)
slideU = NULL
seqwsize = seq(8, 32, 2)
wsize  = 10
chgcor = chgPCA = chgDMD = vector('list')
fMRI1 = readMat('subjects_schizophrenia_all_clean.mat')
datamatrix = vector('list')
for(k in 1:243){
     datamatrix[[k]] =  fMRI1$clean.subjects[[k * 4]][1:90, ]
     }
datamatrix = datamatrix[sp]
####below MEG section
sp = 'MEG1'

fMRI1 = readMat('subj1.mat')
isfMRI = FALSE
rate = 1
if(isfMRI == FALSE){
     fMRI = fMRI1$DK.timecourse
    rate = 10
    }
downfMRI = NULL
for(j in 1:nrow(fMRI)){
    downfMRI= rbind(downfMRI, decimate(fMRI[j, ], rate))
}
datamatrix[[1]] = downfMRI
########
for(k in 1:nsim){
   
 
    downseq =datamatrix[[k]]
    rank = 6
    fMRI = downseq
    time =  seq(0, 2, length.out = ncol(downseq))
    seqslide = seq(1, ncol(downseq) - wsize, 4)
   corres  = segcorr(downseq, wsize, seqslide)
   PCAres  = segPCA(downseq, wsize, seqslide)
   DMDres = DMD(downseq, wsize, seqslide)
kmcorres = kmeans(t(corres), 4)   
chgcor[[k]] = seqslide[which(diff(kmcorres$cluster)!=0) + 1]
kmcorres = kmeans(t(PCAres), 4)   
chgPCA[[k]] = seqslide[which(diff(kmcorres$cluster)!=0) + 1]
kmcorres = kmeans(t(DMDres), 4)   
chgDMD[[k]] = seqslide[which(diff(kmcorres$cluster)!=0) + 1]
    
}

PCAres = DMDres = vector('list')
for(k in 1:nsim){
   
 
    downseq =datamatrix[[k]]
    rank = 8
    fMRI = downseq
    time =  seq(0, 2, length.out = ncol(downseq))
    seqslide = seq(1, ncol(downseq) - wsize, 4)

   PCAres[[k]]  = segPCA1(downseq, wsize, seqslide)
   DMDres[[k]] = DMD1(downseq, wsize, seqslide)
    
}





time1 = seq(1, 60, length.out = ncol(datamatrix[[1]]))#meg = 60 fMRI = 360
jpeg(paste('figures/corwsize_', wsize, '_',  sp, '.jpg', sep = ''))
Xmat = datamatrix[[1]]
chgpoint = chgcor[[1]]
plot(Xmat[1, ]~time1, col = 1, type = 'l', lwd = 0.8, ylab = 'Signal', xlab = 'Time', ylim = range(Xmat), xlim = range(time1))
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


 




###MEG correlation
load('fcR.rdata')
fcR = fcR[-c(1, 5, 37, 42), ]
fcR = fcR
segU =Mod(DMDres[[1]])
names = c("Default", "Dorsal_Attention", "Frontoparietal", "Limbic", "Somatomotor", "Ventral_Attention", "Visual")
rank =7
fcR[is.nan(fcR)] = 0
minmax = function(x){
    return((x - min(x))/(max(x) - min(x)))
}
fcR[1:34, ] = apply(fcR[1:34, ], 2, minmax)
fcR[35:68, ] = apply(fcR[35:68, ], 2, minmax)


ncorr = vector('list', ncol(segU))
for(i in 1:ncol(segU)){
    temp = segU[ ,  i]
    dix = which(duplicated(t(temp)))
    if(length(dix) >0){
        temp = temp[, -dix]
        }
    LH = minmax(temp[1:34])
    RH = minmax(temp[35:68])
   ncorr[[i]] = abs(cor(c(LH, RH), fcR))
}
segcorrDMD = do.call(rbind, ncorr)
res = data.frame(Correlations = c(apply(segcorr, 1, max), apply(segcorrDMD, 1, max), apply(segcorrPCA, 1, max)), Names = c(rep('TVDN', nrow(segcorr)), rep('DMD', nrow(segcorrDMD)), rep('PCA', nrow(segcorrPCA))), Methods = c(rep(1, nrow(segcorr)), rep(2, nrow(segcorrDMD)), rep(3, nrow(segcorrPCA))))
library(ggplot2)
p <- ggplot(res, aes(x=Names, y=Correlations, color=Methods)) + 
    geom_violin(trim=T)
p
ggsave(paste('figures/corr_MEG2', '.pdf', sep = '' ))



##fMRI 
fcR = read.csv('AALICA.csv')
fcR = fcR[1:90, ]
library(ggplot2)
library(reshape2)
segcorrDMD = NULL
for(k in 1){
segU =Mod(DMDres)# sqrt(Re(ResegU)^2 + Im(ResegU)^2)

#fcR[is.nan(fcR)] = 0
minmax = function(x){
    return((x - min(x))/(max(x) - min(x)))
}
fcR = apply(fcR, 2, minmax)


ncorr = vector('list', ncol(segU))
for(i in 1:ncol(segU)){
    temp = (segU[ ,  i])
   ncorr[[i]] = abs(cor(temp, fcR))
}
segcorrDMD= rbind(segcorrDMD, do.call(rbind, ncorr))
}
                                                                                                   
res = data.frame(Correlations = c(apply(segcorr, 1, max), apply(segcorrDMD, 1, max), apply(segcorrPCA, 1, max)), Names = c(rep('TVDN', nrow(segcorr)), rep('DMD', nrow(segcorrDMD)), rep('PCA', nrow(segcorrPCA))), Methods = c(rep(1, nrow(segcorr)), rep(2, nrow(segcorrDMD)), rep(3, nrow(segcorrPCA))))
library(ggplot2)
p <- ggplot(res, aes(x=Names, y=Correlations, color=Methods)) + 
    geom_violin(trim=T)
p
#ggsave(paste('figures/corr_fMRI_15', '.pdf', sep = '' ))


library(oro.nifti)
library(RNifti)
library(neurobase)
load('allfMRIresult_2.Rdata')
Rlibpath = "./"
MNI2mm_brain_1 = readnii(file.path(Rlibpath,"MNI152_T1_in_mask_2mm.nii.gz"))
AALmask = readnii(fname=file.path(Rlibpath,"AAL_MNI_2mm.nii"))
region_list = read.table(file=file.path(Rlibpath,"RegionList.txt"))
temp = readNifti('/Users/feijiang/nilearn_data/aal_SPM12/aal/atlas/AAL.nii')
temp1 = temp
temp[] = NA
ix = 1:numberofsegment
for(em in ix){
tempmode = minmax(Mod(mestU[[1]][, em])) ##mestU[[1]][, em] is the weighted U at em segment. 
for(i in 1:90){
    ix  = which(temp1 == region_list[i, 2])
    temp[ix] = tempmode[i]
}
writeNifti(temp, file = paste('sp_2_mode_', em, '.nii', sep = ''))
}


