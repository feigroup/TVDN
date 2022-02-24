
pen = 1.65
sp = sp###10 pen = 1.65 5 pen = 1.45 1 pen = 1.6 15 pen = 1.85 20 pen = 1.8  MEG1 pen = 2.65 MEG2 pen = 2.95
datamatrix = datamatrix1[sp]
#following block used for MEG
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
pen = 2.95
sp = 'MEG1' ###10 pen = 1.6 5 pen = 1.45 1 pen = 1.6 15 pen = 1.85 20 pen = 1.8  MEG pen = 2.65
######
truematrix = datamatrix
freq = 0.5 #MEG = 60
nsim = length(datamatrix)
candlist0 = NULL
##study cross
crank = seq(4, 12, 2) ##crank for MEG seq(4, 8, 2) for fMRI seq(4, 12, 2)
smX = resmX =resmse =  uniqsix = ldupsix = segEigenValue = mestU = mchgpoint = mchgres = mY = mX = vector('list')

listpen = seq(1.65, 2, 0.1)
resmse = array(NA, c(length(crank), length(listpen), nsim))
for(spen in 1:length(listpen)){
      pen = listpen[spen]
for(l in 1:length(crank) ){	
      	 masterr = crank[l]
	 source('estsimufun1.r')
	 BICq = matrix(NA, nsim, length(seqnum))
	 for(num in 1:length(seqnum)){
	 BICq[, num]= mserror1[, num] +   (log(length(time))) ^pen*( masterr * 2 ) * (num)		
	 }

	 smix = apply(BICq, 1, which.min)
	 cmX = array(NA, dim(lmestX[[1]]))
	 mmse = matrix(NA, nsim, 1)
	 for(i in 1:nsim){
     	 cmX[,  ,i] = lmestX[[smix[i]]][, , i]
     	 mmse[i] = c(mserror[i, smix[i]])

	 }

	 print(mmse)	 
	 resmse[l, spen, ]= mmse
}
}



##single##after select the best parameters
temp = which(resmse== min(resmse), arr.ind = T)[1, ]
masterr =  crank[temp[1]]#8#mcrank[which.min(do.call(rbind, resmse)[, 1])]
pen = listpen[temp[2]]
source('estsimufun1.r')

BICq = matrix(NA, nsim, length(seqnum))
for(num in 1:length(seqnum)){
	BICq[, num]= mserror1[, num] +   (log(length(time))) ^pen*(masterr * 2 ) * (num)		
}

smix = apply(BICq, 1, which.min)

table(smix)



roierror = vector('list')

roierror = apply((abs(lmestX[[smix[1]]][, , 1] -  datamatrix[[1]])/datamatrix[[1]])^2, 1, mean)

plotindex =sapply(quantile(roierror), function(y)which.min(abs(roierror - y)))







###plot
time1 = seq(1, 60, length.out = ncol(datamatrix[[1]])) #MEG = 60, fMRI = 360
meanestX =apply(cmX, c(1, 2), quantile, c(0.025, 0.5, 0.975), na.rm = T)
mmX = apply(cmX, c(1, 2), mean,  na.rm = T)
plotnames = c(0, 25, 50, 75, 100)
for(l in 1:length(plotindex)){
     i = plotindex[l]
      for(k in round(seq(1, nsim, length.out = 10))){
# pdf(paste('figures/real_meannchg', i,'quantile', plotnames[l], '.pdf', sep = ''))

# obsermm = Reduce('+', datamatrix)/length(datamatrix)
# plot(mmX[ i,  ]~time, col = 2, lwd = 3, ylab = 'Signal', xlab = 'Time', ylim = range(c(meanestX[2, i, ])), xlim = range(time), type = 'l', lty = 2)
#  lines(meanestX[2,i,  ]~time, col = 2, lwd = 3, ylab = 'Signal', xlab = 'Time')
# lines(meanestX[1,i,  ]~time, col = 4, lwd = 3, ylab = 'Signal', xlab = 'Time', lty = 2)
# lines(meanestX[3,i,  ]~time, col = 4, lwd = 3, ylab = 'Signal', xlab = 'Time', lty = 2)
# #lines(obsermm[i, ]~time, col =1, lwd = 2, ylab = 'Signal', xlab = 'Time', lty = 5)
# lines(smX[[1]][i, ]~time, col =1, lwd = 2, ylab = 'Signal', xlab = 'Time', lty = 1)
# dev.off()
#meanestX[2, , ] = mestX[, , k]
pdf(paste('figures/realrankrdata_', i,'_',  plotnames[l],  '_', sp, '.pdf', sep = ''))
mestX= lmestX[[smix[k]]]#
plot(mestX[i, , k ]~time1, col = 2, type = 'l', lwd = 3, ylab = 'Signal', xlab = 'Time', ylim = range(c(mestX[i, , k], datamatrix[[k]][i, ])), xlim = range(time1), main = paste('The', i, 'th ROI of ', 'the fMRI sample'))
lines(datamatrix[[k]][i, ]~time1, col =1, lwd = 3, ylab = 'Signal', xlab = 'Time', lty = 5, pch = '+')
#lines(smX[[k]][i, ]~time1, lty = 4, col = 2, lwd  = 3)
#abline(v = time1[mchgres[[1]]$Test[smix[k]-1, ]])

dev.off()
}
}






pdf(paste('figures/fmri_', sp, '.pdf', sep = ''))
Xmat = datamatrix[[1]]
chgpoint = mchgres[[1]]$Test[smix-1, ]
plot(Xmat[1, ]~time1, col = 1, type = 'l', lwd = 0.8, ylab = 'Signal', xlab = 'Time', ylim = range(Xmat), xlim = range(time1))
for(i in 1:nrow(Xmat)){
   lines (Xmat[i, ]~time1, col = i, lwd = 0.8, ylab = 'Signal', xlab = 'Time')
}
abline(v = time1[chgpoint], lty = 2, lwd = 3)
dev.off()
#abline(v = time1[chgpointcb][-length(time1[chgpointcb])], lty = 3, lwd = 3, col = 4)

####The following is to calculate the eigen values and weighed U####


##plot eigenvalues both real and image parts
pdf(paste('figures/realeigenD_', sp, '.pdf', sep = ""))
LamM = Re(mLam[[smix]]) * freq  /30 ##MEG 30, fMRI = 180
plot(LamM[1, ]~time1, type = 'l',  lty = 1, ylab = 'change of growth/decay constant', xlab = 'Time', lwd = 3, ylim = c(min(LamM, na.rm = T), max(LamM, na.rm = T) * 2 ))
for(i in 2:nrow(LamM)){
    lines(LamM[i, ]~time1, type = 'l', col = i, lty = i,  lwd = 3)
}
legend('topright',paste("Lam", 1:nrow(LamM)), col = c(1:nrow(LamM)), lty = c(1:nrow(LamM)), lwd = 3, cex = 0.6)
dev.off()

pdf(paste('figures/ImeigenD_', sp, '.pdf', sep = ""))

LamM = Im(mLam[[smix]])  * freq /(30 * 2 * pi) ##MEG 30, fMRI 180
plot(LamM[1, ]~time1, type = 'l',  lty = 1, ylab = 'change of frequences', xlab = 'Time', lwd = 3, ylim = c(min(LamM, na.rm = T), max(LamM, na.rm = T) * 2))
for(i in 2:nrow(LamM)){
    lines(LamM[i, ]~time1, type = 'l', col = i, lty = i,  lwd = 3)
}
legend('topright',paste("Lam", 1:nrow(LamM)), col = c(1:nrow(LamM)), lty = c(1:nrow(LamM)), lwd = 3, cex = 0.6)
dev.off()

#save(resmse, mestU, lmestX, mchgres, mLam, smix,mserror1,  file = paste('allfMRIresult_', sp, '.Rdata', sep = ''))