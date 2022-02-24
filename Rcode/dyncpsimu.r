rm(list = ls())
library(fda)
library(MASS)
library(glmnet)
library(mvtnorm)
library(R.matlab)
fMRI = readMat('../data/fMRI_sample.mat')
fMRI = fMRI$time.series
time = seq(0, 2, length.out = 180)
set.seed(2021) ##6chg6rank 2021
step = diff(time)[1]
dfMRI = fMRI
basis = create.bspline.basis(range(0, 3), nbasis = 15, norder = 4)
ebase = eval.basis(basis, time)
Lebase = eval.basis(basis, time, 1)
for(i in 1:nrow(fMRI)){
    dfMRI[i, ]= predict(smooth.spline(time, fMRI[i, ], lambda = 0.001), deriv = 1)$y
}
nI = 20
dXmat = dfMRI
Xmat = fMRI
scan = rep(0, length(time))
for(i in nI:(length(time) - nI)){
    ix1 = (i - nI + 1) : i
    ix2 = ( i + 1) : (i + nI)
    ix = (i-nI + 1) : (i + nI)
    fullA = dXmat[, ix] %*% t(Xmat[, ix] ) %*% ginv(Xmat[, ix] %*% t(Xmat[, ix]) )
    leftA = dXmat[, ix1] %*% t(Xmat[, ix1] ) %*% ginv(Xmat[, ix1] %*% t(Xmat[, ix1]))
    rightA = dXmat[, ix2] %*% t(Xmat[, ix2] ) %*% ginv(Xmat[, ix2] %*% t(Xmat[, ix2]))
    temp = as.vector(dXmat[, ix] - fullA %*% Xmat[, ix])
    ssRfull = t(temp) %*% temp
    temp = as.vector(dXmat[, ix1] - leftA %*% Xmat[, ix1])
    ssRleft = t(temp) %*% temp
    temp = as.vector(dXmat[, ix2] - rightA %*% Xmat[, ix2])
    ssRright = t(temp) %*% temp
    scan[i]= ssRfull - ssRleft - ssRright
}
candlist = NULL
for(i in nI:(length(time) - nI)){
    if(scan[i] == max(scan[(i - nI + 1): (i + nI)]) & scan[i] > min(scan[(i - nI + 1): (i + nI)]))
        candlist = c(candlist, i)
}
candlist = c(0, candlist, length(time))
fullA = array(NA, c(nrow(fMRI), nrow(fMRI), length(candlist)-1))
svdA = matrix(NA, nrow(fMRI), length(candlist)-1)
for(j in 1:(length(candlist) -1)){
    left = candlist[j] + 1
    right = candlist[j + 1]
    ix = left :right
    fullA[, , j]= dXmat[, ix] %*% t(Xmat[, ix] ) %*% ginv(Xmat[, ix] %*% t(Xmat[, ix]))
    svdA[, j] = (eigen(fullA[, , j])$values)
}
AA = apply(fullA, c(1, 2), sum)
AA = svd(AA)
r = 6
six = 1:r
AA = AA$u[, six] %*% diag(AA$d[six]) %*% t(AA$v[, six])
temp = eigen(AA)
U = temp$vector
V = temp$values 
V = V #* c(1 , 1,  0.5, 0.5, 0.2, 0.2, rep(0, length(V) - length(six)))
candlist0 = candlist
candlist0 =c(0, 50, 99, 144, 180) #candlist0[-2]# round(c(0.1, x0.23, 0.40, 0.65, 0.76, 0.91) * 180)#36, 99, 144
#candlist0 = c(0, 180)
ratio = seq(-7, 8, length.out = 12)
#ratio = ratio[-c(1:3)]
#ratio = rep(ratio[1:(r/2)], each = 2)
svdA = matrix(NA, nrow(Xmat),  length(candlist0)-1)
for(j in 1:(length(candlist0)-1)){
    svdA[, j]= V * ratio[j]/10
    }



svdA[(r + 1):nrow(svdA), ] = 0

Res <- list()
Res[[1]] <- U
Res[[2]] <- svdA

save(Res, file="SimuEigen.RData")

numchg =3
nsim = 100
chg = matrix(NA, nsim, numchg )
mestU = array(NA, c(nrow(U), ncol(U), nsim))
truematrix = datamatrix = vector('list')
k = 1
errV =  U %*% diag(svdA[, k]/10) %*%solve(U) * matrix(rbinom(90 *90, 1, 0.1), 90, 90)#
Xmat =Xmat1=  dXmat1 = dXmat = matrix(NA, nrow(dfMRI), length(time))
Xmat1[, 1] =Xmat[, 1] = matrix(rnorm(nrow(Xmat), 0, (step)/8), nrow(Xmat),1)
for(itr in 1:nsim){
    time = seq(0, 2, length.out = 180)
    step = diff(time)[1]
    Xmat[, 1] = Xmat1[, 1] +  errV %*% matrix(rnorm(nrow(Xmat), 0, (step)/8), nrow(Xmat),1)
  
    k = 1
    dXmat1[, 1] = U %*% diag(svdA[, 1]) %*%solve(U)%*% Xmat1[, 1] * step
    #Xmat[, 1] = Xmat1[, 1] + errV %*% matrix(rnorm(nrow(Xmat), 0, (step)/8), nrow(Xmat),1)
    for(j in 2:length(time)){
       # errV =  U %*% diag(svdA[, k]/4) %*%t(V)
        if(j %in% candlist0 & j != length(time)){
            k = k + 1
        }
        Xmat1[, j]= Xmat1[, j-1] + dXmat1[, j-1]
        Xmat[, j] = Xmat1[, j] +  errV %*%  matrix(rnorm(nrow(Xmat), 0, (step)/8), nrow(Xmat),1)
        
        dXmat1[, j] = U [,six]%*% diag(svdA[six, k] * step) %*% solve(U)[six, ]%*% Xmat1[, j]
       # print(svdA[, k])
    }
    Xmat = Re(Xmat)
    plot(Xmat[1, ]~time, col = 1, type = 'l', lwd = 0.8, ylab = 'Signal', xlab = 'Time', ylim = range(Xmat), xlim = range(time))
for(i in 1:90){
    lines (Xmat[i, ]~time, col = i, lwd = 0.8, ylab = 'Signal', xlab = 'Time')
}

dev.off()
    datamatrix[[itr]]= Re(Xmat)
    truematrix[[itr]] = Re(Xmat1)
    print(itr)
    }
trueU = U
save(datamatrix, truematrix, candlist0,  trueU, file = 'datamatrix3chg6rank.Rdata')
