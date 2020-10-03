dynProgA<- function(y, x, Kmax, Lmin=1) 
{
    Nr  <- Kmax - 1
    y = y
    x = x
    n <- ncol(y)
    V <- matrix(Inf, nrow = n, ncol = n)
    for (j1 in (1:(n-Lmin+1))){
        for (j2 in ((j1+Lmin-1):n)) {
         
            yj <- y[, j1:j2, drop = F]
            Xj <- x[, j1:j2, drop = F]
            nj <- j2-j1+1
            A = matrix(0, nrow(yj), nrow(Xj))
            for(j in 1:(nrow(Xj)/2)){
                tempY = yj[((j-1)* 2+1): (j * 2), ]
                tempX = Xj[((j-1)* 2+1): (j * 2), ]
                corY = tempY%*%t(tempX)
                corX = sum(diag(tempX %*% t(tempX)))
                a = sum(diag(corY))/corX
                b = (corY[2, 1] - corY[1, 2])/corX
                A[((j-1)* 2+1): (j * 2), ((j-1)* 2+1): (j * 2)] = matrix(c(a, b, -b, a), 2, 2)
                }
           # parm = rep(1, nrow(Xj))
           # parm = optim(parm, obj, gr = NULL, method = 'L-BFGS-B', Y = yj, X = Xj, retresd = 0)$par
            #A= yj%*%t(Xj) %*% solve(Xj %*% t(Xj))
      #  browser()
       # print(A)
       # temp = errV  +  diag(nrow(errV ))
       # temp = svd(diag(nrow(errV )) + temp %*% t(temp))
       # temp = temp$u %*% diag((temp$d)^(-1/2)) %*% t(temp$v)
            resd = t( yj - A %*% Xj)
                                        # print(dim(resd))#%*% errV
                                        # stdresd = resd/sd(resd)
            sig = ((t(resd) %*% (resd)/(nj)))
            svdSig= svd(sig)
          #  print(max(svdSig$d)/min(svdSig$d))
            ix = which(svdSig$d >1.490116e-8 * svdSig$d[1] )
            newcov = t(t(svdSig$u[, ix]) %*% t(resd))
                                        # hatV = solve(sig)
                                        # print(diag(svdSig$d[ix]))
            temp = - sum((dmvnorm(newcov, rep(0, ncol(newcov)), diag(svdSig$d[ix]), log = TRUE)))

            V[j1,j2] <- temp 
    }
  }
  U <- vector(length=Kmax)
  U[1] <- V[1,n]
  D <- V[,n] 
  Pos <- matrix(nrow = n, ncol = Nr) 
  Pos[n,] <- rep(n,Nr)    
  tau.mat <- matrix(nrow = Nr,ncol = Nr) 
  for (k in 1:Nr){
    for (j in 1:(n-1)){
      dist <- V[j,j:(n-1)] + D[(j+1):n]
      D[j] <- min(dist)
      Pos[j,1] <- which.min(dist) + j
      if (k > 1) { Pos[j,2:k] <- Pos[Pos[j,1],1:(k-1)] }
    }
    U[k+1] <- D[1]
    tau.mat[k,1:k] <- Pos[1,1:k]-1
  }
  out <- list(Test=tau.mat, obj=data.frame(K=(1:Kmax),U=U))
  return(out)
}
selectkappa = function(chgres, kappam){
    load('fcR.rdata')
    fcR = fcR[-c(1, 5, 37, 42), ]
    fcR = fcR
    names = c("Default", "Dorsal_Attention", "Frontoparietal", "Limbic", "Somatomotor", "Ventral_Attention", "Visual")
    rank =7
    fcR[is.nan(fcR)] = 0
    minmax = function(x){
        return((x - min(x))/(max(x) - min(x)))
    }
    fcR[1:34, ] = apply(fcR[1:34, ], 2, minmax)
    fcR[35:68, ] = apply(fcR[35:68, ], 2, minmax)

   
    meanmax = rep(NA, length(kappam))
    for(kappaj in 1:length(kappam) ){
        kappa = kappam[kappaj]
        SIC = chgres$obj[, 2] +  (log(ncol(nY)))^kappa*( nrow(nX) ) * ((1:20 + 1)) + log(1:20)


        numchg = which(SIC ==min(SIC)) -1
        chgpoint = chgres$Test[numchg, ]
        chgpoint = c(0, chgpoint[!is.na(chgpoint)], length(time))
        ResegS = matrix(NA, (length(chgpoint)-1), nrow(ndXmat))
        ResegU = matrix(NA, nrow(dXmat), (length(chgpoint)-1))
        for(itr in 1:(length(chgpoint)-1)){
            ix = chgpoint[itr]:(chgpoint[itr + 1]-1)
            yj = nY[, ix]
            Xj = nX[, ix]
            lambda = rep(NA, nrow(ndXmat))
            for(j in 1:(nrow(Xj)/2)){
                tempY = yj[((j-1)* 2+1): (j * 2), ]
                tempX = Xj[((j-1)* 2+1): (j * 2), ]
                corY = tempY%*%t(tempX)
                corX = sum(diag(tempX %*% t(tempX)))
                a = sum(diag(corY))/corX
                b = (corY[2, 1] - corY[1, 2])/corX
                lambda[j]= sqrt(a + b*1i)
            }
            ResegS[itr, ]= sqrt(lambda)#svd(Ymat %*% t(Xmat) %*% ginv(Xmat %*% t(Xmat)))$u
            ResegU[, itr]= apply(estU[, six] %*% diag((lambda)), 1, sum)
        }
        segU = sqrt(Re(ResegU)^2 + Im(ResegU)^2)
        ncorr = vector('list', ncol(segU))
       for(i in 1:ncol(segU)){
           temp = segU[ ,  i]
           ncorr[[i]] = abs(cor(temp, fcR))
       }
        segcorr = do.call(rbind, ncorr)
        if(nrow(segcorr) == 1){
            meanmax[kappaj]  = 0
        }else{meanmax[kappaj] =mean(apply(segcorr, 1, max))}#
    }
return(meanmax)
}
