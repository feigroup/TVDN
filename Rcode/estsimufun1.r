
r = masterr
for(k in 1:nsim){
      fMRI = datamatrix[[k]]

      downfMRI = fMRI 
      fMRI = downfMRI
      time = seq(0, 2, length.out = ncol(fMRI))
      step = diff(time)[1]
      dfMRI = fMRI
      mXmat = Xmat=  dXmat = matrix(NA, nrow(fMRI), length(time))
      Xmat = fMRI
      for(i in 1:nrow(Xmat)){
            dXmat[i, ]=predict(smooth.spline(time, Xmat[i, ], lambda = 1e-6), deriv = 1)$y 
            mXmat[i, ] = predict(smooth.spline(time, Xmat[i, ], lambda = 1e-6), deriv = 0)$y
       }

###simu 6 chag lambda = 1e-6 ##h = bw.nrd0(time) /2 select ranks lambda = 1e-4 
###simu 3 chag lambda = 1e-6 ## h = hw.nrd0(time) select ranks lambda = 1e-6
##real fMRI h = bw /2 lambda = 1e-4  
##real MEG h = bw/2 lambda = 1e-4   

      h =bw.nrd0(time) 

       sampleix = seq(1, length(time), 4) ##MEG sample 20 fMRI = 4
      YMat =   array(NA, c(nrow(dXmat), nrow(Xmat),  length(sampleix)))
       XMat =  array(NA, c(nrow(Xmat), nrow(Xmat),  length(sampleix)))

       Kmat = array(NA, c(nrow(dXmat), nrow(dXmat) ,  length(sampleix )))
       for(i in 1:length(sampleix )){
       s = time[sampleix[i]]
       kernel = dnorm((s - time)/h)^(1/2)
       kerdXmat = diag(kernel) %*% t(dXmat) 
       kerXmat = diag(kernel) %*% t(mXmat)
       #temp = rrr(kerdXmat, kerXmat)
       XY = (t(kerdXmat) %*% (kerXmat) )/length(time)
        M = t(kerXmat) %*% (kerXmat) /length(time)
        sM = svd(M)
        invr = which(cumsum(sM$d)/sum(sM$d) >= 0.999) ###simulation >0.999
       
       
       Kmat[, , i]= (XY ) %*%sM$u[, 1: (invr[1]), drop = F] %*% diag(sM$d[1: (invr[1])]^(-1), invr[1], invr[1]) %*% t(sM$v[, 1:invr[1], drop = F])#ginv(M, h^2 * 1e-2)#Y
      }
       #sM

       sampix = sample(1:length(time), length(time))
       svdKmat = (apply(Kmat, c(1, 2), sum))[1:nrow(fMRI), 1:nrow(fMRI)]
       svdKmat = svdKmat 
       estU = eigen(svdKmat)$vector#Re(eigen(svdKmat)$vector)
       estV =eigen(svdKmat)$value
       six = which(!duplicated(Mod(estV[1:r])))#1:r#(selecrank[k] * 2)
       dupsix = which(duplicated(Mod(estV[1:r])))
      # print(six)
       
       ndXmat =solve(estU)[six, ] %*%  dXmat# 
       ndXmat = matrix(ndXmat, ncol = length(time))

       nXmat =  solve(estU)[six, ] %*%mXmat
       nXmat = matrix(nXmat, ncol = length(time))
       nY = matrix(NA, nrow(ndXmat) * 2, ncol(ndXmat))
       nX = matrix(NA, nrow(nXmat) * 2, ncol(nXmat))
       for(j in 1:length(six)){
       nY[((j-1) * 2 + 1):((j) * 2),  ] = rbind(Re(ndXmat[j, ]), Im(ndXmat[j, ]))
       nX[((j-1) * 2 + 1):((j) * 2),  ] = rbind(Re(nXmat[j, ]), Im(nXmat[j, ]))
       }



       chgres = dynProgA(nY, nX, 10, 18)  ##MEG lmin = 60 fMIR lmin = 4
       mchgres[[k]] = chgres
       print(chgres$Test[length(candlist0)-2, ])
       mestU[[k]] = estU
       mY[[k]] = nY
       mX[[k]] = nX
       uniqsix[[k]] = six
       ldupsix[[k]] = dupsix
       smX[[k]] = mXmat
       #print(k)
}
r = masterr
seqnum = seq(0, 9, 1)
mLam= mchgpoint= lmestX = vector('list')
mserror1 = mserror= corrm = matrix(NA, nsim, length(seqnum))
for(sq in 1:length(seqnum)){
      numchg = seqnum[sq]
      mestX = array(NA, c(dim(fMRI),  nsim))
      for(k in 1:nsim){
      	six = 1:r#(selecrank[k] * 2)
      	nY = mY[[k]]
	      nX = mX[[k]]
	      if(numchg == 0){
	      	chgpoint = c(1, length(time))
		}else{
			chgpoint = mchgres[[k]]$Test[numchg, ]
	      	chgpoint = c(0, chgpoint[!is.na(chgpoint)], length(time)) # 1 -> 0
		}
	      mchgpoint[[k]] = chgpoint
	#      print(chgpoint)
	      ResegS = matrix(NA, (length(chgpoint)-1), r)
	      #ResegU = matrix(NA, 90, (length(chgpoint)-1))
	      for(itr in 1:(length(chgpoint)-1)){
   	      	 ix = (chgpoint[itr]+1):(chgpoint[itr + 1]) # 
    		       yj = nY[, ix]
    		       Xj = nX[, ix]
    		       lambda = rep(NA, r)
            	 for(j in 1:(nrow(Xj)/2)){
                       tempY = yj[((j-1)* 2+1): (j * 2), ]
                	     tempX = Xj[((j-1)* 2+1): (j * 2), ]
                	     corY = tempY%*%t(tempX)
                	     corX = sum(diag(tempX %*% t(tempX)))
                	     a = sum(diag(corY))/corX
                	     b = (corY[2, 1] - corY[1, 2])/corX
                	     lambda[uniqsix[[k]][j]]= (a + b*1i)
                   }
		       temp = which(is.na(lambda))
		       lambda[temp] = Conj(lambda[temp - 1])
		       	ResegS[itr, ]= (lambda)#svd(Ymat %*% t(Xmat) %*% ginv(Xmat %*% t(Xmat)))$u
    		       #ResegU[, itr]= apply(estU[, six] %*% diag((lambda)), 1, sum)
		}
		
		
		LamM = matrix(NA, ncol(ResegS), length(time))
		tix = 1:length(time)
		LamM[, 1] = ResegS[1, ]
		for(itr in 2:length(chgpoint)){
    		ix = tix<=chgpoint[itr] & tix>chgpoint[itr- 1]
    		LamM[, ix] = ((ResegS[itr-1, ]))
		}
		estX= matrix(NA, nrow(datamatrix[[k]]), ncol(datamatrix[[k]]))
		estX[, 1] = truematrix[[k]][, 1]
		for(i in 2:length(time)){
		estX[, i] = (mestU[[k]][, six] %*% diag(LamM[, i], r, r) %*% solve(mestU[[k]])[six, ]%*% estX[, i-1] )* step + estX[, i-1]
		}
		mestX[, , k] = Re(estX)
		corrm[k, sq]= cor(as.vector(mestX[, , k]), as.vector(truematrix[[k]]))
		resd = t(mestX[,1:180, k] - datamatrix[[k]][, 1:180])
		#sig = t(resd) %*% resd
		#svdSig = svd(sig)
		#ix = which(svdSig$d >1.490116e-08 * svdSig$d[1] )
            	#newcov = t(t(svdSig$u[, ix]) %*% t(resd))
    		resd2 = resd^2
		mserror[k,  sq] = mean(resd2)#- sum((dmvnorm(newcov, rep(0, ncol(newcov)), diag(svdSig$d[ix], length(ix), length(ix)), log = TRUE)))
		mserror1[k,  sq] = mchgres[[k]]$obj[numchg+1, 2]# +  (log(length(time))) ^ 1.5 *( length(six) * 2 ) * (numchg + 1)		
		#print(mserror1[k, sq])
		mLam[[sq]] = LamM
}
	lmestX[[sq]] = mestX
}
