# First three to do cluster and obtain change points
segCorr <- function(downseq, wsize, seqw){
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
segPCA <- function(downseq, wsize, seqw, rank=6){
    tix = 1:length(time)
    
    PCAU = NULL
    for(itr in 1:length(seqw)){
        ix = itr : (itr + wsize -1)
        temp = svd(cov(t( downseq[, ix])))
        #  rank = wsize-1#sum(cumsum(temp$d)/sum(temp$d)<= perrank) + 1
        temp1 = as.vector(temp$u[, 1:rank, drop = F] %*%  diag((sign(temp$u[1, 1:rank])), rank, rank))# as.vector
        #why not use the eigen value ??
        PCAU = cbind(PCAU,  temp1)
    }
    return(PCAU)
}

DMD <- function(downseq, wsize, seqw, rank=6){
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
        # same problem ??
        PCAU = cbind(PCAU,  temp1)
        
        
    }
    return(PCAU)
}

# These two for obtaining the correlation with default mode
segPCAOrg <- function(downseq, wsize, seqw, rank=6){
    tix = 1:length(time)
    
    PCAU = NULL
    for(itr in 1:length(seqw)){
        ix = itr : (itr + wsize -1)
        temp = svd(cov(t( downseq[, ix])))
        #  rank = wsize-1#sum(cumsum(temp$d)/sum(temp$d)<= perrank) + 1
        temp1 = (temp$u[, 1:rank, drop = F] %*%  diag((sign(temp$u[1, 1:rank])), rank, rank)) # no as.vector
        PCAU = cbind(PCAU,  temp1)
    }
    return(PCAU)
}


DMDOrg <- function(downseq, wsize, seqw, rank=6){
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
        # no as.vector	
        PCAU = cbind(PCAU,  temp1)
        
        
    }
    return(PCAU)
}

minmax <- function(x){
    return((x - min(x))/(max(x) - min(x)))
}

innOut <- function(absWeightedU, region_list, AAL){
    outPut <- array(dim=c(91, 109, 91))
    wUmode <- minmax(absWeightedU) # weightedU[, em] is the mode of the weighted U at em segment. 
    for(i in 1:length(wUmode)){
        ix  = which(AAL== region_list[i, 2]) # Give the name for each region
        outPut[ix] <-  wUmode[i]
    }
    outPut
}


chgF <- function(res, seqslide, ncen=4){
    # res: num of feature x num of point
    kmres <- kmeans(t(res), ncen)
    seqslide[which(diff(kmres$cluster)!=0) + 1]
}


corF.fMRI <- function(res, fcR){
    fcR <- apply(fcR, 2, minmax)
    ncorr <- list()
    for (i in 1:ncol(res)){
        ncorr[[i]] <- abs(cor(res[, i], fcR))
    }
    do.call(rbind, ncorr)
}