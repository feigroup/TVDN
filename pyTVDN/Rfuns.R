library(fda) 
library(magrittr)


fourier_reg <- function(time, y, nbasis=10){
   n <- length(time)
   basisMat <- fourier(time, nbasis=nbasis) 
   basisMatDer <- fourier(time, nbasis=nbasis, nderiv=1) 
   fit <- lm(y~basisMat-1)
   coefs <- coef(fit)
   coefss <- rep(coefs, n) %>% matrix(ncol=length(coefs), byrow=TRUE)
   yhat <- rowSums(coefss*basisMat)
   dyhat <- rowSums(coefss*basisMatDer)
   list(yhat=yhat, dyhat=dyhat)
}
