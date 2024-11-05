chooseCRANmirror(ind=1)
install.packages("prodlim")
install.packages("Rcpp", dependencies=TRUE)
install.packages("grf")
# random forest 
### Clean Memory ###

# 설치 후 라이브러리 로드
library(grf)

rm(list=ls())
#args <- commandArgs(trailingOnly = TRUE)
#aa <- args[1]
#print(aa)
#print(args[1])
#print(as.integer(args[1]))
options(warn = -1)
### Call Libraries ###
library(rpart)
library(survival)
library(prodlim)
library(timereg)
#library(etm)
library(randomForestSRC)
library(MASS)
library(pec)
ptm <- proc.time()
#작업 디렉토리위치를 잡아줍니다.
setwd("/Users/imac/Documents/연구실/조영주교수님_Lab/MultiCausalSurvivalForest/MultiCausalSurvivalForest_beta")
#로드 하기 전에, sourceCpp.R 파일의 default_dir를 현재 디렉토리로 경로를 수정해줍니다.
source("sourceCpp.R")
rho = 0.9
nump = 20
Sig <- matrix(0,ncol=nump,nrow=nump)
for(r in 1:nump){
  for(v in 1:nump){
    if(r==v){
      Sig[r,v] <- 1
    }else{
      Sig[r,v] <- rho^(abs(r-v)) 
    }
  }
}


### Generate training data
gen.data.tr <- function(myseed,n,p0,beta10,beta20) {
  set.seed(myseed)
  X <- mvrnorm(n,mu = rep(0,20),Sigma = Sig)
  x1 <- X[,1];x2 <- X[,2];x3 <- X[,3]
  x4 <- X[,4];x5 <- X[,5];x6 <- X[,6]
  x7 <- X[,7];x8 <- X[,8];x9 <- X[,9]
  x10 <- X[,10];x11 <- X[,11];x12 <- X[,12]
  x13 <- X[,13];x14 <- X[,14];x15 <- X[,15]
  x16 <- X[,16];x17 <- X[,17];x18 <- X[,18]
  x19 <- X[,19];x20 <- X[,20]
  z <- sample(0:2, size = n, replace=T)
  z1 <- as.numeric(z==1);z2 <- as.numeric(z==2)
  xx.true <- cbind(z1,z2,sin(pi*x1*x2),sqrt(abs(x3)),x10,as.numeric(x11 > 0),abs(x15))
  index.temp <- rbinom(n,1,1-(1-p0)^exp(xx.true %*% beta10))
  index <- 2 - index.temp
  event.comp.time <- rep(0,n)
  for(i in 1:n){
    if(index[i] == 1){
      u <- runif(1,0,1)
      event.comp.time[i] <- -log(1-(1-(1-u*(1-(1-p0)^exp(t(xx.true[i,]) %*% beta10)))^(1/exp(t(xx.true[i,]) %*% beta10)))/p0)
    }else{
      #v <- runif(1,0,1)
      event.comp.time[i] <- rexp(1,exp(t(xx.true[i,]) %*% beta20))
    }
  }
  
  
  # alpha0 <- 1
  #  alpha1 <- 0.5
  #  alpha2 <- -0.3
  #  logit.lp <- alpha0 + x1*alpha1 + x2*alpha2
  #  probs_z <- exp(logit.lp)/(1+exp(logit.lp))
  #  z <- rbinom(n,1,probs_z)
  #  event.comp.time <- z*event.comp.time.z1 + (1-z)*event.comp.time.z0
  #  index <- z*index_z1 + (1-z)*index_z0
  #cen <- rexp(n,1.475) #beta10 = 3
  #cen <- rexp(n,0.75) #beta10 = 2
  mu <- x1+x3+x5
  cen <- exp(rnorm(n,mu+0.1,1))
  #cen <- runif(n,0,2.83) #beta10 = 2
  #cen <- runif(n,0,3.275) #beta10 = 1
  #cen <- runif(n,0,3.075) #beta10 = 1.5
  return(list(event.comp.time,index,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,cen,z,z1,z2))
}

# Length of data vector created
num = 2500
# Length of dataset used to fit model, num - nu is length of test set.
nu=500
test_len <- num - nu
preset=(nu+1):num

beta10_true=c(-1,-0.7,rep(0.5,3),0.6,-0.3)
beta20_true=c(-1.5,-0.4,rep(-0.5,3),0.5,0.1)

p0 = 0.5

# Find time points
km_times <- c(0.3559963, 1.0384856, 2.9275240)
# set seed - possible to change any numbers
seednum <- 1  #as.integer(args[1])
seednum
data.tr.temp <- gen.data.tr(seednum, nu, p0=0.5, beta10 = beta10_true, beta20 = beta20_true)
event.time.temp <- data.tr.temp[[1]]
cen.temp <- data.tr.temp[[23]]
z.temp <- data.tr.temp[[24]]
z1.temp <- data.tr.temp[[25]]
z2.temp <- data.tr.temp[[26]]
obs.time.temp <- pmin(event.time.temp, cen.temp)
status.woc.temp <- data.tr.temp[[2]]
status.temp <- as.numeric(event.time.temp <= cen.temp) * status.woc.temp
status.all.temp <- as.numeric(event.time.temp <= cen.temp)
x1.temp <- data.tr.temp[[3]];x2.temp <- data.tr.temp[[4]];x3.temp <- data.tr.temp[[5]]
x4.temp <- data.tr.temp[[6]];x5.temp <- data.tr.temp[[7]];x6.temp <- data.tr.temp[[8]]
x7.temp <- data.tr.temp[[9]];x8.temp <- data.tr.temp[[10]];x9.temp <- data.tr.temp[[11]]
x10.temp <- data.tr.temp[[12]];x11.temp <- data.tr.temp[[13]];x12.temp <- data.tr.temp[[14]]
x13.temp <- data.tr.temp[[15]];x14.temp <- data.tr.temp[[16]];x15.temp <- data.tr.temp[[17]]
x16.temp <- data.tr.temp[[18]];x17.temp <- data.tr.temp[[19]];x18.temp <- data.tr.temp[[20]]
x19.temp <- data.tr.temp[[21]];x20.temp <- data.tr.temp[[22]]
# Creating the dataset
dat.tr <- data.frame(obs=obs.time.temp,status = status.temp,status.all=status.all.temp,z = z.temp, z1 = z1.temp, z2 = z2.temp, x1=x1.temp, x2=x2.temp, x3=x3.temp, x4=x4.temp, x5=x5.temp, x6=x6.temp, x7=x7.temp, x8=x8.temp, x9=x9.temp, x10=x10.temp,x11=x11.temp, x12=x12.temp, x13=x13.temp, x14=x14.temp, x15=x15.temp, x16=x16.temp, x17=x17.temp, x18=x18.temp, x19=x19.temp, x20=x20.temp)
p <- 20
source("parameters_cr.R")
horizon = km_times[1]

