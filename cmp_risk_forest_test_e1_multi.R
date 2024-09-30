 if (is.null(horizon) || !is.numeric(horizon) || length(horizon) != 1) {
    stop("The `horizon` argument defining the estimand is required.")
  }
X <- dat.tr[,7:(p+6)]
W <- dat.tr$z
#W <- sample(0:2, size=nu, replace=T)
#dat.tr$z <- W
dat.tr$z1 <- as.numeric(W==1)
dat.tr$z2 <- as.numeric(W==2)
Y <- dat.tr$obs
D <- dat.tr$status.all
e <- dat.tr$status
  has.missing.values <- validate_X(X, allow.na = TRUE)
  validate_sample_weights(sample.weights, X)
  Y <- validate_observations(Y, X)
  W <- validate_observations(W, X)
  D <- validate_observations(D, X)
  clusters <- validate_clusters(clusters, X)
  samples.per.cluster <- validate_equalize_cluster_weights(equalize.cluster.weights, clusters, sample.weights)
  num.threads <- validate_num_threads(num.threads)
  if (any(Y < 0)) {
    stop("The event times must be non-negative.")
  }
#  if (!all(W %in% c(0, 1))) {
 #   stop("The treatment values can only be 0 or 1.")
#  }
  if (!all(D %in% c(0, 1))) {
    stop("The censor values can only be 0 or 1.")
  }
  if (sum(D) == 0) {
    stop("All observations are censored.")
  }
  fY <- as.numeric(Y <= horizon & e == 1)
  if (is.null(failure.times)) {
    Y.grid <- sort(unique(Y))
  } else if (min(Y) < min(failure.times)) {
    stop("If provided, `failure.times` should be a grid starting on or before min(Y).")
  } else {
    Y.grid <- failure.times
  }
  if (length(Y.grid) <= 2) {
    stop("The number of distinct event times should be more than 2.")
  }
  if (horizon < min(Y.grid)) {
    stop("`horizon` cannot be before the first event.")
  }
  if (nrow(X) > 5000 && length(Y.grid) / nrow(X) > 0.1) {
    warning(paste0("The number of events are more than 10% of the sample size. ",
                   "To reduce the computational burden of fitting survival and ",
                   "censoring curves, consider discretizing the event values `Y` or ",
                   "supplying a coarser grid with the `failure.times` argument. "), immediate. = TRUE)
  }

  #if (is.null(W.hat)) {
  #  forest.W <- regression_forest(X, W, num.trees = max(50, num.trees / 4),
   #                               sample.weights = sample.weights, clusters = clusters,
    #                              equalize.cluster.weights = equalize.cluster.weights,
     #                             sample.fraction = sample.fraction, mtry = mtry,
      #                            min.node.size = 5, honesty = TRUE,
       #                           honesty.fraction = 0.5, honesty.prune.leaves = TRUE,
        #                          alpha = alpha, imbalance.penalty = imbalance.penalty,
         #                         ci.group.size = 1, tune.parameters = tune.parameters,
          #                        compute.oob.predictions = TRUE,
          #                        num.threads = num.threads, seed = seed, mahalanobis)
#    W.hat <- predict(forest.W)$predictions
#  } else if (length(W.hat) == 1) {
#    W.hat <- rep(W.hat, nrow(X))
#  } else if (length(W.hat) != nrow(X)) {
#    stop("W.hat has incorrect length.")
#  }
library(grf)
  # Compute (generalized) propensity score
  if (is.null(W.hat) & length(unique(W)) > 2) {
    forest.W <- probability_forest(X, factor(W), num.trees = max(50, num.trees / 4),
                                  sample.weights = sample.weights, clusters = clusters,
                                  equalize.cluster.weights = equalize.cluster.weights,
                                  sample.fraction = sample.fraction, mtry = mtry,
                                  min.node.size = 5, honesty = TRUE,
                                  honesty.fraction = 0.5, honesty.prune.leaves = TRUE,
                                  alpha = alpha, imbalance.penalty = imbalance.penalty,
                                  ci.group.size = 1,
                                  compute.oob.predictions = TRUE,
                                  num.threads = num.threads, seed = seed)
    W.hat <- predict(forest.W)$predictions
  } 
  #else (length(W.hat) == 1) {
  #  W.hat <- rep(W.hat, nrow(X))
  #} 
  #else if (length(W.hat) != nrow(X)) {
  #  stop("W.hat has incorrect length.")
  #}
  
  #W.centered <- W - W.hat
  W.centered1 <- as.numeric(W==1) - W.hat[,2]
  W.centered2 <- as.numeric(W==2) - W.hat[,3]
  args.nuisance_surv <- list(failure.times = failure.times,
                        num.trees = max(50, min(num.trees / 4, 500)),
                        sample.weights = sample.weights,
                        clusters = clusters,
                        equalize.cluster.weights = equalize.cluster.weights,
                        sample.fraction = sample.fraction,
                        mtry = mtry,
                        min.node.size = 15,
                        honesty = TRUE,
                        honesty.fraction = 0.5,
                        honesty.prune.leaves = TRUE,
                        alpha = alpha,
                        prediction.type = "Kaplan-Meier", # to guarantee non-zero estimates.
                        compute.oob.predictions = FALSE,
                        num.threads = num.threads,
                        seed = seed)
#dat.tr$z1 <- as.numeric(dat.tr$z==1)
#dat.tr$z2 <- as.numeric(dat.tr$z==2)
  # E[f(T) | X] = e_1(X) E[f(T) | X, Z = 1] + e_2(X) E[f(T) | X, Z = 2] + (1 - e_1(X)-e_2(X)) E[f(T) | X, Z = 0]
form<-"Surv(obs,status)~z1+z2+x1"
for (count in 2:p){
  form<-paste(form,"+x",as.character(count),sep="")
}
fit_forest <- rfsrc(as.formula(form), data=dat.tr, nodesize=20, samptype = "swr")
# prediction
dat.sw.tr0 <- dat.tr
dat.sw.tr1 <- dat.tr
dat.sw.tr2 <- dat.tr
# for Z=0
dat.sw.tr0$z1 <- 0
dat.sw.tr0$z2 <- 0
# for Z=1
dat.sw.tr1$z1 <- 1
dat.sw.tr1$z2 <- 0
# for Z=2
dat.sw.tr2$z1 <- 0
dat.sw.tr2$z2 <- 1
pred.fit.o <- predict(fit_forest)
pred.fit.z0 <- predict(fit_forest,newdata = dat.sw.tr0)
pred.fit.z1 <- predict(fit_forest,newdata = dat.sw.tr1)
pred.fit.z2 <- predict(fit_forest,newdata = dat.sw.tr2)
n <- nu
pred.mat.e1.z2 <- matrix(0, n, n)
pred.mat.e1.z1 <- matrix(0, n, n)
pred.mat.e1.z0 <- matrix(0, n, n)


pred.mat.e1.o <- matrix(0, n, n)
pred.mat.e2.o <- matrix(0, n, n)
a <- round(sort(dat.tr$obs),8)
b <- round(fit_forest$time.interest,8)
#intersect(a,b)
ind <- which(a %in% b)

pred.mat.e1.z0[,ind] <- pred.fit.z0$cif[,,1]
pred.mat.e1.z1[,ind] <- pred.fit.z1$cif[,,1]
pred.mat.e1.z2[,ind] <- pred.fit.z2$cif[,,1]

pred.mat.e1.o[,ind] <- pred.fit.o$cif.oob[,,1]
pred.mat.e2.o[,ind] <- pred.fit.o$cif.oob[,,2]
tlen <- 1:n
clen <- tlen[-ind]

  if(clen[1]==1){
  	pred.mat.e1.o[,clen[1]] <- 0
  	pred.mat.e2.o[,clen[1]] <- 0
  	pred.mat.e1.z2[,clen[1]] <- 0
  	pred.mat.e1.z1[,clen[1]] <- 0
  	pred.mat.e1.z0[,clen[1]] <- 0
      for(j in 2:length(clen)){
  	  pred.mat.e1.o[,clen[j]] <- pred.mat.e1.o[,(clen[j] - 1)]
  	  pred.mat.e2.o[,clen[j]] <- pred.mat.e2.o[,(clen[j] - 1)]
  	  pred.mat.e1.z2[,clen[j]] <- pred.mat.e1.z2[,(clen[j] - 1)]
  	  pred.mat.e1.z1[,clen[j]] <- pred.mat.e1.z1[,(clen[j] - 1)]
  	  pred.mat.e1.z0[,clen[j]] <- pred.mat.e1.z0[,(clen[j] - 1)]
     }
  }else{
     for(j in 1:length(clen)){
  	  pred.mat.e1.o[,clen[j]] <- pred.mat.e1.o[,(clen[j] - 1)]
  	  pred.mat.e2.o[,clen[j]] <- pred.mat.e2.o[,(clen[j] - 1)]
  	  pred.mat.e1.z2[,clen[j]] <- pred.mat.e1.z2[,(clen[j] - 1)]
  	  pred.mat.e1.z1[,clen[j]] <- pred.mat.e1.z1[,(clen[j] - 1)]
  	  pred.mat.e1.z0[,clen[j]] <- pred.mat.e1.z0[,(clen[j] - 1)]
    }
  }

S.hat <- 1 - pred.mat.e1.o - pred.mat.e2.o
pred.mat.e1.z2.f <- pred.mat.e1.z2[,ind]
pred.mat.e1.z1.f <- pred.mat.e1.z1[,ind]
pred.mat.e1.z0.f <- pred.mat.e1.z0[,ind]

horizonS.index <- findInterval(horizon, fit_forest$time.interest)
    if (horizonS.index == 0) {
      Y.hat <- rep(0, nrow(X))
    } else {
      #Y.hat <- W.hat * pred.mat.e1.z1.f[, horizonS.index] + (1 - W.hat) * pred.mat.e1.z0.f[, horizonS.index]
      Y.hat <- W.hat[,2] * pred.mat.e1.z1.f[, horizonS.index] + W.hat[,3] * pred.mat.e1.z2.f[, horizonS.index] + 
        (1 - W.hat[,2]-W.hat[,3]) * pred.mat.e1.z0.f[, horizonS.index]
    }
  
  # The conditional survival function for the censoring process S_C(t, x, w).
  args.nuisance_surv$compute.oob.predictions <- TRUE
  sf.censor <- do.call(survival_forest, c(list(X = cbind(X, dat.tr$z1, dat.tr$z2), Y = Y, D = 1 - D), args.nuisance_surv))
  C.hat <- predict(sf.censor, failure.times = Y.grid)$predictions$s1
 # if (target == "survival.probability") {
    # Evaluate psi up to horizon
    D[Y > horizon] <- 1
    Y[Y > horizon] <- horizon
 # }

  Y.index <- findInterval(Y, Y.grid) # (invariance: Y.index > 0)
  C.Y.hat <- C.hat[cbind(seq_along(Y.index), Y.index)] # Pick out P[Ci > Yi | Xi, Wi]

 # if (target == "RMST" && any(C.Y.hat <= 0.05)) {
 #   warning(paste("Estimated censoring probabilities go as low as:", round(min(C.Y.hat), 5),
 #                 "- an identifying assumption is that there exists a fixed positive constant M",
 #                 "such that the probability of observing an event past the maximum follow-up time ",
 #                 "is at least M (i.e. P(T > horizon | X) > M).",
 #                 "This warning appears when M is less than 0.05, at which point causal survival forest",
 #                 "can not be expected to deliver reliable estimates."), immediate. = TRUE)
 # } else if (target == "RMST" && any(C.Y.hat < 0.2)) {
 #   warning(paste("Estimated censoring probabilities are lower than 0.2",
 #                 "- an identifying assumption is that there exists a fixed positive constant M",
 #                 "such that the probability of observing an event past the maximum follow-up time ",
 #                 "is at least M (i.e. P(T > horizon | X) > M)."))
 # } else if (target == "survival.probability" && any(C.Y.hat <= 0.001)) {
 #   warning(paste("Estimated censoring probabilities go as low as:", round(min(C.Y.hat), 5),
 #                 "- forest estimates will likely be very unstable, a larger target `horizon`",
 #                 "is recommended."), immediate. = TRUE)
 # } else if (target == "survival.probability" && any(C.Y.hat < 0.05)) {
  #  warning(paste("Estimated censoring probabilities are lower than 0.05",
  #                "and forest estimates may not be stable. Using a smaller target `horizon`",
  #                "may help."))
  #}

#  eta <- compute_eta(S.hat, C.hat, C.Y.hat, Y.hat, W.centered,
#                     D, fY, Y.index, Y.grid, target, horizon)

horizon.cr.index <- findInterval(horizon, Y.grid)
# Compute P(t < T <= horizon, e=1)
Q.num.hat <- matrix(0,n,n)
Q.hat <- matrix(0,n,n) # Q(t, X) =  P(T_i <= horizon, e=1 | Z_i, T_i > t)
for(j in 1:n){
	Q.num.hat[,j] <- pred.mat.e1.o[, horizon.cr.index]-pred.mat.e1.o[,j]
	Q.hat[,j] <- Q.num.hat[,j]/S.hat[,j]
}
Q.hat[, horizon.cr.index:ncol(Q.hat)] <- 0

 

  # Pick out Q(Yi, X)
  Q.Y.hat <- Q.hat[cbind(seq_along(Y.index), Y.index)]
  numerator.one1 <- (D * (fY - Y.hat) + (1 - D) * (Q.Y.hat - Y.hat)) * W.centered1 / C.Y.hat
  numerator.one2 <- (D * (fY - Y.hat) + (1 - D) * (Q.Y.hat - Y.hat)) * W.centered2 / C.Y.hat
  # The conditional hazard function differential -d log(C.hat(t, x, w))
  # This simple forward difference approximation works reasonably well.
  # (note the "/dt" term is not needed as it cancels out in the lambda.C.hat / C.hat integral)
  log.surv.C <- -log(cbind(1, C.hat))
  dlambda.C.hat <- log.surv.C[, 2:(ncol(C.hat) + 1)] - log.surv.C[, 1:ncol(C.hat)]

  integrand <- dlambda.C.hat / C.hat * (Q.hat - Y.hat)
  numerator.two1 <- rep(0, length(Y.index))
  numerator.two2 <- rep(0, length(Y.index))
  for (sample in seq_along(Y.index)) {
    Yi.index <- Y.index[sample]
    numerator.two1[sample] <- sum(integrand[sample, seq_len(Yi.index)]) * W.centered1[sample]
    numerator.two2[sample] <- sum(integrand[sample, seq_len(Yi.index)]) * W.centered2[sample]
  }
### doubly robust ###
  numerator1 <- numerator.one1 - numerator.two1
  denominator1 <- W.centered1^2 
  numerator2 <- numerator.one2 - numerator.two2
  denominator2 <- W.centered2^2 
  
  # denominator simplifies to this.


  eta <- list(numerator1 = numerator1, denominator1 = denominator1,numerator2 = numerator2, denominator2 = denominator2,
       numerator.one1 = numerator.one1, numerator.two1 = numerator.two1, numerator.one2 = numerator.one2, numerator.two2 = numerator.two2,
       C.Y.hat = C.Y.hat)

  validate_observations(eta[["numerator1"]], X)
  validate_observations(eta[["denominator1"]], X)

  validate_observations(eta[["numerator2"]], X)
  validate_observations(eta[["denominator2"]], X)
  
  source("~/Desktop/학교/공모전/konkuk_lab/compriskCRgrf/input_utilities.R")
  # ================================================================================#
  data <- create_train_multi_matrices(X,
                                treatment1 = W.centered1,
                                treatment2  = W.centered2,
                                survival.numerator1 = eta[["numerator1"]],
                                survival.denominator1 = eta[["denominator1"]],
                                survival.numerator2 = eta[["numerator2"]],
                                survival.denominator2 = eta[["denominator2"]],
                                censor = D,
                                sample.weights = sample.weights)#create_train_muti_matrices로 변경
  #========
  
  #create_train_matrices 기존 -> 위의 매개변수를 받을 수 있도록 수정
  #function(X,
  #         outcome = NULL,
  #         treatment = NULL,
  #         instrument = NULL,
  #         survival.numerator = NULL,
  #         survival.denominator = NULL,
  #         status = NULL,
  #         sample.weights = FALSE) {
  
  #========
  data
  sigma_ = matrix()

  args <- list(num.trees = num.trees,
               clusters = clusters,
               samples.per.cluster = samples.per.cluster,
               sample.fraction = sample.fraction,
               mtry = mtry,
               min.node.size = min.node.size,
               honesty = honesty,
               honesty.fraction = honesty.fraction,
               honesty.prune.leaves = honesty.prune.leaves,
               alpha = alpha,
               imbalance.penalty = imbalance.penalty,
               stabilize.splits = stabilize.splits,
               ci.group.size = ci.group.size,
               compute.oob.predictions = compute.oob.predictions,
               num.threads = num.threads,
               seed = seed, mahalanobis, sigma = sigma_)

  #forest <- do.call.rcpp(causal_survival_train, c(data, args))#causal_survival_train -> muti_causal_survival_train 으로 구현
  forest <- do.call.rcpp(multi_causal_survival_train, c(data, args))
  class(forest) <- c("multi_causal_survival_forest", "grf")
  forest[["seed"]] <- seed
  forest[["_eta"]] <- eta
  forest[["X.orig"]] <- X
  forest[["Y.orig"]] <- Y
  forest[["W.orig"]] <- W
  forest[["D.orig"]] <- D
  forest[["Y.hat"]] <- Y.hat
  forest[["W.hat"]] <- W.hat
  forest[["sample.weights"]] <- sample.weights
  forest[["clusters"]] <- clusters
  forest[["equalize.cluster.weights"]] <- equalize.cluster.weights
  forest[["has.missing.values"]] <- has.missing.values
  #forest[["target"]] <- target
  forest[["horizon"]] <- horizon

  forest

#===============  
  
### Buckley-James ###
  numerator.one_bj <- (D * (fY - Y.hat) + (1 - D) * (Q.Y.hat - Y.hat)) * W.centered
  numerator_bj <- numerator.one_bj
  denominator_bj <- W.centered^2 # denominator simplifies to this.


  eta_bj <- list(numerator = numerator_bj, denominator = denominator_bj,
       numerator.one = numerator.one_bj, numerator.two = 0)

  validate_observations(eta_bj[["numerator"]], X)
  validate_observations(eta_bj[["denominator"]], X)

  data_bj <- create_train_matrices(X,
                                treatment = W.centered,
                                survival.numerator = eta_bj[["numerator"]],
                                survival.denominator = eta_bj[["denominator"]],
                                censor = D,
                                sample.weights = sample.weights)
  sigma_bj = matrix()

  args_bj <- list(num.trees = num.trees,
               clusters = clusters,
               samples.per.cluster = samples.per.cluster,
               sample.fraction = sample.fraction,
               mtry = mtry,
               min.node.size = min.node.size,
               honesty = honesty,
               honesty.fraction = honesty.fraction,
               honesty.prune.leaves = honesty.prune.leaves,
               alpha = alpha,
               imbalance.penalty = imbalance.penalty,
               stabilize.splits = stabilize.splits,
               ci.group.size = ci.group.size,
               compute.oob.predictions = compute.oob.predictions,
               num.threads = num.threads,
               seed = seed, mahalanobis, sigma = sigma_bj)

  forest_bj <- do.call.rcpp(causal_survival_train, c(data_bj, args_bj)) #수정
  class(forest_bj) <- c("causal_survival_forest", "grf")
  forest_bj[["seed"]] <- seed
  forest_bj[["_eta"]] <- eta_bj
  forest_bj[["X.orig"]] <- X
  forest_bj[["Y.orig"]] <- Y
  forest_bj[["W.orig"]] <- W
  forest_bj[["D.orig"]] <- D
  forest_bj[["Y.hat"]] <- Y.hat
  forest_bj[["W.hat"]] <- W.hat
  forest_bj[["sample.weights"]] <- sample.weights
  forest_bj[["clusters"]] <- clusters
  forest_bj[["equalize.cluster.weights"]] <- equalize.cluster.weights
  forest_bj[["has.missing.values"]] <- has.missing.values
  #forest[["target"]] <- target
  forest_bj[["horizon"]] <- horizon

  forest_bj


