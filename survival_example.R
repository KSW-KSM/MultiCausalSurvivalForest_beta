data_file = paste0(default_dir, "/wihs2.csv")
#data_file = paste0(default_dir, "/wihs3.csv")
data = read.csv(data_file)
X = data[,3:6]
Y = data[['time']]
D = data[['status']]

#library(tidyverse)
#data(Melanoma, package = "MASS")
#Melanoma <- 
#  Melanoma %>% 
#  mutate(
#    status = as.factor(recode(status, `2` = 0, `1` = 1, `3` = 2))
#  )
#D = as.numeric(Melanoma[['status']])
#Y = Melanoma[['time']]
#X = Melanoma[3:7]
#X = X[-3]
#X = as.data.frame(X, ncol=1)

s.forest <- survival_forest(X, Y, D, honesty=FALSE)
s.pred <- predict(s.forest, X)
s.pred$predictions
