data_file = paste0(default_dir, "/dat_multiRF.csv")
data = read.csv(data_file)
X = data[3:12]
Y = data[1:2]
sigma = matrix(c(1,0,0,1), nrow=2, ncol=2)
forest = multi_regression_forest(X,Y,1, honesty=F)
new_data = matrix(c(0.791961346, -0.547185228, -0.597154391, -0.385432707, -0.660770696, 0.536028078, 0.283616233, -0.179933712, -0.020123282, -0.735890979), nrow=1, ncol=10)
outcome = predict.multi_regression_forest(forest, new_data)
outcome