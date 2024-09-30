default_dir = "/Users/seong-ugang/Desktop/학교/공모전/konkuk_lab/compriskCRgrf"
install.packages("Rcpp")

run <- function(dir = default_dir) {
	source(paste0(dir, "/input_utilities.R"))

	source(paste0(dir, "/average_treatment_effect.R"))
	source(paste0(dir, "/get_scores.R"))
	source(paste0(dir, "/forest_summary.R"))

	#source(paste0(dir, "/regression_forest.R"))
	#source(paste0(dir, "/multi_regression_forest.R"))
	#source(paste0(dir, "/survival_forest.R"))
	#source(paste0(dir, "/causal_survival_forest.R"))

	library(Rcpp)

	#sourceCpp(paste0(dir, "/", "RegressionForestBindings.cpp"))
	#sourceCpp(paste0(dir, "/", "MultiRegressionForestBindings.cpp"))
	#sourceCpp(paste0(dir, "/", "SurvivalForestBindings.cpp"))
	#sourceCpp(paste0(dir, "/", "CausalSurvivalForestBindings.cpp"))
	sourceCpp(paste0(dir, "/", "MultiCausalSurvivalForestBindings.cpp"))
}

run()

source(paste0(default_dir, "/survival_forest.R"))
sourceCpp(paste0(default_dir, "/", "SurvivalForestBindings.cpp"))

