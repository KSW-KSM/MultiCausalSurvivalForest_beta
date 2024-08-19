/*-------------------------------------------------------------------------------
  This file is part of generalized random forest (grf).

  grf is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  grf is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with grf. If not, see <http://www.gnu.org/licenses/>.
 #-------------------------------------------------------------------------------*/

#include "ForestPredictors.h"
#include "../prediction/RegressionPredictionStrategy.h"
#include "../prediction/MultiRegressionPredictionStrategy.h"
#include "../prediction/SurvivalPredictionStrategy.h"
#include "../prediction/CausalSurvivalPredictionStrategy.h"

namespace grf {

ForestPredictor regression_predictor(uint num_threads) {
	num_threads = ForestOptions::validate_num_threads(num_threads);
	std::unique_ptr<OptimizedPredictionStrategy> prediction_strategy(new RegressionPredictionStrategy());
	return ForestPredictor(num_threads, std::move(prediction_strategy));
}

ForestPredictor multi_regression_predictor(uint num_threads, size_t num_outcomes) {
  num_threads = ForestOptions::validate_num_threads(num_threads);
  std::unique_ptr<OptimizedPredictionStrategy> prediction_strategy(new MultiRegressionPredictionStrategy(num_outcomes));
  return ForestPredictor(num_threads, std::move(prediction_strategy));
}

ForestPredictor survival_predictor(uint num_threads, size_t num_failures, int prediction_type, size_t status_max) {
	num_threads = ForestOptions::validate_num_threads(num_threads);
	std::unique_ptr<DefaultPredictionStrategy> prediction_strategy(
		new SurvivalPredictionStrategy(num_failures, prediction_type, status_max));
	return ForestPredictor(num_threads, std::move(prediction_strategy));
}

ForestPredictor causal_survival_predictor(uint num_threads) {
	num_threads = ForestOptions::validate_num_threads(num_threads);
	std::unique_ptr<OptimizedPredictionStrategy> prediction_strategy(new CausalSurvivalPredictionStrategy());
	return ForestPredictor(num_threads, std::move(prediction_strategy));
}

} // namespace grf
