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

#include "ForestTrainers.h"
#include "../prediction/RegressionPredictionStrategy.h"
#include "../prediction/MultiRegressionPredictionStrategy.h"
#include "../prediction/CausalSurvivalPredictionStrategy.h"

#include "../relabeling/NoopRelabelingStrategy.h"
#include "../relabeling/MultiNoopRelabelingStrategy.h"
#include "../relabeling/CausalSurvivalRelabelingStrategy.h"

#include "../splitting/factory/MultiRegressionSplittingRuleFactory.h"
#include "../splitting/factory/RegressionSplittingRuleFactory.h"
#include "../splitting/factory/SurvivalSplittingRuleFactory.h"
#include "../splitting/factory/CausalSurvivalSplittingRuleFactory.h"

namespace grf {

ForestTrainer regression_trainer() {
    std::unique_ptr<RelabelingStrategy> relabeling_strategy(new NoopRelabelingStrategy());
    std::unique_ptr<SplittingRuleFactory> splitting_rule_factory(new RegressionSplittingRuleFactory());
    std::unique_ptr<OptimizedPredictionStrategy> prediction_strategy(new RegressionPredictionStrategy());

    return ForestTrainer(std::move(relabeling_strategy),
        std::move(splitting_rule_factory),
        std::move(prediction_strategy));
}

ForestTrainer multi_regression_trainer(size_t num_outcomes) {
  std::unique_ptr<RelabelingStrategy> relabeling_strategy(new MultiNoopRelabelingStrategy(num_outcomes));
  std::unique_ptr<SplittingRuleFactory> splitting_rule_factory(new MultiRegressionSplittingRuleFactory(num_outcomes));
  std::unique_ptr<OptimizedPredictionStrategy> prediction_strategy(new MultiRegressionPredictionStrategy(num_outcomes));

  return ForestTrainer(std::move(relabeling_strategy),
                       std::move(splitting_rule_factory),
                       std::move(prediction_strategy));
}

ForestTrainer survival_trainer() {
    std::unique_ptr<RelabelingStrategy> relabeling_strategy(new NoopRelabelingStrategy());
    std::unique_ptr<SplittingRuleFactory> splitting_rule_factory(new SurvivalSplittingRuleFactory());

    return ForestTrainer(std::move(relabeling_strategy),
        std::move(splitting_rule_factory),
        nullptr);
}

ForestTrainer causal_survival_trainer(bool stabilize_splits) {

    std::unique_ptr<RelabelingStrategy> relabeling_strategy(new CausalSurvivalRelabelingStrategy());
    std::unique_ptr<SplittingRuleFactory> splitting_rule_factory = stabilize_splits
        ? std::unique_ptr<SplittingRuleFactory>(new CausalSurvivalSplittingRuleFactory())
        : std::unique_ptr<SplittingRuleFactory>(new RegressionSplittingRuleFactory());
    std::unique_ptr<OptimizedPredictionStrategy> prediction_strategy(new CausalSurvivalPredictionStrategy());

    return ForestTrainer(std::move(relabeling_strategy),
        std::move(splitting_rule_factory),
        std::move(prediction_strategy));
}

} // namespace grf
