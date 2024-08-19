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

#include "MultiRegressionSplittingRuleFactory.h"
#include "../MultiRegressionSplittingRule.h"

namespace grf {

MultiRegressionSplittingRuleFactory::MultiRegressionSplittingRuleFactory(size_t num_outcomes):
  num_outcomes(num_outcomes) {}

std::unique_ptr<SplittingRule> MultiRegressionSplittingRuleFactory::create(size_t max_num_unique_values,
                                                                           const TreeOptions& options) const {
  return std::unique_ptr<SplittingRule>(new MultiRegressionSplittingRule(
      max_num_unique_values,
      options.get_alpha(),
      options.get_imbalance_penalty(),
      num_outcomes));
}

} // namespace grf