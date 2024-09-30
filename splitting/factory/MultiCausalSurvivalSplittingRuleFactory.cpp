#include "MultiCausalSurvivalSplittingRuleFactory.h"
#include "../MultiCausalSurvivalSplittingRule.h"

namespace grf {

std::unique_ptr<SplittingRule> MultiCausalSurvivalSplittingRuleFactory::create(size_t max_num_unique_values,
                                                                               const TreeOptions& options) const {
  return std::unique_ptr<SplittingRule>(new MultiCausalSurvivalSplittingRule(
      max_num_unique_values,
      options.get_min_node_size(),
      options.get_alpha(),
      options.get_imbalance_penalty(),
      options.get_num_treatments()));
}

} // namespace grf