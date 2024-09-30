#ifndef GRF_MULTICAUSALSURVIVALSPITTINGRULEFACTORY_H
#define GRF_MULTICAUSALSURVIVALSPITTINGRULEFACTORY_H

#include "SplittingRuleFactory.h"

namespace grf {

class MultiCausalSurvivalSplittingRuleFactory final: public SplittingRuleFactory {
public:
  MultiCausalSurvivalSplittingRuleFactory() = default;
  std::unique_ptr<SplittingRule> create(size_t max_num_unique_values,
                                        const TreeOptions& options) const;
};

} // namespace grf

#endif //GRF_MULTICAUSALSURVIVALSPITTINGRULEFACTORY_H