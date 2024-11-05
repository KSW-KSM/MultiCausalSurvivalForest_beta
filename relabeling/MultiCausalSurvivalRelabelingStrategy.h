#ifndef GRF_MULTICAUSALSURVIVALRELABELINGSTRATEGY_H
#define GRF_MULTICAUSALSURVIVALRELABELINGSTRATEGY_H

#include <vector>


#include "RelabelingStrategy.h"
#include "../commons/Data.h"

namespace grf {
class MultiCausalSurvivalRelabelingStrategy final: public RelabelingStrategy {
public:
  bool relabel(
      const std::vector<size_t>& samples,
      const Data& data,
      Eigen::ArrayXXd& responses_by_sample) const;

private:
  void compute_pseudo_outcomes(
      const Data& data,
      const std::vector<size_t>& samples,
      std::vector<double>& pseudo_outcomes) const;
};

} // namespace grf

#endif // GRF_MULTICAUSALSURVIVALRELABELINGSTRATEGY_H
