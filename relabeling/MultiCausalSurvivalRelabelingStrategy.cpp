#include "../commons/utility.h"
#include "MultiCausalSurvivalRelabelingStrategy.h"
#include "../commons/Data.h"

namespace grf {

bool MultiCausalSurvivalRelabelingStrategy::relabel(
    const std::vector<size_t>& samples,
    const Data& data,
    Eigen::ArrayXXd& responses_by_sample) const {

  size_t num_treatments = data.get_num_treatments();

  // Prepare the relevant averages.
  std::vector<double> numerator_sum(num_treatments, 0.0);
  std::vector<double> denominator_sum(num_treatments, 0.0);
  double sum_weight = 0.0;

  for (size_t sample : samples) {
    double sample_weight = data.get_weight(sample);
    for (size_t t = 0; t < num_treatments; t++) {
      numerator_sum[t] += sample_weight * data.get_multi_causal_survival_numerator(sample, t);
      denominator_sum[t] += sample_weight * data.get_multi_causal_survival_denominator(sample, t);
    }
    sum_weight += sample_weight;
  }

  if (std::abs(sum_weight) <= 1e-16) {
    return true;
  }

  std::vector<double> eta(num_treatments);
  bool all_zero = true;
  for (size_t t = 0; t < num_treatments; t++) {
    if (equal_doubles(denominator_sum[t], 0.0, 1.0e-10)) {
      eta[t] = 0.0;
    } else {
      eta[t] = numerator_sum[t] / denominator_sum[t];
      all_zero = false;
    }
  }

  if (all_zero) {
    return true;
  }

  // Create the new outcomes.
  for (size_t sample : samples) {
    for (size_t t = 0; t < num_treatments; t++) {
      if (equal_doubles(denominator_sum[t], 0.0, 1.0e-10)) {
        responses_by_sample(sample, t) = 0.0;
      } else {
        double response = (data.get_multi_causal_survival_numerator(sample, t) -
          data.get_multi_causal_survival_denominator(sample, t) * eta[t]) / denominator_sum[t];
        responses_by_sample(sample, t) = response;
      }
    }
  }
  return false;
}

} // namespace grf