#ifndef GRF_MULTICAUSALSURVIVALPREDICTIONSTRATEGY_H
#define GRF_MULTICAUSALSURVIVALPREDICTIONSTRATEGY_H

#include <cstddef>
#include <vector>


#include "Prediction.h"
#include "OptimizedPredictionStrategy.h"
#include "PredictionValues.h"
#include "ObjectiveBayesDebiaser.h"


namespace grf {
class Data;

class MultiCausalSurvivalPredictionStrategy final: public OptimizedPredictionStrategy {
public:
  void set_num_treatments(size_t num_treatments);

  size_t prediction_value_length() const;
  PredictionValues precompute_prediction_values(
      const std::vector<std::vector<size_t>>& leaf_samples,
      const Data& data) const;

  size_t prediction_length() const;

  std::vector<double> predict(const std::vector<double>& average) const;

  std::vector<double> compute_variance(const std::vector<double>& average,
                          const PredictionValues& leaf_values,
                          size_t ci_group_size) const;

  std::vector<std::pair<double, double>> compute_error(
      size_t sample,
      const std::vector<double>& average,
      const PredictionValues& leaf_values,
      const Data& data) const;

private:
  size_t num_treatments;
  ObjectiveBayesDebiaser bayes_debiaser;
};

} // namespace grf

#endif //GRF_MULTICAUSALSURVIVALPREDICTIONSTRATEGY_H