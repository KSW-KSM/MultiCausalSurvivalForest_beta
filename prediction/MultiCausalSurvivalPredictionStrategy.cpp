#include <cmath>
#include <vector>

#include "../commons/utility.h"
#include "../Eigen/Dense"
#include "../commons/Data.h"
#include "MultiCausalSurvivalPredictionStrategy.h"



namespace grf {
void MultiCausalSurvivalPredictionStrategy::set_num_treatments(size_t num_treatments) {
  this->num_treatments = num_treatments;
}

size_t MultiCausalSurvivalPredictionStrategy::prediction_length() const {
  return num_treatments;
}

std::vector<double> MultiCausalSurvivalPredictionStrategy::predict(const std::vector<double>& average) const {
  std::vector<double> predictions(num_treatments);
  for (size_t i = 0; i < num_treatments; ++i) {
    predictions[i] = average[i * 2] / average[i * 2 + 1];
  }
  return predictions;
}

std::vector<double> MultiCausalSurvivalPredictionStrategy::compute_variance(
    const std::vector<double>& average,
    const PredictionValues& leaf_values,
    size_t ci_group_size) const {

  std::vector<double> variance_estimates(num_treatments);

  for (size_t t = 0; t < num_treatments; ++t) {
    double v_est = average[t * 2 + 1];
    double average_eta = average[t * 2] / average[t * 2 + 1];

    double num_good_groups = 0;
    double psi_squared = 0;
    double psi_grouped_squared = 0;

    for (size_t group = 0; group < leaf_values.get_num_nodes() / ci_group_size; ++group) {
      bool good_group = true;
      for (size_t j = 0; j < ci_group_size; ++j) {
        if (leaf_values.empty(group * ci_group_size + j)) {
          good_group = false;
          break;
        }
      }
      if (!good_group) continue;

      num_good_groups++;

      double group_psi = 0;

      for (size_t j = 0; j < ci_group_size; ++j) {
        size_t i = group * ci_group_size + j;
        const std::vector<double>& leaf_value = leaf_values.get_values(i);

        double psi_1 = leaf_value[t * 2] - leaf_value[t * 2 + 1] * average_eta;

        psi_squared += psi_1 * psi_1;
        group_psi += psi_1;
      }

      group_psi /= ci_group_size;
      psi_grouped_squared += group_psi * group_psi;
    }

    double var_between = psi_grouped_squared / num_good_groups;
    double var_total = psi_squared / (num_good_groups * ci_group_size);

    double group_noise = (var_total - var_between) / (ci_group_size - 1); //빼자

    double var_debiased = bayes_debiaser.debias(var_between, group_noise, num_good_groups);

    variance_estimates[t] = var_debiased / (v_est * v_est);
  }

  return variance_estimates;
}

size_t MultiCausalSurvivalPredictionStrategy::prediction_value_length() const {
  return num_treatments * 2;
}

PredictionValues MultiCausalSurvivalPredictionStrategy::precompute_prediction_values(
    const std::vector<std::vector<size_t>>& leaf_samples,
    const Data& data) const {
  size_t num_leaves = leaf_samples.size();

  std::vector<std::vector<double>> values(num_leaves);

  for (size_t i = 0; i < leaf_samples.size(); ++i) {
    size_t leaf_size = leaf_samples[i].size();
    if (leaf_size == 0) {
      continue;
    }

    std::vector<double> numerator_sums(num_treatments, 0);
    std::vector<double> denominator_sums(num_treatments, 0);
    double sum_weight = 0;

    for (auto& sample : leaf_samples[i]) {
      double weight = data.get_weight(sample);
      for (size_t t = 0; t < num_treatments; ++t) {
        numerator_sums[t] += weight * data.get_multi_causal_survival_numerator(sample, t);
        denominator_sums[t] += weight * data.get_multi_causal_survival_denominator(sample, t);
      }
      sum_weight += weight;
    }

    if (std::abs(sum_weight) <= 1e-16) {
      continue;
    }
    std::vector<double>& value = values[i];
    value.resize(num_treatments * 2);

    for (size_t t = 0; t < num_treatments; ++t) {
      value[t * 2] = numerator_sums[t] / leaf_size;
      value[t * 2 + 1] = denominator_sums[t] / leaf_size;
    }
  }

  return PredictionValues(values, num_treatments * 2);
}

std::vector<std::pair<double, double>> MultiCausalSurvivalPredictionStrategy::compute_error(
    size_t sample,
    const std::vector<double>& average,
    const PredictionValues& leaf_values,
    const Data& data) const {
  std::vector<std::pair<double, double>> errors(num_treatments, std::make_pair(NAN, NAN));
  return errors;
}

} // namespace grf