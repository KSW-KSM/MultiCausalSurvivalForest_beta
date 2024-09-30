#ifndef GRF_MULTICAUSALSURVIVALSPITTINGRULE_H
#define GRF_MULTICAUSALSURVIVALSPITTINGRULE_H

#include "splitting/SplittingRule.h"
#include <vector>

namespace grf {

class MultiCausalSurvivalSplittingRule final: public SplittingRule {
public:
  MultiCausalSurvivalSplittingRule(size_t max_num_unique_values,
                                   uint min_node_size,
                                   double alpha,
                                   double imbalance_penalty,
                                   size_t num_treatments);

  ~MultiCausalSurvivalSplittingRule();

  bool find_best_split(const Data& data,
                       size_t node,
                       const std::vector<size_t>& possible_split_vars,
                       const Eigen::ArrayXXd& responses_by_sample,
                       const std::vector<std::vector<size_t>>& samples,
                       std::vector<size_t>& split_vars,
                       std::vector<double>& split_values,
                       std::vector<bool>& send_missing_left,
                       bool mahalanobis, Eigen::MatrixXd sigma);

private:
  void find_best_split_value(const Data& data,
                             size_t node,
                             size_t var,
                             size_t num_samples,
                             double weight_sum_node,
                             const Eigen::VectorXd& sum_node,
                             const Eigen::VectorXd& mean_node_z,
                             const Eigen::VectorXd& num_node_small_z,
                             const Eigen::VectorXd& sum_node_z,
                             const Eigen::VectorXd& sum_node_z_squared,
                             size_t num_failures_node,
                             double min_child_size,
                             size_t min_child_size_survival,
                             double& best_value,
                             size_t& best_var,
                             double& best_decrease,
                             bool& best_send_missing_left,
                             const Eigen::ArrayXXd& responses_by_sample,
                             const std::vector<std::vector<size_t>>& samples,
                             bool mahalanobis, Eigen::MatrixXd sigma);

      double compute_decrease(
                            double weight_sum_node,
                            const Eigen::VectorXd& sum_node,
                            const Eigen::VectorXd& sum_node_z,
                            const Eigen::VectorXd& sum_node_z_squared,
                            double weight_sum_left,
                            const Eigen::VectorXd& sum_left,
                            const Eigen::VectorXd& sum_left_z,
                            const Eigen::VectorXd& sum_left_z_squared,
                            size_t num_treatments,
                            bool mahalanobis,
    const Eigen::MatrixXd& sigma);

  size_t* counter;
  double* weight_sums;
  Eigen::VectorXd* sums;
  size_t* num_small_z;
  Eigen::VectorXd* sums_z;
  Eigen::VectorXd* sums_z_squared;
  size_t* failure_count;

  uint min_node_size;
  double alpha;
  double imbalance_penalty;
  size_t num_treatments;
};

} // namespace grf

#endif //GRF_MULTICAUSALSURVIVALSPITTINGRULE_H