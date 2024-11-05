#include "MultiCausalSurvivalSplittingRule.h"
#include <algorithm>
#include <cmath>
#include "Rcpp.h"

namespace grf {

MultiCausalSurvivalSplittingRule::MultiCausalSurvivalSplittingRule(size_t max_num_unique_values,
                                                                   uint min_node_size,
                                                                   double alpha,
                                                                   double imbalance_penalty,
                                                                   size_t num_treatments):
    min_node_size(min_node_size),
    alpha(alpha),
    imbalance_penalty(imbalance_penalty),
    num_treatments(num_treatments) {
    
    Rcpp::Rcout << "\n=== Initializing MultiCausalSurvivalSplittingRule ===" << std::endl;
    Rcpp::Rcout << "Parameters:" << std::endl;
    Rcpp::Rcout << "- max_num_unique_values: " << max_num_unique_values << std::endl;
    Rcpp::Rcout << "- min_node_size: " << min_node_size << std::endl;
    Rcpp::Rcout << "- alpha: " << alpha << std::endl;
    Rcpp::Rcout << "- imbalance_penalty: " << imbalance_penalty << std::endl;
    Rcpp::Rcout << "- num_treatments: " << num_treatments << std::endl;

    try {
        Rcpp::Rcout << "\nAllocating memory..." << std::endl;
        
        // 메모리 할당
        this->counter = new size_t[max_num_unique_values];
        Rcpp::Rcout << "- Allocated counter array" << std::endl;
        
        this->weight_sums = new double[max_num_unique_values];
        Rcpp::Rcout << "- Allocated weight_sums array" << std::endl;
        
        this->sums = new Eigen::VectorXd[max_num_unique_values];
        Rcpp::Rcout << "- Allocated sums array" << std::endl;
        
        this->num_small_z = new size_t[max_num_unique_values];
        Rcpp::Rcout << "- Allocated num_small_z array" << std::endl;
        
        this->sums_z = new Eigen::VectorXd[max_num_unique_values];
        Rcpp::Rcout << "- Allocated sums_z array" << std::endl;
        
        this->sums_z_squared = new Eigen::VectorXd[max_num_unique_values];
        Rcpp::Rcout << "- Allocated sums_z_squared array" << std::endl;
        
        this->failure_count = new size_t[max_num_unique_values];
        Rcpp::Rcout << "- Allocated failure_count array" << std::endl;

        Rcpp::Rcout << "\nInitializing vectors..." << std::endl;
        try { //메모리 할당 이슈발생으로 터미널 세션이 끊어짐
            // 한 번에 하나의 벡터만 초기화
            for (size_t i = 0; i < max_num_unique_values; ++i) {
                sums[i] = Eigen::VectorXd::Zero(num_treatments);
                if (i % 1000 == 0) {
                    Rcpp::Rcout << "Initialized sums[" << i << "]" << std::endl;
                }
            }
            Rcpp::Rcout << "Sums vectors initialized" << std::endl;
            
            for (size_t i = 0; i < max_num_unique_values; ++i) {
                sums_z[i] = Eigen::VectorXd::Zero(num_treatments);
                if (i % 1000 == 0) {
                    Rcpp::Rcout << "Initialized sums_z[" << i << "]" << std::endl;
                }
            }
            Rcpp::Rcout << "Sums_z vectors initialized" << std::endl;
            
            for (size_t i = 0; i < max_num_unique_values; ++i) {
                sums_z_squared[i] = Eigen::VectorXd::Zero(num_treatments);
                if (i % 1000 == 0) {
                    Rcpp::Rcout << "Initialized sums_z_squared[" << i << "]" << std::endl;
                }
            }
            Rcpp::Rcout << "Sums_z_squared vectors initialized" << std::endl;
            
        } catch (const std::bad_alloc& e) {
            Rcpp::Rcout << "Memory allocation failed: " << e.what() << std::endl;
            Rcpp::Rcout << "max_num_unique_values: " << max_num_unique_values << std::endl;
            Rcpp::Rcout << "num_treatments: " << num_treatments << std::endl;
            throw;
        } catch (const std::exception& e) {
            Rcpp::Rcout << "Error during vector initialization: " << e.what() << std::endl;
            throw;
        }
        Rcpp::Rcout << "Vectors initialized successfully" << std::endl;
        
        Rcpp::Rcout << "=== Initialization Completed ===\n" << std::endl;
        
    } catch (const std::exception& e) {
        Rcpp::Rcout << "Error during initialization: " << e.what() << std::endl;
        throw;
    } catch (...) {
        Rcpp::Rcout << "Unknown error during initialization" << std::endl;
        throw;
    }
}

MultiCausalSurvivalSplittingRule::~MultiCausalSurvivalSplittingRule() {
    Rcpp::Rcout << "\n=== Cleaning up MultiCausalSurvivalSplittingRule ===" << std::endl;
    
    try {
        // 메모리 해제
        if (counter != nullptr) {
            delete[] counter;
            Rcpp::Rcout << "- Freed counter array" << std::endl;
            counter = nullptr;
        }
        
        if (weight_sums != nullptr) {
            delete[] weight_sums;
            Rcpp::Rcout << "- Freed weight_sums array" << std::endl;
            weight_sums = nullptr;
        }
        
        if (sums != nullptr) {
            delete[] sums;
            Rcpp::Rcout << "- Freed sums array" << std::endl;
            sums = nullptr;
        }
        
        if (sums_z != nullptr) {
            delete[] sums_z;
            Rcpp::Rcout << "- Freed sums_z array" << std::endl;
            sums_z = nullptr;
        }
        
        if (sums_z_squared != nullptr) {
            delete[] sums_z_squared;
            Rcpp::Rcout << "- Freed sums_z_squared array" << std::endl;
            sums_z_squared = nullptr;
        }
        
        if (num_small_z != nullptr) {
            delete[] num_small_z;
            Rcpp::Rcout << "- Freed num_small_z array" << std::endl;
            num_small_z = nullptr;
        }
        
        if (failure_count != nullptr) {
            delete[] failure_count;
            Rcpp::Rcout << "- Freed failure_count array" << std::endl;
            failure_count = nullptr;
        }
        
        Rcpp::Rcout << "=== Cleanup Completed ===\n" << std::endl;
        
    } catch (const std::exception& e) {
        Rcpp::Rcout << "Error during cleanup: " << e.what() << std::endl;
    } catch (...) {
        Rcpp::Rcout << "Unknown error during cleanup" << std::endl;
    }
}


bool MultiCausalSurvivalSplittingRule::find_best_split(const Data& data,
                                                       size_t node,
                                                       const std::vector<size_t>& possible_split_vars,
                                                       const Eigen::ArrayXXd& responses_by_sample,
                                                       const std::vector<std::vector<size_t>>& samples,
                                                       std::vector<size_t>& split_vars,
                                                       std::vector<double>& split_values,
                                                       std::vector<bool>& send_missing_left,
                                                       bool mahalanobis, Eigen::MatrixXd sigma) {
  size_t num_samples = samples[node].size();

  // 이 노드에 대한 관련 수량 계산
  double weight_sum_node = 0.0;
  Eigen::VectorXd sum_node = Eigen::VectorXd::Zero(num_treatments);
  Eigen::VectorXd sum_node_z = Eigen::VectorXd::Zero(num_treatments);
  Eigen::VectorXd sum_node_z_squared = Eigen::VectorXd::Zero(num_treatments);
  size_t num_failures_node = 0;
  for (auto& sample : samples[node]) {
    double sample_weight = data.get_weight(sample);
    weight_sum_node += sample_weight;
    for (size_t t = 0; t < num_treatments; ++t) {
      sum_node(t) += sample_weight * responses_by_sample(sample, t);

      double z = data.get_instrument(sample);
      sum_node_z(t) += sample_weight * z;
      sum_node_z_squared(t) += sample_weight * z * z;
    }

    if (data.is_failure(sample)) {
      num_failures_node++;
    }
  }

  // 각 처리에 대한 평균 z 계산
  Eigen::VectorXd mean_node_z = sum_node_z / weight_sum_node;

  // 각 처리에 대해 z가 평균보다 작은 샘플 수 계산
  Eigen::VectorXd num_node_small_z = Eigen::VectorXd::Zero(num_treatments);
  for (auto& sample : samples[node]) {
    double z = data.get_instrument(sample);
    for (size_t t = 0; t < num_treatments; ++t) {
      if (z < mean_node_z(t)) {
        num_node_small_z(t)++;
      }
    }
  }

  // 최소 자식 노드 크기 계산
  Eigen::VectorXd size_node = sum_node_z_squared - sum_node_z.cwiseProduct(sum_node_z) / weight_sum_node;
  double min_child_size = size_node.maxCoeff() * alpha;
  size_t min_child_size_survival = std::max<size_t>(static_cast<size_t>(std::ceil(num_samples * alpha)), 1uL);

  // 최상의 분할 변수 초기화
  size_t best_var = 0;
  double best_value = 0;
  double best_decrease = 0.0;
  bool best_send_missing_left = true;

  // 가능한 모든 분할 변수에 대해 최상의 분할 찾기
  for (auto& var : possible_split_vars) {
    find_best_split_value(data, node, var, num_samples, weight_sum_node, sum_node, mean_node_z, num_node_small_z,
                          sum_node_z, sum_node_z_squared, num_failures_node, min_child_size, min_child_size_survival,
                          best_value, best_var, best_decrease, best_send_missing_left, responses_by_sample, samples, mahalanobis, sigma);
  }

  // 좋은 분할을 찾지 못했다면 중지
  if (best_decrease <= 0.0) {
    return true;
  }

  // 최상의 값 저장
  split_vars[node] = best_var;
  split_values[node] = best_value;
  send_missing_left[node] = best_send_missing_left;
  return false;
}

void MultiCausalSurvivalSplittingRule::find_best_split_value(const Data& data,
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
                                                             bool mahalanobis, Eigen::MatrixXd sigma) {
    Rcpp::Rcout << "\n=== Finding Best Split Value for Variable " << var << " ===" << std::endl;
    Rcpp::Rcout << "Node: " << node << ", Samples: " << num_samples << std::endl;
    
    try {
        // 이 변수에 대한 모든 가능한 분할 값 얻기
        std::vector<double> possible_split_values;
        std::vector<size_t> sorted_samples;
        Rcpp::Rcout << "Getting all possible split values..." << std::endl;
        data.get_all_values(possible_split_values, sorted_samples, samples[node], var);

        Rcpp::Rcout << "Number of possible split values: " << possible_split_values.size() << std::endl;
        if (possible_split_values.size() < 2) {
            Rcpp::Rcout << "Insufficient unique values, skipping variable" << std::endl;
            return;
        }

        size_t num_splits = possible_split_values.size() - 1;
        Rcpp::Rcout << "Number of potential splits: " << num_splits << std::endl;

        // 카운터와 합계 초기화
        Rcpp::Rcout << "\nInitializing arrays..." << std::endl;
        std::fill(counter, counter + num_splits, 0);
        std::fill(weight_sums, weight_sums + num_splits, 0);
        for (size_t i = 0; i < num_splits; ++i) {
            sums[i].setZero();
            num_small_z[i] = 0;
            sums_z[i].setZero();
            sums_z_squared[i].setZero();
            failure_count[i] = 0;
        }
        Rcpp::Rcout << "Arrays initialized" << std::endl;

        Rcpp::Rcout << "\nInitializing missing value statistics..." << std::endl;
        size_t n_missing = 0;
        double weight_sum_missing = 0;
        Eigen::VectorXd sum_missing = Eigen::VectorXd::Zero(num_treatments);
        Eigen::VectorXd sum_z_missing = Eigen::VectorXd::Zero(num_treatments);
        Eigen::VectorXd sum_z_squared_missing = Eigen::VectorXd::Zero(num_treatments);
        size_t num_small_z_missing = 0;
        size_t num_failures_missing = 0;

        Rcpp::Rcout << "\nProcessing samples..." << std::endl;
        size_t split_index = 0;
        for (size_t i = 0; i < num_samples - 1; i++) {
            if (i % 1000 == 0) {
                Rcpp::Rcout << "Processing sample " << i << " of " << num_samples << std::endl;
            }
            
            size_t sample = sorted_samples[i];
            size_t next_sample = sorted_samples[i + 1];
            double sample_value = data.get(sample, var);
            double z = data.get_instrument(sample);
            double sample_weight = data.get_weight(sample);

            if (std::isnan(sample_value)) {
                Rcpp::Rcout << "Found missing value at index " << i << std::endl;
                weight_sum_missing += sample_weight;
                for (size_t t = 0; t < num_treatments; ++t) {
                    sum_missing(t) += sample_weight * responses_by_sample(sample, t);
                }
                ++n_missing;

                sum_z_missing += sample_weight * z * Eigen::VectorXd::Ones(num_treatments);
                sum_z_squared_missing += sample_weight * z * z * Eigen::VectorXd::Ones(num_treatments);
                if (z < mean_node_z.mean()) {
                    ++num_small_z_missing;
                }
                if (data.is_failure(sample)) {
                    num_failures_missing++;
                }
            } else {
                weight_sums[split_index] += sample_weight;
                for (size_t t = 0; t < num_treatments; ++t) {
                    sums[split_index](t) += sample_weight * responses_by_sample(sample, t);
                }
                ++counter[split_index];

                sums_z[split_index] += sample_weight * z * Eigen::VectorXd::Ones(num_treatments);
                sums_z_squared[split_index] += sample_weight * z * z * Eigen::VectorXd::Ones(num_treatments);
                if (z < mean_node_z.mean()) {
                    ++num_small_z[split_index];
                }
                if (data.is_failure(sample)) {
                    ++failure_count[split_index];
                }
            }

            double next_sample_value = data.get(next_sample, var);
            if (sample_value != next_sample_value && !std::isnan(next_sample_value)) {
                Rcpp::Rcout << "Moving to next split bucket at index " << i << std::endl;
                ++split_index;
            }
        }

        Rcpp::Rcout << "\nMissing value statistics:" << std::endl;
        Rcpp::Rcout << "- Number of missing values: " << n_missing << std::endl;
        Rcpp::Rcout << "- Missing weight sum: " << weight_sum_missing << std::endl;
        Rcpp::Rcout << "- Missing failures: " << num_failures_missing << std::endl;

        size_t n_left = n_missing;
        double weight_sum_left = weight_sum_missing;
        Eigen::VectorXd sum_left = sum_missing;
        Eigen::VectorXd sum_left_z = sum_z_missing;
        Eigen::VectorXd sum_left_z_squared = sum_z_squared_missing;
        size_t num_left_small_z = num_small_z_missing;
        size_t num_failures_left = num_failures_missing;

        // 각 가능한 분할에 대해 불순도 감소 계산
        Rcpp::Rcout << "\nEvaluating possible splits..." << std::endl;
        for (bool send_left : {true, false}) {
            Rcpp::Rcout << "\nEvaluating send_missing_left = " << send_left << std::endl;
            
            if (!send_left) {
                if (n_missing == 0) {
                    Rcpp::Rcout << "No missing values, skipping right-send evaluation" << std::endl;
                    break;
                }
                Rcpp::Rcout << "Resetting left statistics for right-send evaluation" << std::endl;
                n_left = 0;
                weight_sum_left = 0;
                sum_left.setZero();
                sum_left_z.setZero();
                sum_left_z_squared.setZero();
                num_left_small_z = 0;
                num_failures_left = 0;
            }

            for (size_t i = 0; i < num_splits; ++i) {
                if (i % 100 == 0) {
                    Rcpp::Rcout << "Evaluating split " << i << " of " << num_splits << std::endl;
                }

                if (i == 0 && !send_left) {
                    Rcpp::Rcout << "Skipping first split for right-send" << std::endl;
                    continue;
                }

                n_left += counter[i];
                num_left_small_z += num_small_z[i];
                weight_sum_left += weight_sums[i];
                sum_left += sums[i];
                sum_left_z += sums_z[i];
                sum_left_z_squared += sums_z_squared[i];
                num_failures_left += failure_count[i];

                if (num_failures_left < min_child_size_survival) {
                    Rcpp::Rcout << "Insufficient failures in left child, skipping" << std::endl;
                    continue;
                }

                size_t num_failures_right = num_failures_node - num_failures_left;
                if (num_failures_right < min_child_size_survival) {
                    Rcpp::Rcout << "Insufficient failures in right child, breaking" << std::endl;
                    break;
                }

                size_t num_left_large_z = n_left - num_left_small_z;
                if (num_left_small_z < min_node_size || num_left_large_z < min_node_size) {
                    Rcpp::Rcout << "Insufficient z values in left child, skipping" << std::endl;
                    continue;
                }

                size_t n_right = num_samples - n_left;
                size_t num_right_small_z = num_node_small_z.sum() - num_left_small_z;
                size_t num_right_large_z = n_right - num_right_small_z;
                if (num_right_small_z < min_node_size || num_right_large_z < min_node_size) {
                    Rcpp::Rcout << "Insufficient z values in right child, breaking" << std::endl;
                    break;
                }

                double decrease = compute_decrease(
                    weight_sum_node, sum_node, sum_node_z, sum_node_z_squared,
                    weight_sum_left, sum_left, sum_left_z, sum_left_z_squared,
                    num_treatments, mahalanobis, sigma
                );

                decrease -= imbalance_penalty * (1.0 / n_left + 1.0 / n_right);

                if (decrease > best_decrease) {
                    Rcpp::Rcout << "\nNew best split found:" << std::endl;
                    Rcpp::Rcout << "- Split value: " << possible_split_values[i] << std::endl;
                    Rcpp::Rcout << "- Decrease: " << decrease << std::endl;
                    Rcpp::Rcout << "- Left samples: " << n_left << std::endl;
                    Rcpp::Rcout << "- Right samples: " << n_right << std::endl;
                    
                    best_value = possible_split_values[i];
                    best_var = var;
                    best_decrease = decrease;
                    best_send_missing_left = send_left;
                }
            }
        }

        Rcpp::Rcout << "\n=== Split Value Evaluation Completed ===" << std::endl;
        Rcpp::Rcout << "Final best decrease: " << best_decrease << std::endl;
        
    } catch (const std::exception& e) {
        Rcpp::Rcout << "Error in find_best_split_value: " << e.what() << std::endl;
        throw;
    }
}

double MultiCausalSurvivalSplittingRule::compute_decrease(
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
    const Eigen::MatrixXd& sigma) {

    Rcpp::Rcout << "\n=== Computing Decrease ===" << std::endl;
    Rcpp::Rcout << "Initial parameters:" << std::endl;
    Rcpp::Rcout << "- Weight sum node: " << weight_sum_node << std::endl;
    Rcpp::Rcout << "- Weight sum left: " << weight_sum_left << std::endl;
    Rcpp::Rcout << "- Number of treatments: " << num_treatments << std::endl;
    Rcpp::Rcout << "- Using Mahalanobis: " << (mahalanobis ? "true" : "false") << std::endl;

    try {
        // 오른쪽 자식 노드 계산
        Rcpp::Rcout << "\nCalculating right child statistics..." << std::endl;
        double weight_sum_right = weight_sum_node - weight_sum_left;
        Eigen::VectorXd sum_right = sum_node - sum_left;
        Eigen::VectorXd sum_right_z = sum_node_z - sum_left_z;
        Eigen::VectorXd sum_right_z_squared = sum_node_z_squared - sum_left_z_squared;

        Rcpp::Rcout << "Right child weight sum: " << weight_sum_right << std::endl;

        // 평균 계산
        Rcpp::Rcout << "\nCalculating averages..." << std::endl;
        Eigen::VectorXd average_left = sum_left / weight_sum_left;
        Eigen::VectorXd average_right = sum_right / weight_sum_right;
        Eigen::VectorXd average_node = sum_node / weight_sum_node;

        Rcpp::Rcout << "Averages computed:" << std::endl;
        Rcpp::Rcout << "- Node average: " << average_node.transpose() << std::endl;
        Rcpp::Rcout << "- Left average: " << average_left.transpose() << std::endl;
        Rcpp::Rcout << "- Right average: " << average_right.transpose() << std::endl;

        // 분산 계산
        Rcpp::Rcout << "\nCalculating variances..." << std::endl;
        Eigen::VectorXd var_left = (sum_left_z_squared / weight_sum_left) - 
                                  (sum_left_z / weight_sum_left).array().square().matrix();
        Eigen::VectorXd var_right = (sum_right_z_squared / weight_sum_right) - 
                                   (sum_right_z / weight_sum_right).array().square().matrix();

        Rcpp::Rcout << "Variances computed:" << std::endl;
        Rcpp::Rcout << "- Left variance: " << var_left.transpose() << std::endl;
        Rcpp::Rcout << "- Right variance: " << var_right.transpose() << std::endl;

        double decrease = 0.0;

        // 거리 계산
        if (mahalanobis) {
            Rcpp::Rcout << "\nComputing Mahalanobis distance..." << std::endl;
            Eigen::VectorXd diff_left = average_left - average_node;
            Eigen::VectorXd diff_right = average_right - average_node;
            
            Rcpp::Rcout << "Differences from node mean:" << std::endl;
            Rcpp::Rcout << "- Left diff: " << diff_left.transpose() << std::endl;
            Rcpp::Rcout << "- Right diff: " << diff_right.transpose() << std::endl;
            
            Eigen::MatrixXd sigma_inv = sigma.inverse();
            Rcpp::Rcout << "Sigma inverse computed" << std::endl;
            
            double left_distance = (diff_left.transpose() * sigma_inv * diff_left).eval()(0);
            double right_distance = (diff_right.transpose() * sigma_inv * diff_right).eval()(0);
            
            decrease = weight_sum_left * left_distance + weight_sum_right * right_distance;
            
            Rcpp::Rcout << "Mahalanobis distances:" << std::endl;
            Rcpp::Rcout << "- Left distance: " << left_distance << std::endl;
            Rcpp::Rcout << "- Right distance: " << right_distance << std::endl;
            
        } else {
            Rcpp::Rcout << "\nComputing Euclidean distance..." << std::endl;
            double left_distance = (average_left - average_node).squaredNorm();
            double right_distance = (average_right - average_node).squaredNorm();
            
            decrease = weight_sum_left * left_distance + weight_sum_right * right_distance;
            
            Rcpp::Rcout << "Euclidean distances:" << std::endl;
            Rcpp::Rcout << "- Left distance: " << left_distance << std::endl;
            Rcpp::Rcout << "- Right distance: " << right_distance << std::endl;
        }

        Rcpp::Rcout << "\nRaw decrease before normalization: " << decrease << std::endl;

        // 분산 정규화
        double variance_normalization = (var_left.sum() / num_treatments + 
                                       var_right.sum() / num_treatments);
        Rcpp::Rcout << "Variance normalization factor: " << variance_normalization << std::endl;
        
        decrease /= variance_normalization;

        Rcpp::Rcout << "Final normalized decrease: " << decrease << std::endl;
        Rcpp::Rcout << "=== Decrease Computation Completed ===\n" << std::endl;

        return decrease;
        
    } catch (const std::exception& e) {
        Rcpp::Rcout << "Error in compute_decrease: " << e.what() << std::endl;
        throw;
    }
}


} // namespace grf