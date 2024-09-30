#include "MultiCausalSurvivalSplittingRule.h"
#include <algorithm>
#include <cmath>

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
  // 메모리 할당
  this->counter = new size_t[max_num_unique_values];
  this->weight_sums = new double[max_num_unique_values];
  this->sums = new Eigen::VectorXd[max_num_unique_values];
  this->num_small_z = new size_t[max_num_unique_values];
  this->sums_z = new Eigen::VectorXd[max_num_unique_values];
  this->sums_z_squared = new Eigen::VectorXd[max_num_unique_values];
  this->failure_count = new size_t[max_num_unique_values];

  // 벡터 초기화
  for (size_t i = 0; i < max_num_unique_values; ++i) {
    this->sums[i] = Eigen::VectorXd::Zero(num_treatments);
    this->sums_z[i] = Eigen::VectorXd::Zero(num_treatments);
    this->sums_z_squared[i] = Eigen::VectorXd::Zero(num_treatments);
  }
}

MultiCausalSurvivalSplittingRule::~MultiCausalSurvivalSplittingRule() {
  // 메모리 해제
  if (counter != nullptr) {
    delete[] counter;
  }
  if (weight_sums != nullptr) {
    delete[] weight_sums;
  }
  if (sums != nullptr) {
    delete[] sums;
  }
  if (sums_z != nullptr) {
    delete[] sums_z;
  }
  if (sums_z_squared != nullptr) {
    delete[] sums_z_squared;
  }
  if (num_small_z != nullptr) {
    delete[] num_small_z;
  }
  if (failure_count != nullptr) {
    delete[] failure_count;
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
  // 이 변수에 대한 모든 가능한 분할 값 얻기
  std::vector<double> possible_split_values;
  std::vector<size_t> sorted_samples;
  data.get_all_values(possible_split_values, sorted_samples, samples[node], var);

  // 이 변수에 대해 모든 값이 동일하다면 다음 변수로
  if (possible_split_values.size() < 2) {
    return;
  }

  size_t num_splits = possible_split_values.size() - 1;

  // 카운터와 합계 초기화
  std::fill(counter, counter + num_splits, 0);
  std::fill(weight_sums, weight_sums + num_splits, 0);
  for (size_t i = 0; i < num_splits; ++i) {
    sums[i].setZero();
    num_small_z[i] = 0;
    sums_z[i].setZero();
    sums_z_squared[i].setZero();
    failure_count[i] = 0;
  }

  size_t n_missing = 0;
  double weight_sum_missing = 0;
  Eigen::VectorXd sum_missing = Eigen::VectorXd::Zero(num_treatments);
  Eigen::VectorXd sum_z_missing = Eigen::VectorXd::Zero(num_treatments);
  Eigen::VectorXd sum_z_squared_missing = Eigen::VectorXd::Zero(num_treatments);
  size_t num_small_z_missing = 0;
  size_t num_failures_missing = 0;

  size_t split_index = 0;
  for (size_t i = 0; i < num_samples - 1; i++) {
    size_t sample = sorted_samples[i];
    size_t next_sample = sorted_samples[i + 1];
    double sample_value = data.get(sample, var);
    double z = data.get_instrument(sample);
    double sample_weight = data.get_weight(sample);

    if (std::isnan(sample_value)) {
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
    // 다음 샘플 값이 다르다면 (NaN에서 Xij로의 전이 포함) 다음 버킷으로 이동
    if (sample_value != next_sample_value && !std::isnan(next_sample_value)) {
      ++split_index;
    }
  }

  size_t n_left = n_missing;
  double weight_sum_left = weight_sum_missing;
  Eigen::VectorXd sum_left = sum_missing;
  Eigen::VectorXd sum_left_z = sum_z_missing;
  Eigen::VectorXd sum_left_z_squared = sum_z_squared_missing;
  size_t num_left_small_z = num_small_z_missing;
  size_t num_failures_left = num_failures_missing;

  // 각 가능한 분할에 대해 불순도 감소 계산
  for (bool send_left : {true, false}) {
    if (!send_left) {
      // NaN이 없는 정상적인 분할이므로 일찍 중단 가능
      if (n_missing == 0) {
        break;
      }
      // 누락된 부분이 전체 합에 포함되어 있으므로 n_right나 sum_right를 조정할 필요 없음
      n_left = 0;
      weight_sum_left = 0;
      sum_left.setZero();
      sum_left_z.setZero();
      sum_left_z_squared.setZero();
      num_left_small_z = 0;
      num_failures_left = 0;
    }

    for (size_t i = 0; i < num_splits; ++i) {
      // NaN에 대한 분할 시 오른쪽으로 보내는 것을 평가할 필요 없음
      if (i == 0 && !send_left) {
        continue;
      }

      n_left += counter[i];
      num_left_small_z += num_small_z[i];
      weight_sum_left += weight_sums[i];
      sum_left += sums[i];
      sum_left_z += sums_z[i];
      sum_left_z_squared += sums_z_squared[i];
      num_failures_left += failure_count[i];

      // 왼쪽 자식에 충분한 실패가 없다면 이 분할 건너뛰기
      if (num_failures_left < min_child_size_survival) {
        continue;
      }

      // 오른쪽 자식에 충분한 실패가 없다면 중단
      size_t num_failures_right = num_failures_node - num_failures_left;
      if (num_failures_right < min_child_size_survival) {
        break;
      }

      // 왼쪽 자식에 부모의 평균 위아래로 충분한 z 값이 없다면 이 분할 건너뛰기
      size_t num_left_large_z = n_left - num_left_small_z;
      if (num_left_small_z < min_node_size || num_left_large_z < min_node_size) {
        continue;
      }

      // 오른쪽 자식에 부모의 평균 위아래로 충분한 z 값이 없다면 중단
      size_t n_right = num_samples - n_left;
      size_t num_right_small_z = num_node_small_z.sum() - num_left_small_z;
      size_t num_right_large_z = n_right - num_right_small_z;
      if (num_right_small_z < min_node_size || num_right_large_z < min_node_size) {
        break;
      }

      // 불순도 감소 계산
      double decrease = compute_decrease(
        weight_sum_node, sum_node, sum_node_z, sum_node_z_squared,
        weight_sum_left, sum_left, sum_left_z, sum_left_z_squared,
        num_treatments, mahalanobis, sigma
      );

      // 불균형 페널티 적용
      decrease -= imbalance_penalty * (1.0 / n_left + 1.0 / n_right);

      // 현재까지의 최상의 감소보다 크다면 값 갱신
      if (decrease > best_decrease) {
        best_value = possible_split_values[i];
        best_var = var;
        best_decrease = decrease;
        best_send_missing_left = send_left;
      }
    }
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

  double weight_sum_right = weight_sum_node - weight_sum_left;
  Eigen::VectorXd sum_right = sum_node - sum_left;
  Eigen::VectorXd sum_right_z = sum_node_z - sum_left_z;
  Eigen::VectorXd sum_right_z_squared = sum_node_z_squared - sum_left_z_squared;

  // 각 처리에 대한 평균 계산
  Eigen::VectorXd average_left = sum_left / weight_sum_left;
  Eigen::VectorXd average_right = sum_right / weight_sum_right;
  Eigen::VectorXd average_node = sum_node / weight_sum_node;

  // 각 처리에 대한 분산 계산
  Eigen::VectorXd var_left = (sum_left_z_squared / weight_sum_left) - (sum_left_z / weight_sum_left).array().square().matrix();
  Eigen::VectorXd var_right = (sum_right_z_squared / weight_sum_right) - (sum_right_z / weight_sum_right).array().square().matrix();

  double decrease = 0.0;

  if (mahalanobis) {
    // Mahalanobis 거리 계산
    Eigen::VectorXd diff_left = average_left - average_node;
    Eigen::VectorXd diff_right = average_right - average_node;
    decrease = (weight_sum_left * diff_left.transpose() * sigma.inverse() * diff_left +
            weight_sum_right * diff_right.transpose() * sigma.inverse() * diff_right).eval()(0);
  } else {
    // 유클리드 거리 계산
    decrease = weight_sum_left * (average_left - average_node).squaredNorm() +
               weight_sum_right * (average_right - average_node).squaredNorm();
  }

  // 분산에 대한 정규화
  decrease /= (var_left.sum() / num_treatments + var_right.sum() / num_treatments);

  return decrease;
}


} // namespace grf