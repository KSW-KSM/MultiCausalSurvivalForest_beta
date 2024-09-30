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

#ifndef GRF_DATA_H_
#define GRF_DATA_H_

#include <set>
#include <vector>

#include "../Eigen/Dense"

#include "globals.h"
#include "../optional/optional.hpp"

namespace grf {

/**
 * Data wrapper for GRF.
 * Serves as a read-only (immutable) wrapper of a column major (Fortran order)
 * array accessed through its pointer (data_ptr). This class does not own
 * data.
 *
 * The GRF data model is a contiguous array [X, Y, z, ...] of covariates X,
 * outcomes Y, and other optional variables z.
 *
 */
class Data {
public:
  Data(const double* data_ptr, size_t num_rows, size_t num_cols);
  Data(const std::vector<double>& data, size_t num_rows, size_t num_cols);
  Data(const std::pair<std::vector<double>, std::vector<size_t>>& data);

  /*추가*/

  // 인라인 함수로 정의
  double get_causal_survival_numerator(size_t row) const {
    return get(row, causal_survival_numerator_index.value());
  }

  double get_causal_survival_denominator(size_t row) const {
    return get(row, causal_survival_denominator_index.value());
  }

  // 선언만 남기고 정의는 Data.cpp로 이동
  void set_causal_survival_numerator_index(size_t index);
  void set_causal_survival_denominator_index(size_t index);

  // 다중 처리를 위한 새 메소드 추가
  double get_multi_causal_survival_numerator(size_t row, size_t treatment) const {
    return get(row, multi_causal_survival_numerator_index[treatment][0]);
  }

  double get_multi_causal_survival_denominator(size_t row, size_t treatment) const {
    return get(row, multi_causal_survival_denominator_index[treatment][0]);
  }

  // 다중 처리를 위한 메소드는 그대로 유지
  void set_multi_causal_survival_numerator_index(const std::vector<std::vector<size_t>>& index) {
    this->multi_causal_survival_numerator_index = index;
  }

  void set_multi_causal_survival_denominator_index(const std::vector<std::vector<size_t>>& index) {
    this->multi_causal_survival_denominator_index = index;
  }
  // 중복 선언 제거
  // size_t get_num_treatments() const;
  // double get_causal_survival_numerator(size_t row) const;
  // double get_causal_survival_denominator(size_t row) const;

  void set_outcome_index(size_t index);

  void set_outcome_index(const std::vector<size_t>& index);

  void set_treatment_index(size_t index);

  void set_treatment_index(const std::vector<size_t>& index);

  void set_instrument_index(size_t index);

  void set_weight_index(size_t index);

  void set_status_index(size_t index);

  void set_status_max(size_t value);

  /**
   * Sorts and gets the unique values in `samples` at variable `var`.
   *
   * @param all_values: the unique values in sorted order (filled in place).
   * @param sorted_samples: the sample IDs in sorted order (filled in place).
   * @param samples: the samples to sort.
   * @param var: the feature variable.
   * @return: (optional) the index (arg sort) of `sorted_samples` (integers from 0,...,samples.size() - 1).
   *
   * If all the values in `samples` is unique, then `all_values` and `sorted_samples`
   * have the same length.
   *
   * If any of the covariates are NaN, they will be placed first in the returned sort order.
   */
  std::vector<size_t> get_all_values(std::vector<double>& all_values,
                                     std::vector<size_t>& sorted_samples,
                                     const std::vector<size_t>& samples, size_t var) const;

  size_t get_num_cols() const;

  size_t get_num_rows() const;

  size_t get_num_outcomes() const;

  size_t get_num_treatments() const;

  const std::set<size_t>& get_disallowed_split_variables() const;

  double get_outcome(size_t row) const;

  Eigen::VectorXd get_outcomes(size_t row) const;

  double get_treatment(size_t row) const;

  Eigen::VectorXd get_treatments(size_t row) const;

  double get_instrument(size_t row) const;

  double get_weight(size_t row) const;

  size_t get_status(size_t row) const;

  size_t get_status_max() const;

  bool is_failure(size_t row) const;

  bool is_failure_status(size_t row, size_t status) const;

  double get(size_t row, size_t col) const;

  // 새로운 멤버 함수 선언 추가

private:
  const double* data_ptr;
  size_t num_rows;
  size_t num_cols;

  std::set<size_t> disallowed_split_variables;
  std::vector<std::vector<size_t>> multi_causal_survival_numerator_index; // 다중 처리를 위한 새 변수
  std::vector<std::vector<size_t>> multi_causal_survival_denominator_index; // 다중 처리를 위한 새 변수
  nonstd::optional<std::vector<size_t>> outcome_index;
  nonstd::optional<std::vector<size_t>> treatment_index;
  nonstd::optional<size_t> instrument_index;
  nonstd::optional<size_t> weight_index;
  nonstd::optional<size_t> causal_survival_numerator_index;
  nonstd::optional<size_t> causal_survival_denominator_index;
  nonstd::optional<size_t> status_index;
  nonstd::optional<size_t> status_max;
  size_t num_treatments; // num_treatments 멤버 변수 추가
};

// 인라인 함수 정의 제거

} // namespace grf
#endif /* GRF_DATA_H_ */
