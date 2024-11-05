/*-------------------------------------------------------------------------------
  This file is part of generalized-random-forest.

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

#include <algorithm>
#include <cmath>
#include <numeric>
#include <iterator>
#include <stdexcept>

#include "Data.h"

namespace grf {

Data::Data(const double* data_ptr, size_t num_rows, size_t num_cols) {
  if (data_ptr == nullptr) {
    throw std::runtime_error("Invalid data storage: nullptr");
  }
  this->data_ptr = data_ptr;
  this->num_rows = num_rows;
  this->num_cols = num_cols;
}

Data::Data(const std::vector<double>& data, size_t num_rows, size_t num_cols) :
  Data(data.data(), num_rows, num_cols) {}

Data::Data(const std::pair<std::vector<double>, std::vector<size_t>>& data) :
  Data(data.first.data(), data.second.at(0), data.second.at(1)) {}

void Data::set_outcome_index(size_t index) {
  set_outcome_index(std::vector<size_t>({index}));
}

void Data::set_outcome_index(const std::vector<size_t>& index) {
  this->outcome_index = index;
  disallowed_split_variables.insert(index.begin(), index.end());
}

void Data::set_treatment_index(size_t index) {
  set_treatment_index(std::vector<size_t>({index}));
}

void Data::set_treatment_index(const std::vector<size_t>& index) {
  this->treatment_index = index;
  disallowed_split_variables.insert(index.begin(), index.end());
}

void Data::set_instrument_index(size_t index) {
  this->instrument_index = index;
  disallowed_split_variables.insert(index);
}

void Data::set_weight_index(size_t index) {
  this->weight_index = index;
  disallowed_split_variables.insert(index);
}

void Data::set_causal_survival_numerator_index(size_t index) {
  this->causal_survival_numerator_index = index;
}

void Data::set_causal_survival_denominator_index(size_t index) {
  this->causal_survival_denominator_index = index;
}

void Data::set_status_index(size_t index) {
  this->status_index = index;
  disallowed_split_variables.insert(index);
}

void Data::set_status_max(size_t value) {
    this->status_max = value;
}

std::vector<size_t> Data::get_all_values(std::vector<double>& all_values,
                                         std::vector<size_t>& sorted_samples,
                                         const std::vector<size_t>& samples,
                                         size_t var) const {
  all_values.resize(samples.size());
  for (size_t i = 0; i < samples.size(); i++) {
    size_t sample = samples[i];
    all_values[i] = get(sample, var);
  }

  sorted_samples.resize(samples.size());
  std::vector<size_t> index(samples.size());
   // fill with [0, 1,..., samples.size() - 1]
  std::iota(index.begin(), index.end(), 0);
  // sort index based on the split values (argsort)
  // the NaN comparison places all NaNs at the beginning
  // stable sort is needed for consistent element ordering cross platform,
  // otherwise the resulting sums used in the splitting rules may compound rounding error
  // differently and produce different splits.
  std::stable_sort(index.begin(), index.end(), [&](const size_t& lhs, const size_t& rhs) {
    return all_values[lhs] < all_values[rhs] || (std::isnan(all_values[lhs]) && !std::isnan(all_values[rhs]));
  });

  for (size_t i = 0; i < samples.size(); i++) {
    sorted_samples[i] = samples[index[i]];
    all_values[i] = get(sorted_samples[i], var);
  }

  all_values.erase(unique(all_values.begin(), all_values.end(), [&](const double& lhs, const double& rhs) {
    return lhs == rhs || (std::isnan(lhs) && std::isnan(rhs));
  }), all_values.end());

  return index;
}

size_t Data::get_num_cols() const {
  return num_cols;
}

size_t Data::get_num_rows() const {
  return num_rows;
}

size_t Data::get_num_outcomes() const {
  if (outcome_index.has_value()) {
    return outcome_index.value().size();
  } else {
    return 1;
  }
}
/*
size_t Data::get_num_treatments() const {
  if (treatment_index.has_value()) {
    return treatment_index.value().size();
  } else {
    return 1;
  }
}*/

const std::set<size_t>& Data::get_disallowed_split_variables() const {
  return disallowed_split_variables;
}

size_t Data::get_status(size_t row) const {
    if (!status_index.has_value()) {
        throw std::runtime_error("Status index has not been set");
    }
    if (row >= num_rows) {
        throw std::runtime_error("Row index out of bounds");
    }
    return static_cast<size_t>(get(row, status_index.value()));
}

double Data::get_weight(size_t row) const {
    if (!weight_index.has_value()) {
        return 1.0;  // 가중치가 설정되지 않은 경우 기본값
    }
    return get(row, weight_index.value());
}

double Data::get_outcome(size_t row) const {
    if (!outcome_index.has_value()) {
        throw std::runtime_error("Outcome index has not been set");
    }
    return get(row, outcome_index.value()[0]);
}

Eigen::VectorXd Data::get_outcomes(size_t row) const {
    if (!outcome_index.has_value()) {
        throw std::runtime_error("Outcome indices have not been set");
    }
    Eigen::VectorXd outcomes(outcome_index.value().size());
    for (size_t i = 0; i < outcome_index.value().size(); i++) {
        outcomes[i] = get(row, outcome_index.value()[i]);
    }
    return outcomes;
}

double Data::get_treatment(size_t row) const {
    if (!treatment_index.has_value()) {
        throw std::runtime_error("Treatment index has not been set");
    }
    return get(row, treatment_index.value()[0]);
}

Eigen::VectorXd Data::get_treatments(size_t row) const {
    if (!treatment_index.has_value()) {
        throw std::runtime_error("Treatment indices have not been set");
    }
    Eigen::VectorXd treatments(treatment_index.value().size());
    for (size_t i = 0; i < treatment_index.value().size(); i++) {
        treatments[i] = get(row, treatment_index.value()[i]);
    }
    return treatments;
}

double Data::get_instrument(size_t row) const {
    if (!instrument_index.has_value()) {
        throw std::runtime_error("Instrument index has not been set");
    }
    return get(row, instrument_index.value());
}

size_t Data::get_num_treatments() const {
    if (treatment_index.has_value()) {
        return treatment_index.value().size();
    }
    return 1;
}

bool Data::is_failure(size_t row) const {
    return get_status(row) == 1;
}

bool Data::is_failure_status(size_t row, size_t status) const {
    return get_status(row) == status;
}

size_t Data::get_status_max() const {
    if (!status_max.has_value()) {
        throw std::runtime_error("Status max has not been set");
    }
    return status_max.value();
}

double Data::get(size_t row, size_t col) const {
    if (row >= num_rows || col >= num_cols) {
        throw std::runtime_error("Index out of bounds");
    }
    return data_ptr[col * num_rows + row];
}

// 중복 정의 제거
// size_t Data::get_num_treatments() const {
//   return num_treatments;
// }

// double Data::get_causal_survival_numerator(size_t row) const {
//   if (causal_survival_numerator_index != -1) {
//     return get(row, causal_survival_numerator_index);
//   }
//   return 0.0;
// }

// double Data::get_causal_survival_denominator(size_t row) const {
//   if (causal_survival_denominator_index != -1) {
//     return get(row, causal_survival_denominator_index);
//   }
//   return 1.0;
// }

} // namespace grf
