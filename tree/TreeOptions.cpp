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

#include "TreeOptions.h"

namespace grf {

TreeOptions::TreeOptions(uint mtry,
                         uint min_node_size,
                         bool honesty,
                         double honesty_fraction,
                         bool honesty_prune_leaves,
                         double alpha,
                         double imbalance_penalty,
                         bool mahalanobis,
                         Eigen::MatrixXd sigma,
                         int num_treatments):
  mtry(mtry),
  min_node_size(min_node_size),
  honesty(honesty),
  honesty_fraction(honesty_fraction),
  honesty_prune_leaves(honesty_prune_leaves),
  alpha(alpha),
  imbalance_penalty(imbalance_penalty),
  mahalanobis(mahalanobis),
  sigma(sigma),
  num_treatments(num_treatments) {}

uint TreeOptions::get_mtry() const {
  return mtry;
}

uint TreeOptions::get_min_node_size() const {
  return min_node_size;
}

bool TreeOptions::get_honesty() const {
  return honesty;
}

double TreeOptions::get_honesty_fraction() const {
  return honesty_fraction;
}

bool TreeOptions::get_honesty_prune_leaves() const {
  return honesty_prune_leaves;
}

double TreeOptions::get_alpha() const {
  return alpha;
}

double TreeOptions::get_imbalance_penalty() const {
  return imbalance_penalty;
}

bool TreeOptions::get_mahalanobis() const {
    return mahalanobis;
}
Eigen::MatrixXd TreeOptions::get_sigma() const {
    return sigma;
}

int TreeOptions::get_num_treatments() const {
  return num_treatments;
}
} // namespace grf
