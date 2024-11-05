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

#include <algorithm>
#include <ctime>
#include <future>
#include <stdexcept>

#include "../commons/utility.h"
#include "ForestTrainer.h"
#include "../random/random.hpp"


namespace grf {

ForestTrainer::ForestTrainer(std::unique_ptr<RelabelingStrategy> relabeling_strategy,
                             std::unique_ptr<SplittingRuleFactory> splitting_rule_factory,
                             std::unique_ptr<OptimizedPredictionStrategy> prediction_strategy) :
    tree_trainer(std::move(relabeling_strategy),
                 std::move(splitting_rule_factory),
                 std::move(prediction_strategy)) {}

Forest ForestTrainer::train(const Data& data, const ForestOptions& options) const {
  Rcpp::Rcout << "Starting train_trees..." << std::endl;
  std::vector<std::unique_ptr<Tree>> trees = train_trees(data, options);
  Rcpp::Rcout << "train_trees completed." << std::endl;

  Rcpp::Rcout << "Calculating num_variables..." << std::endl;
  size_t num_variables = data.get_num_cols() - data.get_disallowed_split_variables().size();
  Rcpp::Rcout << "num_variables calculated: " << num_variables << std::endl;

  Rcpp::Rcout << "Getting ci_group_size..." << std::endl;
  size_t ci_group_size = options.get_ci_group_size();
  Rcpp::Rcout << "ci_group_size retrieved: " << ci_group_size << std::endl;

  Rcpp::Rcout << "Creating and returning Forest object..." << std::endl;
  return Forest(trees, num_variables, ci_group_size);
}

std::vector<std::unique_ptr<Tree>> ForestTrainer::train_trees(const Data& data,
                                                              const ForestOptions& options) const {
  size_t num_samples = data.get_num_rows();
  uint num_trees = options.get_num_trees();

  // 기본 유효성 검사
  const TreeOptions& tree_options = options.get_tree_options();
  bool honesty = tree_options.get_honesty();
  double honesty_fraction = tree_options.get_honesty_fraction();
  if ((size_t) num_samples * options.get_sample_fraction() < 1) {
    throw std::runtime_error("The sample fraction is too small, as no observations will be sampled.");
  } else if (honesty && ((size_t) num_samples * options.get_sample_fraction() * honesty_fraction < 1
             || (size_t) num_samples * options.get_sample_fraction() * (1-honesty_fraction) < 1)) {
    throw std::runtime_error("The honesty fraction is too close to 1 or 0, as no observations will be sampled.");
  }

    // 스레드 수 조정
  uint num_threads = std::min(options.get_num_threads(), 
                            static_cast<uint>(std::thread::hardware_concurrency()));
  if (num_threads == 0) num_threads = 1;
  Rcpp::Rcout << "Adjusted number of threads: " << num_threads << std::endl;

  // 그룹 크기 조정
  uint num_groups = std::max(static_cast<uint>(num_trees / options.get_ci_group_size()), 1u);
  Rcpp::Rcout << "Number of groups: " << num_groups << ", CI group size: " << options.get_ci_group_size() << std::endl;
  
  // 배치 크기 계산
  uint batch_size = std::max(static_cast<uint>(num_trees / num_threads), 1u);
  Rcpp::Rcout << "Batch size: " << batch_size << ", Total trees: " << num_trees << std::endl;
  
  std::vector<std::unique_ptr<Tree>> trees;
  trees.reserve(num_trees);
  Rcpp::Rcout << "Reserved space for " << num_trees << " trees" << std::endl;

  // 동기 실행으로 변경
  for (uint i = 0; i < num_trees; i += batch_size) {
    try {
      Rcpp::Rcout << "\n=== Starting batch " << i / batch_size + 1 << " ===" << std::endl;
      Rcpp::Rcout << "Current tree index: " << i << std::endl;
      
      uint current_batch_size = std::min(batch_size, num_trees - i);
      Rcpp::Rcout << "Current batch size: " << current_batch_size << std::endl;
      
      Rcpp::Rcout << "Calling train_batch..." << std::endl;
      auto thread_trees = train_batch(i, current_batch_size, data, options);
      Rcpp::Rcout << "train_batch returned " << thread_trees.size() << " trees" << std::endl;
      
      Rcpp::Rcout << "Inserting trees into result vector..." << std::endl;
      size_t before_size = trees.size();
      trees.insert(trees.end(),
                  std::make_move_iterator(thread_trees.begin()),
                  std::make_move_iterator(thread_trees.end()));
      Rcpp::Rcout << "Inserted " << (trees.size() - before_size) << " trees" << std::endl;
      
      Rcpp::Rcout << "Total trees trained so far: " << trees.size() << " / " << num_trees << std::endl;
      
      // 메모리 사용량 체크 (대략적인 추정)
      Rcpp::Rcout << "Approximate memory usage per tree: " 
                  << (sizeof(Tree) + data.get_num_rows() * sizeof(size_t)) / (1024.0 * 1024.0) 
                  << " MB" << std::endl;
      
      R_CheckUserInterrupt(); // R의 인터럽트 체크
      
    } catch (const std::exception& e) {
      Rcpp::Rcout << "Error in batch " << i / batch_size + 1 << ": " << e.what() << std::endl;
      throw;
    } catch (...) {
      Rcpp::Rcout << "Unknown error in batch " << i / batch_size + 1 << std::endl;
      throw;
    }
  }

  Rcpp::Rcout << "\nTraining completed. Total trees trained: " << trees.size() << std::endl;
  return trees;
}

std::vector<std::unique_ptr<Tree>> ForestTrainer::train_batch(
    size_t start,
    size_t num_trees,
    const Data& data,
    const ForestOptions& options) const {
  Rcpp::Rcout << "\nEntering train_batch: start=" << start << ", num_trees=" << num_trees << std::endl;
  
  size_t ci_group_size = options.get_ci_group_size();
  Rcpp::Rcout << "CI group size: " << ci_group_size << std::endl;

  try {
    Rcpp::Rcout << "Initializing random number generator..." << std::endl;
    std::mt19937_64 random_number_generator(options.get_random_seed() + start);
    nonstd::uniform_int_distribution<uint> udist;
    
    std::vector<std::unique_ptr<Tree>> trees;
    size_t reserve_size = num_trees * ci_group_size;
    Rcpp::Rcout << "Reserving space for " << reserve_size << " trees" << std::endl;
    trees.reserve(reserve_size);

    for (size_t i = 0; i < num_trees; i++) {
      Rcpp::Rcout << "Training tree " << i << " of " << num_trees << std::endl;
      
      try {
        uint tree_seed = udist(random_number_generator);
        Rcpp::Rcout << "Generated tree seed: " << tree_seed << std::endl;
        
        Rcpp::Rcout << "Creating RandomSampler..." << std::endl;
        RandomSampler sampler(tree_seed, options.get_sampling_options());
        
        if (ci_group_size == 1) {
          Rcpp::Rcout << "Training single tree..." << std::endl;
          std::unique_ptr<Tree> tree = train_tree(data, sampler, options);
          if (tree) {
            Rcpp::Rcout << "Tree training successful" << std::endl;
            trees.push_back(std::move(tree));
          } else {
            Rcpp::Rcout << "Warning: train_tree returned nullptr" << std::endl;
          }
        } else {
          Rcpp::Rcout << "Training CI group..." << std::endl;
          std::vector<std::unique_ptr<Tree>> group = train_ci_group(data, sampler, options);
          Rcpp::Rcout << "CI group training completed, got " << group.size() << " trees" << std::endl;
          
          trees.insert(trees.end(),
              std::make_move_iterator(group.begin()),
              std::make_move_iterator(group.end()));
        }
        
        Rcpp::Rcout << "Current total trees: " << trees.size() << std::endl;
        R_CheckUserInterrupt();
        
      } catch (const std::exception& e) {
        Rcpp::Rcout << "Error training tree " << i << ": " << e.what() << std::endl;
        throw;
      } catch (...) {
        Rcpp::Rcout << "Unknown error training tree " << i << std::endl;
        throw;
      }
    }
    
    Rcpp::Rcout << "train_batch completed successfully. Returning " << trees.size() << " trees" << std::endl;
    return trees;
    
  } catch (const std::exception& e) {
    Rcpp::Rcout << "Error in train_batch: " << e.what() << std::endl;
    throw;
  } catch (...) {
    Rcpp::Rcout << "Unknown error in train_batch" << std::endl;
    throw;
  }
}

std::unique_ptr<Tree> ForestTrainer::train_tree(const Data& data,
                                                RandomSampler& sampler,
                                                const ForestOptions& options) const {
  std::vector<size_t> clusters;
  sampler.sample_clusters(data.get_num_rows(), options.get_sample_fraction(), clusters);
  return tree_trainer.train(data, sampler, clusters, options.get_tree_options());
}

std::vector<std::unique_ptr<Tree>> ForestTrainer::train_ci_group(const Data& data,
                                                                 RandomSampler& sampler,
                                                                 const ForestOptions& options) const {
  Rcpp::Rcout << "\nEntering train_ci_group..." << std::endl;
  
  try {
    std::vector<std::unique_ptr<Tree>> trees;
    size_t ci_group_size = options.get_ci_group_size();
    Rcpp::Rcout << "CI group size: " << ci_group_size << std::endl;
    trees.reserve(ci_group_size);

    Rcpp::Rcout << "Sampling clusters..." << std::endl;
    std::vector<size_t> clusters;
    size_t num_rows = data.get_num_rows();
    Rcpp::Rcout << "Number of rows: " << num_rows << std::endl;
    
    try {
      sampler.sample_clusters(num_rows, 0.5, clusters);
      Rcpp::Rcout << "Sampled " << clusters.size() << " clusters" << std::endl;
    } catch (const std::exception& e) {
      Rcpp::Rcout << "Error in sample_clusters: " << e.what() << std::endl;
      throw;
    }

    double sample_fraction = options.get_sample_fraction();
    Rcpp::Rcout << "Sample fraction: " << sample_fraction << std::endl;

    for (size_t i = 0; i < ci_group_size; ++i) {
      Rcpp::Rcout << "\nTraining tree " << i << " of CI group" << std::endl;
      
      try {
        std::vector<size_t> cluster_subsample;
        Rcpp::Rcout << "Subsampling clusters..." << std::endl;
        sampler.subsample(clusters, sample_fraction * 2, cluster_subsample);
        Rcpp::Rcout << "Subsampled " << cluster_subsample.size() << " clusters" << std::endl;

        Rcpp::Rcout << "Training tree with TreeTrainer..." << std::endl;
        std::unique_ptr<Tree> tree = tree_trainer.train(
            data, 
            sampler, 
            cluster_subsample, 
            options.get_tree_options()
        );
        
        if (tree) {
          Rcpp::Rcout << "Tree training successful" << std::endl;
          trees.push_back(std::move(tree));
        } else {
          Rcpp::Rcout << "Warning: TreeTrainer returned nullptr" << std::endl;
        }
        
        Rcpp::Rcout << "Current trees in group: " << trees.size() << std::endl;
        R_CheckUserInterrupt();
        
      } catch (const std::exception& e) {
        Rcpp::Rcout << "Error training tree " << i << " in CI group: " << e.what() << std::endl;
        throw;
      } catch (...) {
        Rcpp::Rcout << "Unknown error training tree " << i << " in CI group" << std::endl;
        throw;
      }
    }

    Rcpp::Rcout << "train_ci_group completed successfully. Returning " << trees.size() << " trees" << std::endl;
    return trees;
    
  } catch (const std::exception& e) {
    Rcpp::Rcout << "Error in train_ci_group: " << e.what() << std::endl;
    throw;
  } catch (...) {
    Rcpp::Rcout << "Unknown error in train_ci_group" << std::endl;
    throw;
  }
}
}