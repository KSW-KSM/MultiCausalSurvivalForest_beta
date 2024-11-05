#include <Rcpp.h>
#include <vector>

#include "commons/globals.h"
#include "forest/ForestPredictors.h"
#include "forest/ForestTrainers.h"
#include "RcppUtilities.h"
#include "Eigen/Dense"

#include "commons/utility.h"

using namespace grf;

// [[Rcpp::export]]
Rcpp::List multi_causal_survival_train(const Rcpp::NumericMatrix& train_matrix,
                                      size_t treatment1_index,
                                      size_t treatment2_index,
                                      size_t causal_survival_numerator1_index,
                                      size_t causal_survival_denominator1_index,
                                      size_t causal_survival_numerator2_index,
                                      size_t causal_survival_denominator2_index,
                                      size_t censor_index,
                                      size_t sample_weight_index,
                                      bool use_sample_weights,
                                      unsigned int num_trees,
                                      const std::vector<size_t>& clusters,
                                      unsigned int samples_per_cluster,
                                      double sample_fraction,
                                      unsigned int mtry,
                                      unsigned int min_node_size,
                                      bool honesty,
                                      double honesty_fraction,
                                      bool honesty_prune_leaves,
                                      size_t ci_group_size,
                                      double alpha,
                                      double imbalance_penalty,
                                      bool stabilize_splits,
                                      bool compute_oob_predictions,
                                      unsigned int num_threads,
                                      unsigned int seed,
                                      bool mahalanobis,
                                      const Rcpp::NumericMatrix& sigma) {
    try {
        Rcpp::Rcout << "Step 1: Starting training process..." << std::endl;
        
        Rcpp::Rcout << "Step 2: Creating trainer..." << std::endl;
        ForestTrainer trainer = multi_causal_survival_trainer(stabilize_splits);
        
        Rcpp::Rcout << "Step 3: Converting data..." << std::endl;
        Data forest_data = RcppUtilities::convert_data(train_matrix);
        
        std::vector<size_t> treatment_indices = {treatment1_index, treatment2_index};
        Rcpp::Rcout << "treatment_indices: " << treatment_indices.size() << std::endl;
        size_t num_treatments = treatment_indices.size();
        
        Rcpp::Rcout << "Step 4: Setting up sigma matrix..." << std::endl;
        Eigen::MatrixXd _sigma = Eigen::MatrixXd(num_treatments, num_treatments);
        if (!mahalanobis) {
            _sigma.setIdentity();
        }
        
        Rcpp::Rcout << "Step 5: Setting up indices..." << std::endl;
        std::vector<std::vector<size_t>> numerator_indices;
        numerator_indices.push_back({causal_survival_numerator1_index});
        numerator_indices.push_back({causal_survival_numerator2_index});
        forest_data.set_multi_causal_survival_numerator_index(numerator_indices);

        std::vector<std::vector<size_t>> denominator_indices;
        denominator_indices.push_back({causal_survival_denominator1_index});
        denominator_indices.push_back({causal_survival_denominator2_index});
        forest_data.set_multi_causal_survival_denominator_index(denominator_indices);

        for (size_t i = 0; i < num_treatments; ++i) {
            forest_data.set_treatment_index(treatment_indices[i]);
        }
        
        forest_data.set_status_index(censor_index);
        
        if (use_sample_weights) {
            forest_data.set_weight_index(sample_weight_index);
        }

                Eigen::MatrixXd eigen_sigma;
        if (mahalanobis) {
            eigen_sigma = _sigma;  // 직접 대입
        } else {
            eigen_sigma = Eigen::MatrixXd::Identity(num_treatments, num_treatments);
        }

        ForestOptions options(num_trees, 
                            ci_group_size,
                            sample_fraction,
                            mtry,
                            min_node_size,
                            honesty,
                            honesty_fraction,
                            honesty_prune_leaves,
                            alpha,
                            imbalance_penalty,
                            num_threads,
                            seed,
                            clusters,
                            samples_per_cluster,
                            mahalanobis,
                            eigen_sigma,
                            num_treatments);

                Rcpp::Rcout << "Step 7: Training forest..." << std::endl;
        
        // 데이터 유효성 검사 추가
        if (forest_data.get_num_rows() == 0 || forest_data.get_num_cols() == 0) {
            throw std::runtime_error("Empty forest data");
        }
        
        // options 유효성 검사
        if (num_trees == 0 || mtry == 0) {
            throw std::runtime_error("Invalid forest options");
        }
        
        // 메모리 상태 로깅
        Rcpp::Rcout << "Number of rows: " << forest_data.get_num_rows() << std::endl;
        Rcpp::Rcout << "Number of columns: " << forest_data.get_num_cols() << std::endl;
        Rcpp::Rcout << "Number of trees: " << num_trees << std::endl;
        
        // R의 인터럽트 체크 추가
        if (num_trees > 100) {  // 큰 작업의 경우
            R_CheckUserInterrupt();
        }
        
        Forest forest = trainer.train(forest_data, options);
        Rcpp::Rcout << "Step 8: Forest training completed." << std::endl;
        
        // 학습된 forest 유효성 검사
        if (forest.get_trees().empty()) {
            throw std::runtime_error("Training resulted in empty forest");
        }
        
        std::vector<Prediction> predictions;
        if (compute_oob_predictions) {
            // predictor 생성 전 체크
            if (!forest.get_trees().empty()) {
                ForestPredictor predictor = multi_causal_survival_predictor(num_threads);
                predictions = predictor.predict_oob(forest, forest_data, false);
            }
        }

        return RcppUtilities::create_forest_object(forest, predictions);
        
    } catch (const std::exception& e) {
        Rcpp::stop("Error in multi_causal_survival_train: %s", e.what());
    }
}

// [[Rcpp::export]]
Rcpp::List multi_causal_survival_predict(const Rcpp::List& forest_object,
                                        const Rcpp::NumericMatrix& train_matrix,
                                        const Rcpp::NumericMatrix& test_matrix,
                                        unsigned int num_threads,
                                        bool estimate_variance) {
    try {
        Rcpp::Rcout << "Starting prediction..." << std::endl;
        
        Data train_data = RcppUtilities::convert_data(train_matrix);
        Data data = RcppUtilities::convert_data(test_matrix);
        
        Forest forest = RcppUtilities::deserialize_forest(forest_object);
        
        ForestPredictor predictor = multi_causal_survival_predictor(num_threads);
        std::vector<Prediction> predictions = predictor.predict(forest, train_data, data, estimate_variance);
        Rcpp::List result = RcppUtilities::create_prediction_object(predictions);
        
        return result;
    } catch (const std::exception& e) {
        Rcpp::stop("Error in multi_causal_survival_predict: %s", e.what());
    }
}

// [[Rcpp::export]]
Rcpp::List multi_causal_survival_predict_oob(const Rcpp::List& forest_object,
                                            const Rcpp::NumericMatrix& train_matrix,
                                            unsigned int num_threads,
                                            bool estimate_variance) {
    try {
        Rcpp::Rcout << "Starting OOB prediction..." << std::endl;
        
        Data data = RcppUtilities::convert_data(train_matrix);
        
        Forest forest = RcppUtilities::deserialize_forest(forest_object);
        
        ForestPredictor predictor = multi_causal_survival_predictor(num_threads);
        std::vector<Prediction> predictions = predictor.predict_oob(forest, data, estimate_variance);
        Rcpp::List result = RcppUtilities::create_prediction_object(predictions);
        
        return result;
    } catch (const std::exception& e) {
        Rcpp::stop("Error in multi_causal_survival_predict_oob: %s", e.what());
    }
}