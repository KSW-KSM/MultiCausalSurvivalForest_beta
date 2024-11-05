#include "MultiCausalSurvivalSplittingRuleFactory.h"
#include "../MultiCausalSurvivalSplittingRule.h"
#include "Rcpp.h" 

namespace grf {

std::unique_ptr<SplittingRule> MultiCausalSurvivalSplittingRuleFactory::create(size_t max_num_unique_values,
                                                                               const TreeOptions& options) const {
    Rcpp::Rcout << "\n=== Creating MultiCausalSurvivalSplittingRule ===" << std::endl;
    
    try {
        // 입력 파라미터 로깅
        Rcpp::Rcout << "Parameters:" << std::endl;
        Rcpp::Rcout << "- max_num_unique_values: " << max_num_unique_values << std::endl;
        Rcpp::Rcout << "- min_node_size: " << options.get_min_node_size() << std::endl;
        Rcpp::Rcout << "- alpha: " << options.get_alpha() << std::endl;
        Rcpp::Rcout << "- imbalance_penalty: " << options.get_imbalance_penalty() << std::endl;
        Rcpp::Rcout << "- num_treatments: " << options.get_num_treatments() << std::endl;

        Rcpp::Rcout << "Creating splitting rule object..." << std::endl;
        auto splitting_rule = std::unique_ptr<SplittingRule>(
            new MultiCausalSurvivalSplittingRule(
                max_num_unique_values,
                options.get_min_node_size(),
                options.get_alpha(),
                options.get_imbalance_penalty(),
                options.get_num_treatments()
            )
        );
        
        if (splitting_rule) {
            Rcpp::Rcout << "Successfully created MultiCausalSurvivalSplittingRule" << std::endl;
        } else {
            Rcpp::Rcout << "Warning: Failed to create splitting rule (nullptr)" << std::endl;
        }
        
        Rcpp::Rcout << "=== Splitting Rule Creation Completed ===\n" << std::endl;
        return splitting_rule;
        
    } catch (const std::exception& e) {
        Rcpp::Rcout << "Error creating splitting rule: " << e.what() << std::endl;
        throw;
    } catch (...) {
        Rcpp::Rcout << "Unknown error creating splitting rule" << std::endl;
        throw;
    }
}

} // namespace grf