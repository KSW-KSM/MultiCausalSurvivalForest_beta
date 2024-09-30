#ifndef GRF_RELABELINGSTRATEGY_H
#define GRF_RELABELINGSTRATEGY_H

#include <vector>
#include "../Eigen/Dense"

namespace grf {

class Data; // 전방 선언

/**
 * Produces a relabelled set of outcomes for a set of training samples. These outcomes
 * will then be used in calculating a standard regression (or classification) split.
 *
 * The optional override `get_response_length()` is used for forests splitting on
 * vector-valued outcomes.
 */
class RelabelingStrategy {
public:

  virtual ~RelabelingStrategy() = default;

 /**
   * samples: the subset of samples to relabel.
   * data: the training data matrix.
   * responses_by_sample: the output of the method, an array of relabelled response for each sample ID in `samples`.
   * The dimension of this array is N * K where N is the total number of samples in the data, and K is given
   * by `get_response_length()`.
   *
   * In most cases, like a single-variable regression forest, K is 1, and `responses_by_sample` is a scalar for
   * each sample. In other forests, like multi-output regression forest, K is equal to the number of outcomes,
   * and `responses_by_sample` is a length K vector for each sample (working with a vector-valued splitting rule).
   *
   * Note that for performance reasons (avoiding clearing out the array after each split) this array may
   * contain garbage values for indices outside of the given set of sample IDs.
   *
   * returns: a boolean that will be 'true' if splitting should stop early.
   */
  virtual bool relabel(const std::vector<size_t>& samples,
                       const Data& data, //이부분에서 에러
                       Eigen::ArrayXXd& responses_by_sample) const = 0;

 /**
   * Override to specify the column dimension of `responses_by_sample`.
   * The default value of 1 is used for most forests splitting on scalar values.
   */
  virtual size_t get_response_length() const { return 1; };
};

} // namespace grf

#endif //GRF_RELABELINGSTRATEGY_H
