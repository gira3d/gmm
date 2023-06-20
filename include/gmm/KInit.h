#ifndef KINIT_H
#define KINIT_H

#include <stdexcept>
#include <random>
#include <cstdlib>
#include <iostream>

#include <boost/shared_ptr.hpp>

#include<Eigen/Dense>

template<typename T> class RandomNLocalTrialsGenerator;
template<typename T> class UniformIntDistribution;

// this function calculates 2 + floor(log2(K))
template <int NCLUSTERS=-1>
  static constexpr int N()
{
  if constexpr(NCLUSTERS < 0) return -1;
  else if constexpr(NCLUSTERS < 2) return 0;
  else if constexpr(NCLUSTERS < 4) return 1;
  else if constexpr(NCLUSTERS < 8) return 2;
  else if constexpr(NCLUSTERS < 16) return 3;
  else if constexpr(NCLUSTERS < 32) return 4;
  else if constexpr(NCLUSTERS < 64) return 5;
  else if constexpr(NCLUSTERS < 128) return 6;
  else if constexpr(NCLUSTERS < 256) return 7;
  else if constexpr(NCLUSTERS > 255)
  {
    static_assert((NCLUSTERS > 255) ? false : true,
                  "Static allocation of more than 256 components not supported. Try dynamic allocation via templated -1 parameter instead.");
  }
}

template <int K>
int nLocalTrials(int n_clusters_)
{
  if constexpr (K < 0)
  {
    return 2 + std::floor(std::log(n_clusters_));
  }
  else return N<K>();
}

/* Initializes cluster centers for KMeans */
template<typename T, uint32_t F, int K=-1, int S=-1>
class KInit
{

	public:

  typedef boost::shared_ptr<KInit> Ptr;
  typedef boost::shared_ptr<KInit const> ConstPtr;

  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatrixX;
  typedef Eigen::Matrix<T, N<K>(), 1, Eigen::ColMajor> VectornX;
  typedef Eigen::Matrix<T, Eigen::Dynamic, 1, Eigen::ColMajor> VectorX;
  typedef Eigen::Matrix<T, F, Eigen::Dynamic, Eigen::ColMajor> MatrixfX;
  typedef Eigen::Matrix<T, F, S, Eigen::ColMajor> Matrixfs;
  typedef Eigen::Matrix<T, F, N<K>(), Eigen::ColMajor> Matrixfn;
  typedef Eigen::Matrix<T, N<K>(), Eigen::Dynamic, Eigen::ColMajor> MatrixnX;
  typedef Eigen::Matrix<T, S, 1, Eigen::ColMajor> VectorsX;

	KInit(const typename RandomNLocalTrialsGenerator<T>::Ptr& random_number_generator,
        const uint32_t n_clusters = K)
  {
    if (n_clusters <= 0) throw std::invalid_argument("KInit: Number of clusters must be > 0.");
    n_clusters_ = n_clusters;
    n_local_trials_ = nLocalTrials<K>(n_clusters_);
    rand_num_gen_ = random_number_generator;
  }

	~KInit() {}

  MatrixfX k_init(const MatrixfX& X, const VectorX& x_squared_norms)
  {
    //create empty centers of size n_features x n_clusters
    MatrixfX centers = MatrixfX(F, n_clusters_);

    //pick first center randomly
    uint32_t center_id = rand_num_gen_->getCenterID();
    centers.col(0) << X.col(center_id);

    //initialize list of closest distances and calculate current potential
    VectorsX closest_dist_sq = euclidean_distances(centers.col(0), X, x_squared_norms);
    T current_pot = closest_dist_sq.array().sum();

    for (uint32_t i = 1; i < n_clusters_; ++i)
    {
      VectornX rand_vals = rand_num_gen_->getRandomVector(n_local_trials_, current_pot);

      //CHECK: n_local_trials x 1
      VectornX candidate_ids = searchSorted(closest_dist_sq, rand_vals);

      // n_features x n_local_trials
      Matrixfn extracted_candidates = extractCandidates(X, candidate_ids);

      //Compute distances to center candidates n_samples x n_local_trials
      MatrixnX distance_to_candidates =
        fast_euclidean_distances(extracted_candidates, X, x_squared_norms);

      uint32_t best_candidate = std::numeric_limits<uint32_t>::max();

      T best_pot = 0.0;
      MatrixX best_dist_sq;

      for (uint32_t trial = 0; trial < n_local_trials_; ++trial)
      {
        //Compute potential when including center candidate
        VectorX new_dist_sq = closest_dist_sq.cwiseMin(distance_to_candidates.row(trial).transpose());
        T new_pot = new_dist_sq.array().abs().sum();

        if ( (best_candidate == std::numeric_limits<uint32_t>::max()) || (new_pot < best_pot) )
        {
          best_candidate = candidate_ids(trial, 0);
          best_pot = new_pot;
          best_dist_sq = new_dist_sq;
        }
      }

      centers.col(i) << X.col(best_candidate);
      current_pot = best_pot;
      closest_dist_sq = best_dist_sq;
      std::cerr << "current_pot: " << current_pot << std::endl;

    }
    return centers;
  }

private:

  //  Find the indices into a sorted array a such that, if the corresponding elements
  //  in v were inserted before the indices, the order of a would be preserved.
  VectornX searchSorted(const VectorsX& closest_dist_sq, const VectornX& rand_vals)
  {
    const uint32_t n_samples = closest_dist_sq.rows();
    VectorsX Y = cumsum(closest_dist_sq); // 10 x 1
    MatrixnX Z = Y.transpose().replicate(rand_vals.rows(), 1);
    MatrixnX W = rand_vals.replicate(1,Y.rows());

    MatrixnX diff = Z-W;

    //TODO: doesn't always hit the inside case
    VectornX indices = MatrixX::Zero(rand_vals.rows(), rand_vals.cols());

    for (uint32_t i = 0; i < n_local_trials_; ++i)
      for (uint32_t j = 0; j < n_samples; ++j)
        if (diff(i,j) > 0)
        {
          indices(i) = j;
          break;
        }
    return indices;
  }


  MatrixX euclidean_distances(const MatrixX& X, const MatrixX& Y, const MatrixX& Y_squared_norms)
  {
    MatrixX XX = X.array().square().colwise().sum();
    MatrixX YY;
    if ((X.rows() == Y.rows()) && (X.cols() == Y.cols()) && X.isApprox(Y))
      YY = XX.transpose();
    else if (Y_squared_norms.rows() != 0)
      YY = Y_squared_norms;
    else
      YY = Y.colwise().normalized().array().square();

    MatrixX distances = X.transpose() * Y;
    distances *= -2;
    distances += XX.transpose().replicate(1, distances.cols());
    distances += YY.transpose().replicate(distances.rows(), 1);

    return distances.transpose();
  }

  MatrixnX fast_euclidean_distances(const Matrixfn& X, const MatrixfX& Y, const VectorX& Y_squared_norms)
  {
    VectornX XX = X.array().square().colwise().sum().transpose();
    MatrixnX distances = X.transpose() * Y;
    distances *= -2;
    distances += XX.replicate(1, distances.cols());
    distances += Y_squared_norms.transpose().replicate(distances.rows(), 1);
    return distances;
  }

  Matrixfn extractCandidates(const Matrixfs& X, const VectornX& candidate_ids)
  {
    Matrixfn XMod(F, n_local_trials_);
    for (uint32_t i = 0; i < n_local_trials_; ++i)
    {
      XMod.col(i) << X.col(candidate_ids(i));
    }
    return XMod;
  }

  VectorX cumsum(const VectorX& X)
  {
    VectorX Y = X;

    for (uint32_t i = 1; i < X.rows(); ++i)
      Y(i, 0) += Y(i-1, 0);

    return Y;
  }

  uint32_t n_clusters_;
  uint32_t n_local_trials_;

  typename RandomNLocalTrialsGenerator<T>::Ptr rand_num_gen_;

};

// Base class to randomly generate numbers for KInit
template<typename T>
class RandomNLocalTrialsGenerator
{
	public:

  typedef boost::shared_ptr<RandomNLocalTrialsGenerator<T> > Ptr;
  typedef boost::shared_ptr<RandomNLocalTrialsGenerator<T> const> ConstPtr;

	RandomNLocalTrialsGenerator() {}
	~RandomNLocalTrialsGenerator() {}

	virtual uint32_t getCenterID() = 0;

	virtual Eigen::Matrix<T, -1, 1, Eigen::ColMajor>
		getRandomVector(uint32_t n_local_trials, T current_pot) = 0;

	private:
};

// Derived class to randomly generate numbers for KInit
template<typename T>
class UniformIntDistribution : public RandomNLocalTrialsGenerator<T>
{

public:

  typedef boost::shared_ptr<UniformIntDistribution> Ptr;
  typedef boost::shared_ptr<UniformIntDistribution const> ConstPtr;

	UniformIntDistribution(uint32_t n_samples)
	{
    distribution_ = std::uniform_int_distribution<int>(0, n_samples-1);
	}

	~UniformIntDistribution() {}

  virtual uint32_t getCenterID()
  {
    return distribution_(generator_);
  }

  Eigen::Matrix<T, -1, 1, Eigen::ColMajor>
    getRandomVector(uint32_t n_local_trials, T current_pot)
  {
    return Eigen::Matrix<T, -1, 1, Eigen::ColMajor>::Random(n_local_trials, 1).array().abs() * current_pot;
  }

private:

  std::uniform_int_distribution<int> distribution_;
  std::default_random_engine generator_;
};

#endif //KINIT
