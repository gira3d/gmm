#ifndef KMEANS_H
#define KMEANS_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <boost/shared_ptr.hpp>

#include <stdexcept>
#include <random>

#include <cstdlib>
#include <iostream>

#include <vector>

#include <gmm/KInit.h>
#include <gmm/KMeansElkan.h>

#include <boost/shared_ptr.hpp>

// T is for the type (e.g., float, double etc)
// F is for the feature dimension (e.g., 3 for 3d points)
// K is for the number of components
// N is for n_local_trials...to compute this number type 2 + floor(log(K)) into matlab
// N = 6 for 100 components
template<typename T, uint32_t F, int K=-1, int S=-1>
  class KMeans
{
  public:

  typedef Eigen::Matrix<T, F, Eigen::Dynamic, Eigen::ColMajor> MatrixfX;
  typedef Eigen::Matrix<T, F, K, Eigen::ColMajor> Matrixfk;
  typedef Eigen::Matrix<T, K, Eigen::Dynamic, Eigen::ColMajor> MatrixkX;
  typedef Eigen::Matrix<T, K, S, Eigen::ColMajor> Matrixks;
  typedef Eigen::Matrix<T, F, S, Eigen::ColMajor> Matrixfs;
  typedef Eigen::Matrix<T, N<K>(), Eigen::Dynamic, Eigen::ColMajor> MatrixnX;
  typedef Eigen::Matrix<T, K, K, Eigen::ColMajor> Matrixkk;
  typedef Eigen::Matrix<T, F, N<K>(), Eigen::ColMajor> Matrixfn;
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatrixX;
  typedef Eigen::Matrix<T, Eigen::Dynamic, 1, Eigen::ColMajor> VectorX;
  typedef Eigen::Matrix<T, K, 1, Eigen::ColMajor> VectorkX;
  typedef Eigen::Matrix<T, F, 1, Eigen::ColMajor> VectorfX;
  typedef Eigen::Matrix<T, N<K>(), 1, Eigen::ColMajor> VectornX;
  typedef Eigen::Matrix<T, S, 1, Eigen::ColMajor> VectorsX;

  KMeans()
  {
    max_iter_             = 300;
    tol_                  = 1e-4;

    //faster, but takes more memory
    precompute_distances_ = true;
    verbose_              = false;

    // by default elkan data is used as pointcloud data is never sparse
    // whether or not to return the number of iterations
  }

  std::pair<Matrixfk, VectorsX> fit(Matrixfs X, int n_clusters = K)
  {
    if (n_clusters < 0)
      throw std::invalid_argument("Sklearn::fit(X, n_clusters) must have n_clusters > 0.");

    n_clusters_ = n_clusters;
    n_local_trials_ = nLocalTrials<K>(n_clusters_);

    uint32_t n_samples = X.cols();

    rand_num_gen_ = typename UniformIntDistribution<T>::Ptr(new UniformIntDistribution<T>(n_samples));
    partitioner_ = typename CXXPartitioner<T,K>::Ptr(new CXXPartitioner<T,K>());

    return fit(X, rand_num_gen_, partitioner_, n_clusters);
  }

  std::pair<Matrixfk, VectorsX> fit(Matrixfs X,
                                    const typename RandomNLocalTrialsGenerator<T>::Ptr& rand_num_gen,
                                    const typename Partitioner<T,K>::Ptr& partitioner,
                                    int n_clusters = K)
  {
    if (n_clusters < 0)
      throw std::invalid_argument("Sklearn::fit(X, n_clusters) must have n_clusters > 0.");

    n_clusters_ = n_clusters;
    n_local_trials_ = nLocalTrials<K>(n_clusters_);

    rand_num_gen_ = rand_num_gen;
    partitioner_ = partitioner;

    const uint32_t SS = X.cols();

    if (max_iter_ <= 0)
      throw std::invalid_argument("Number of iterations should be a positive number");

    //verify that the number of samples given is larger than number of clusters
    if (X.cols() < n_clusters_)
      throw std::invalid_argument("n_samples should be >= n_clusters");

    tol_ = tolerance(X, tol_);

    //If distances are precomputed every job will create a matrix of shape
    //(n_clusters, n_samples). To stop KMeans from eating up memory, we only
    //activate this if the created matrix is guaranteed to be under 100MB. 12
    //million entries consum a little under 100MB if they are of type double
    bool precompute_distances = (n_clusters_ * SS) < 12e7;
    if (precompute_distances == false)
      throw std::invalid_argument("Requires < 12e7 elements");

    //subtract the mean of X for more accuate distance calculation
    X.noalias() = (X.colwise() - X.rowwise().mean());

    // precompute squared norms of data points
    //x_squared_norms = col_norms(X, squared=True)
    VectorsX X_square_norms = X.array().square().colwise().sum().transpose();

    KInit<T,F,K,S> k_init = KInit<T,F,K,S>(rand_num_gen_, n_clusters_);
    Matrixfk centroids = k_init.k_init(X, X_square_norms);

    KMeansElkan<T,F,K,S> kmeans_elkan = KMeansElkan<T,F,K,S>(partitioner_, max_iter_, n_clusters_);
    std::pair<Matrixfk,VectorsX> result =
      kmeans_elkan.kmeans_elkan(X, VectorsX::Ones(SS,1), centroids, tol_);

    return result;
  }

  VectorfX variance(const Matrixfs& X)
  {
    //var = mean(abs(x - x.mean())**2)
    return ((X.colwise() - X.rowwise().mean()).array().square()).rowwise().mean();
  }

  T tolerance(const Matrixfs& X, const T& tol)
  {
    // mean(variances) * tol
    return variance(X).sum()/F * tol;
  }

private:

  typename RandomNLocalTrialsGenerator<T>::Ptr rand_num_gen_;
  typename Partitioner<T,K>::Ptr partitioner_;

  int n_clusters_;
  int n_local_trials_;

  uint32_t max_iter_;
  T tol_;

  std::string precompute_distances_;
  bool verbose_;
};
#endif //KMeans
