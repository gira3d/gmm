#ifndef KMEANS_ELKAN_H
#define KMEANS_ELKAN_H

#include <random>

#include <boost/shared_ptr.hpp>

#include<Eigen/Dense>

#include <gmm/KInit.h>

template<typename T, int K> class Partitioner;
template <typename T, int K> class CXXPartitioner;

static bool isLessThanZero(float i) { return (i < 0); }

/* Initializes cluster centers for KMeans */
template<typename T, uint32_t F, int K=-1, int S=-1>
class KMeansElkan
{
	public:

  typedef boost::shared_ptr<KMeansElkan> Ptr;
  typedef boost::shared_ptr<KMeansElkan const> ConstPtr;

  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatrixX;
  typedef Eigen::Matrix<T, F, K, Eigen::ColMajor> Matrixfk;
  typedef Eigen::Matrix<T, N<K>(), 1, Eigen::ColMajor> VectornX;
  typedef Eigen::Matrix<T, Eigen::Dynamic, 1, Eigen::ColMajor> VectorX;
  typedef Eigen::Matrix<T, F, Eigen::Dynamic, Eigen::ColMajor> MatrixfX;
  typedef Eigen::Matrix<T, F, S, Eigen::ColMajor> Matrixfs;
  typedef Eigen::Matrix<T, F, N<K>(), Eigen::ColMajor> Matrixfn;
  typedef Eigen::Matrix<T, N<K>(), Eigen::Dynamic, Eigen::ColMajor> MatrixnX;
  typedef Eigen::Matrix<T, S, 1, Eigen::ColMajor> VectorsX;
  typedef Eigen::Matrix<T, K, K, Eigen::ColMajor> Matrixkk;
  typedef Eigen::Matrix<T, F, 1, Eigen::ColMajor> VectorfX;
  typedef Eigen::Matrix<T, K, Eigen::Dynamic, Eigen::ColMajor> MatrixkX;
  typedef Eigen::Matrix<T, K, 1, Eigen::ColMajor> VectorkX;

	KMeansElkan(const typename Partitioner<T,K>::Ptr& partitioner,
              const uint32_t max_iter,
              const int n_clusters = K)
  {
    if (n_clusters <= 0) throw std::invalid_argument("KMeansElkan: Number of clusters must be > 0.");
    n_clusters_ = n_clusters;
    n_local_trials_ = nLocalTrials<K>(n_clusters_);
    partitioner_ = partitioner;
    max_iter_ = max_iter;
  }

	~KMeansElkan() {}

    std::pair<Matrixfk, VectorsX> kmeans_elkan(const Matrixfs& X, const VectorsX& sample_weight,
                                               Matrixfk centers, const T tol)
    {
      uint32_t n_samples = X.cols();
      uint32_t n_features = X.rows();

      int label;

      T upper_bound, distance;

      MatrixX none;
      Matrixkk center_half_distances = euclidean_distances(centers,centers,none) / 2;
      MatrixkX lower_bounds = MatrixkX::Zero(n_clusters_, n_samples);
      VectorkX distance_next_center;
      VectorsX labels = VectorsX::Zero(n_samples, 1);
      VectorsX upper_bounds = VectorsX::Zero(n_samples,1);

      // Get the initial set of upper bounds and lower bounds for each sample
      update_labels_distances_inplace(X, centers, center_half_distances,
                                      labels, lower_bounds, upper_bounds,
                                      n_samples, n_features);

      VectorsX bounds_tight = VectorsX::Ones(n_samples,1);
      Matrixfk new_centers;
      T center_shift_total = 0.0;

      for (uint32_t iteration = 0; iteration < max_iter_; ++iteration)
      {
        //should go inside for loop from here down
        distance_next_center = partitioner_->partition(center_half_distances);

        for (uint32_t point_index = 0; point_index < n_samples; ++point_index)
        {
          upper_bound = upper_bounds(point_index);
          label = labels(point_index);

          // This means that the next likely center is far away from the
          // currently assigned center and the sample is unlikely to be
          // reassigned.
          if (distance_next_center(label) >= upper_bound)
            continue;

          VectorfX x_p = X.col(point_index);

          for (int center_index = 0; center_index < n_clusters_; ++center_index)
          {

            // If this holds, then center_index is a good candidate for the
            // sample to be relabelled, and we need to confirm this by
            // recomputing the upper and lower bounds
            if ( (center_index != label) &&
                 (upper_bound > lower_bounds(center_index, point_index)) &&
                 (upper_bound > center_half_distances(label, center_index)) )
            {
              // Recompute the upper bound by calculating the actual distance
              // between the sample and the label
              if (!bounds_tight(point_index) )
              {
                upper_bound = fast_euclidean_dist(x_p, centers.col(label));
                lower_bounds(label, point_index) = upper_bound;
                bounds_tight(point_index) = 1;
              }

              //If the condition still holds, then compute the actual distance
              //between the sample and the center_index.  If this is still lesser
              //than the previous distance, reassign labels.
              if ( (upper_bound > lower_bounds(center_index, point_index)) ||
                   (upper_bound > center_half_distances(center_index, label)))
              {
                distance = fast_euclidean_dist(x_p, centers.col(center_index));
                lower_bounds(center_index, point_index) = distance;

                if (distance < upper_bound)
                {
                  label = center_index;
                  upper_bound = distance;
                }
              }
            }
          }
          labels(point_index) = label;
          upper_bounds(point_index) = upper_bound;
        }

        new_centers = centers_dense(X, sample_weight, labels, upper_bounds);

        bounds_tight = VectorsX::Zero(n_samples,1);

        VectorkX center_shift = ((centers - new_centers).array().square()).colwise().sum().transpose().array().sqrt();

        MatrixkX slb = lower_bounds - center_shift.replicate(1, lower_bounds.cols());
        std::vector<T> v(slb.data(), slb.data() + slb.rows()*slb.cols());
        std::replace_if (v.begin(), v.end(), isLessThanZero, 0);
        Eigen::Map<MatrixkX> slb2(v.data(), lower_bounds.rows(), lower_bounds.cols());
        lower_bounds = slb2;

        VectorsX temp_upper = VectorsX::Zero(n_samples, 1);
        for (uint32_t ubi = 0; ubi < n_samples; ++ubi)
          temp_upper(ubi) = center_shift((int) labels(ubi));
        upper_bounds = upper_bounds + temp_upper;

        centers = new_centers;

        center_half_distances = euclidean_distances(centers,centers,none) / 2.;

        center_shift_total = center_shift.array().sum();
        if (center_shift_total*center_shift_total < tol)
          break;
      }

      if (center_shift_total > 0)
        update_labels_distances_inplace(X, centers, center_half_distances,
                                        labels, lower_bounds, upper_bounds,
                                        n_samples, n_features);

      return std::pair<Matrixfk, VectorsX>(new_centers,labels);
    }



private:

  /////////////////////////
  // X: n_features x n_samples
  // centers: n_features x n_clusters
  // center_half_distances: n_clusters x n_clusters
  ////////////////////////
  void update_labels_distances_inplace(const Matrixfs& X, const Matrixfk& centers, const Matrixkk& center_half_distances,
                                       VectorsX& labels, MatrixkX& lower_bounds, VectorsX& upper_bounds,
                                       const uint32_t n_samples, const uint32_t n_features)
  {
    float d_c, dist;
    uint32_t c_x, sample;
    VectorfX c;
    VectorfX x;
    int j;
    for (sample = 0; sample < n_samples; ++sample)
    {
      c_x = 0;
      x = X.col(sample);
      d_c = euclidean_dist(x, centers, n_features);
      lower_bounds(0, sample) = d_c;

      for (j = 1; j < n_clusters_; ++j)
      {
        if (d_c > center_half_distances(j,c_x))
        {
          c = centers.col(j);
          dist = fast_euclidean_dist(x, c);
          lower_bounds(j, sample) = dist;
          if (dist < d_c)
          {
            d_c = dist;
            c_x = j;
          }
        }
      }
      labels(sample) = c_x;
      upper_bounds(sample) = d_c;
    }
  }

  T euclidean_dist(const VectorfX& a, const MatrixfX& b, const uint32_t n_features)
  {
    T result = (a-b.col(0)).array().square().sum();
    return std::sqrt(result);
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
    distances = distances.array().sqrt();
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

  T fast_euclidean_dist(const VectorfX& a, const VectorfX& b)
  {
    T result = (a-b).array().square().sum();
    return std::sqrt(result);
  }

  //
  // returns matrix containing locations of zero entries
  //
  MatrixX centers_dense(const Matrixfs& X, const VectorsX& sample_weight,
                        const VectorsX& labels, const VectorsX& distances)
  {
    uint32_t n_samples = X.cols();

    Matrixfk centers = MatrixX::Zero(F, n_clusters_);
    VectorkX weight_in_cluster = VectorkX::Zero(n_clusters_, 1);

    // TODO: move clusters if empty_clusters is > 0
    for (uint32_t i = 0; i < n_samples; ++i)
    {
      uint32_t c = labels(i);
      weight_in_cluster(c) += sample_weight(i);
    }

    std::vector<uint32_t> empty_clusters = where(weight_in_cluster);

    //TODO: haven't checked this case
    if (empty_clusters.size() > 0)
    {
      std::vector<uint32_t> far_from_centers = argsort(distances);
      for (uint32_t i = 0; i < empty_clusters.size(); ++i)
      {
        uint32_t cluster_id = empty_clusters[i];
        size_t far_index = far_from_centers[i];
        MatrixX new_center = X.col(far_index);
        centers.col(cluster_id) << new_center;
        weight_in_cluster(cluster_id) = sample_weight(far_index);
      }
    } // END TODO

    for (uint32_t i = 0; i < n_samples; ++i)
      for (uint32_t j = 0; j < F; ++j)
        centers(j, (int)labels(i)) += X(j,i) * sample_weight(i);

    for (int i = 0; i < n_clusters_; ++i)
      centers.col(i) = centers.col(i) / weight_in_cluster(i);

    return centers;
  }

  std::vector<uint32_t> where(const VectorkX& weight_in_cluster)
  {
    std::vector<T> v(weight_in_cluster.data(),
                     weight_in_cluster.data() +
                     weight_in_cluster.rows());

    std::vector<uint32_t> empty_clusters;
    empty_clusters.reserve(v.size());

    for (uint32_t i = 0; i < v.size(); ++i)
      if (std::abs(v[i]) < 1e-5)
        empty_clusters.push_back(i);

    return empty_clusters;
  }

  //TODO: clean up this code
  std::vector<uint32_t> argsort(const VectorsX& distances)
  {
    std::vector<T> v(distances.data(),
                     distances.data() +
                     distances.rows()*distances.cols());

    std::vector<uint32_t> idx(v.size());

    // initialize original index locations
    std::iota(idx.begin(), idx.end(), 0);

    // sort indexes from largest to smallest value
    std::sort(idx.begin(), idx.end(),
              [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});

    return idx;
  }

  int n_clusters_;
  int n_local_trials_;
  uint32_t max_iter_;
  typename Partitioner<T,K>::Ptr partitioner_;

};

template <typename T, int K>
  class Partitioner
{

public:

  typedef boost::shared_ptr<Partitioner<T,K> > Ptr;
  typedef boost::shared_ptr<Partitioner<T,K> const> ConstPtr;

  typedef Eigen::Matrix<T, K, K, Eigen::ColMajor> Matrixkk;
  typedef Eigen::Matrix<T, K, 1, Eigen::ColMajor> VectorkX;

  Partitioner() {}
  ~Partitioner() {}

  virtual VectorkX partition(const Matrixkk& X) = 0;

private:

};

template <typename T, int K>
  class CXXPartitioner : public Partitioner<T,K>
{
public:

  typedef boost::shared_ptr<CXXPartitioner<T,K> > Ptr;
  typedef boost::shared_ptr<CXXPartitioner<T,K> const> ConstPtr;

  typedef Eigen::Matrix<T, K, K, Eigen::ColMajor> Matrixkk;
  typedef Eigen::Matrix<T, K, 1, Eigen::ColMajor> VectorkX;

  CXXPartitioner() {}
  ~CXXPartitioner() {}

  virtual VectorkX partition(const Matrixkk& X)
  {
    std::vector<T> v(X.data(), X.data() + X.rows()*X.cols());
    std::partition(v.begin(), v.end(), partitionFunction);
    Eigen::Map<Matrixkk> sorted(v.data(), X.rows(), X.cols());
    return sorted.col(1);
  }

private:
  static bool partitionFunction(float i) { return (i < 1); }

};

#endif //KMEANS_ELKAN
