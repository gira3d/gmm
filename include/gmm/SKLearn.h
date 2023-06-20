#ifndef SKLEARN_H
#define SKLEARN_H

#include <Eigen/Core>
#include <Eigen/Dense>

#include <gmm/KMeans.h>

#define LOG_2_M_PI 1.83788

// T is for the type (e.g., float, double etc)
// F is for the feature dimension (e.g., 3 for 3d points)
// K is for the number of components
// S is for the number of samples
template <typename T, uint32_t F, int K=-1, int S=-1>
  class SKLearn
{
public:
  static constexpr uint32_t C = F*F;
  typedef boost::shared_ptr<SKLearn> Ptr;
  typedef boost::shared_ptr<SKLearn const> ConstPtr;

  typedef Eigen::Matrix<T, Eigen::Dynamic, 1, Eigen::ColMajor> VectorX;
  typedef Eigen::Matrix<T, S, 1, Eigen::ColMajor> VectorsX;
  typedef Eigen::Matrix<T, K, 1, Eigen::ColMajor> VectorkX;
  typedef Eigen::Matrix<T, C, 1, Eigen::ColMajor> VectorcX;
  typedef Eigen::Matrix<T, F, 1, Eigen::ColMajor> VectorfX;

  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatrixX;
  typedef Eigen::Matrix<T, F, Eigen::Dynamic, Eigen::ColMajor> MatrixfX;
  typedef Eigen::Matrix<T, K, Eigen::Dynamic, Eigen::ColMajor> MatrixkX;
  typedef Eigen::Matrix<T, S, Eigen::Dynamic, Eigen::ColMajor> MatrixsX;
  typedef Eigen::Matrix<T, F, S, Eigen::ColMajor> Matrixfs;
  typedef Eigen::Matrix<T, S, F, Eigen::ColMajor> Matrixsf;
  typedef Eigen::Matrix<T, F, K, Eigen::ColMajor> Matrixfk;
  typedef Eigen::Matrix<T, C, K, Eigen::ColMajor> Matrixck;
  typedef Eigen::Matrix<T, F, F, Eigen::ColMajor> Matrixff;


  typedef struct Parameters
  {
    uint32_t max_iter_ = 100;
    T reg_covar_       = 1e-6f;
    T tol_             = 1e-3f;
  } params;

  SKLearn() {}

  ~SKLearn() { }

  void fit (const MatrixX& X, const int n_clusters = K)
  {
    // call kmeans on the data
    std::pair<Matrixfk,VectorsX> result = kmeans_.fit(X, n_clusters);
    VectorsX labels = result.second;
    fit(X, labels, n_clusters);
  }

  void fit(const MatrixfX& X,
           const VectorsX& labels,
           const int n_clusters = K)
  {
    if (n_clusters <= 0) throw std::invalid_argument("SKLearn: Number of clusters must be > 0.");
    n_clusters_ = n_clusters;

    const uint32_t SS = X.cols();

    // create a K x S responsibility matrix
    resp_T_ = MatrixsX::Zero(SS,n_clusters_);

    // assign the value 1 to corresponding label
    for (uint32_t i = 0; i < SS; ++i)
      resp_T_(i, (int)labels(i)) = 1.0;

    weights_ = VectorkX::Zero(n_clusters_,1);
    means_ = Matrixfk::Zero(F,n_clusters_);
    covs_ = Matrixck::Zero(C,n_clusters_);
    precs_ = Matrixck::Zero(C,n_clusters_);

    estimate_gaussian_parameters(X, resp_T_);
    weights_.array() /= SS;
    compute_precisions_cholesky(covs_, precs_);

    converged_ = false;

    expectationMaximization(X);
  }

  inline void expectationMaximization(const MatrixfX& X)
  {
    T lower_bound = -std::numeric_limits<T>::infinity();

    for (uint32_t itr = 0; itr < p.max_iter_; ++itr)
    {
      T prev_lower_bound = lower_bound;

      lower_bound = eStep(X, means_, precs_, weights_, resp_T_);
      T change = lower_bound - prev_lower_bound;

      mStep(X, resp_T_);

      if (!std::isinf(change) && std::abs(change) < p.tol_)
      {
        converged_ = true;
        break;
      }
    }
  }

  inline T eStep(const MatrixfX& X,
          const Matrixfk& means,
          const Matrixck& precs,
          const VectorkX& weights,
          MatrixsX& resp_T)
  {
    return estimate_log_prob_resp(X, means, precs, weights, resp_T);
  }

  inline void mStep(const MatrixfX& X,
             const MatrixsX& resp_T)
  {
    const uint32_t SS = X.cols();
    estimate_gaussian_parameters(X, resp_T);
    weights_.array() /= SS;
    compute_precisions_cholesky(covs_, precs_);
  }

  inline T estimate_log_prob_resp(const MatrixfX& X,
                           const Matrixfk& means,
                           const Matrixck& precs,
                           const VectorkX& weights,
                           MatrixsX& resp_T)
  {
    MatrixkX weighted_log_prob = estimate_weighted_log_prob(X, means, precs, weights);
    VectorsX log_prob_norm = (weighted_log_prob.array().exp().colwise().sum() + std::numeric_limits<T>::min()).log().transpose();

    MatrixsX log_resp = (weighted_log_prob.transpose().colwise() - log_prob_norm);
    resp_T = log_resp.array().exp();

    //log_prob_norm is Sx1 and log_resp is SxK
    return log_prob_norm.array().mean();
  }

  inline MatrixkX estimate_weighted_log_prob(const MatrixfX& X,
                                      const Matrixfk& means,
                                      const Matrixck& precs,
                                      const VectorkX& weights)
  {
    // output is K x S
    return estimate_log_gaussian_prob(X, means, precs).array().colwise()
      + weights.array().log();
  }

  ////////////////////////////////////////////////////////////////
  // Compute the log-det of the cholesky decomposition of matrices
  // matrix_chol: matrix of dimension n_features * n_features x n_components
  // output: vector consisting of n_components entries
  ////////////////////////////////////////////////////////////////
  inline VectorkX compute_log_det_cholesky(const Matrixck& matrix_chol)
  {
    VectorkX log_det_chol = VectorkX::Zero(n_clusters_,1);
    Matrixff chol = Matrixff::Zero(F,F);
    VectorcX c = VectorcX::Zero(C,1);

    for (int k = 0; k < n_clusters_; ++k)
    {
      c = matrix_chol.col(k);
      chol << Eigen::Map<Matrixff>(c.data(), F, F);
      log_det_chol(k) = chol.diagonal().array().log().sum();
    }
    return log_det_chol;
  }

  ////////////////////////////////////////////////////
  // Estimate the log Gaussian probability
  ////////////////////////////////////////////////////
  inline MatrixkX estimate_log_gaussian_prob(const MatrixfX& X,
                                             const Matrixfk& means,
                                             const Matrixck& precs)
  {
    Matrixff prec_chol = Matrixff::Zero(F,F);
    VectorcX p = VectorcX::Zero(C,1);

    for (int k = 0; k < n_clusters_; ++k)
    {
      p = precs.col(k);
      prec_chol << Eigen::Map<Matrixff>(p.data(), F, F);
      resp_T_.col(k) << ((prec_chol * X).colwise() - (prec_chol * means.col(k))).array().square().colwise().sum().transpose();
    }

    return (-0.5 * ((F * LOG_2_M_PI) + resp_T_.array())).transpose().colwise() + compute_log_det_cholesky(precs).array();
  }

  ////////////////////////////////////////////////////
  // Estimate the Gaussian distribution parameters.
  ////////////////////////////////////////////////////
  inline void estimate_gaussian_parameters(const MatrixfX& X,
                                           const MatrixsX& resp_T)
  {
    const uint32_t SS = X.cols();
    MatrixfX diff = MatrixfX::Zero(F,SS);
    MatrixsX diff_temp = MatrixsX::Zero(SS,F);
    Matrixff cov = Matrixff::Zero(F,F);

    weights_.noalias() = resp_T.colwise().sum() + Eigen::Matrix<T, 1, K, Eigen::RowMajor>::Constant(1, n_clusters_, 10 * std::numeric_limits<T>::epsilon());

    // (F * S) * (S * K)
    means_ = (X * resp_T).array().rowwise() / weights_.transpose().array();

    //estimate the full covariances as Matrixck where c = n_features*n_features
    //e.g. 9 in the 3D point case
    for (int k = 0; k < n_clusters_; ++k)
    {
      diff.noalias() = X.colwise() - means_.col(k);
      diff_temp = (diff.transpose().array().colwise() * resp_T.col(k).array());
      cov.noalias() = diff * diff_temp / weights_(k);
      cov.diagonal().array() += p.reg_covar_;
      covs_.col(k) << Eigen::Map<VectorcX>(cov.data(), cov.size());
    }
  }

  inline void compute_precisions_cholesky(const Matrixck& covs,
                                          Matrixck& precs)
  {
    Matrixff cov = Matrixff::Zero(F,F);
    Eigen::LLT<Matrixff> lltOfCov;
    Matrixff choleskyL = Matrixff::Zero(F,F);
    Matrixff prec = Matrixff::Zero(F,F);
    VectorcX c = VectorcX::Zero(C,1);

    for (int k = 0; k < n_clusters_; ++k)
    {
      c = covs.col(k);
      cov << Eigen::Map<Matrixff>(c.data(), F, F);
      choleskyL = cov.llt().matrixL();
      prec.noalias() = (choleskyL.transpose() * choleskyL).ldlt().solve(choleskyL.transpose() * Matrixff::Identity(F, F));
      precs.col(k) << Eigen::Map<VectorcX>(prec.data(), prec.size());
    }
  }

  VectorkX weights_;
  Matrixfk means_;
  Matrixck covs_;
  Matrixck precs_;
  MatrixsX resp_T_;
  bool converged_;
  Parameters p;

  KMeans<T, F, K, S> kmeans_;

  VectorkX getWeights() const { return weights_; }
  Matrixfk getMeans() const { return means_; }
  Matrixck getCovs() const { return covs_; }
  Matrixck getPrecs() const { return precs_; }
  uint32_t getSize() const { return (uint32_t) n_clusters_; }

private:
  int n_clusters_;

};

#endif //SKLEARN_GMM
