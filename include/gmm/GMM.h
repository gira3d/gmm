#ifndef GMM_H
#define GMM_H

#include <ros/ros.h>

#include <Eigen/Core>
#include <Eigen/Dense>

/*****************************************************************************
 * We choose column-major storage because it is much better supported by
 * Eigen. From the manual: The default in Eigen is column-major. Naturally,
 * most of the development and testing of the Eigen library is thus done with
 * column-major matrices. This means that, even though we aim to support
 * column-major and row-major storage orders transparently, the Eigen library
 * may well work best with column-major matrices.
 *
 * See https://eigen.tuxfamily.org/dox/group__TopicStorageOrders.html for
 * more information
 *
 *****************************************************************************/

class GMM
{
public:

  typedef boost::shared_ptr<GMM> Ptr;
  typedef boost::shared_ptr<GMM const> ConstPtr;

  typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatrixX;
  typedef Eigen::Matrix<float, 3, 3, Eigen::ColMajor> Matrix3X;
  typedef Eigen::Matrix<float, Eigen::Dynamic, 1, Eigen::ColMajor> VectorX;
  typedef Eigen::Matrix<float, 3, 1, Eigen::ColMajor> Vector3X;
  typedef Eigen::Matrix<float, 9, 1, Eigen::ColMajor> Vector9X;
  typedef Eigen::Transform<float, 3, Eigen::Affine, Eigen::ColMajor> Transform3f;

  GMM() { }
  ~GMM() { }

  uint32_t getSize() const {return size_;}
  VectorX getWeights() const {return weights_;}
  MatrixX getCovs() const {return covs_;}
  MatrixX getMeans() const {return means_;}
  MatrixX getEigenvalues() const {return eigenvalues_;}
  MatrixX getEigenvectors() const {return eigenvectors_;}
  Vector3X multivariate_normal_sample(const Vector3X&,
                                      const Vector9X&,
                                      const Vector3X&) const;

  template <typename EM>
  void train(const MatrixX& X)
  {
    EM em;
    em.fit(X);
    setWeights(em.getWeights());
    setMeans(em.getMeans());
    setCovs(em.getCovs());
    setSize(weights_.size());
    eigenDecomposition();
  }

  void setSize(uint32_t size) {size_ = size;}
  void setWeights(const MatrixX& weights) {weights_ = weights;}
  void setCovs(const MatrixX& covs) {covs_ = covs;}
  void setMeans(const MatrixX& means) {means_ = means;}
  void setParameters(const MatrixX& means, const MatrixX& covs,
                     const VectorX& weights);

  void save(const std::string& filepath) const;
  void load(const std::string& filepath);

  void computeIsoplanarCovs();
  void useAnisotropicCovs();
  void useIsoplanarCovs();
  bool usingAnisotropicCovs() {return using_aniso_covs_;}
  bool usingIsoPlanarCovs() {return using_isopl_covs_;}

  bool eigenDecomposition();
  bool checkPSD();
  MatrixX sample(const uint32_t n_samples) const;
  void multinomial(const uint32_t n_samples, std::vector<uint32_t>& n_samples_comp) const;
  void transform(const Eigen::Affine3f& Tr);

private:
  MatrixX loadCSV(const std::string& path) const;
  MatrixX means_, covs_, weights_;
  MatrixX anisotropic_covs_;
  MatrixX isoplanar_covs_;
  uint32_t size_;
  bool using_aniso_covs_;
  bool using_isopl_covs_;
  MatrixX eigenvalues_;
  MatrixX eigenvectors_;
};

#endif //GMM
