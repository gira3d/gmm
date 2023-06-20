#ifndef GMM_3_BASE_H
#define GMM_3_BASE_H

#include <Eigen/Core>
#include <Eigen/Dense>

#include <gmm/GMMNBase.h>

namespace gmm_utils
{
  template <typename T, int K=-1, int S=-1>
    class GMM3Base : public GMMNBase<T,3,K,S>
    {
      public:

      using Ptr = std::shared_ptr<GMM3Base<T,K,S> >;
      using ConstPtr = std::shared_ptr<GMM3Base<T,K,S> const>;

      GMM3Base() : GMMNBase<T,3,K,S>() {}
      GMM3Base(const GMM3Base &d) : GMMNBase<T,3,K,S>(d) {}
      GMM3Base(const GMMNBase<T,3,K,S> &d) : GMMNBase<T,3,K,S>(d) {}

      // Apply transform to 3D GMM
      inline
      void transform(const Eigen::Affine3f& Tr)
      {
        Eigen::Matrix<T,3,3> R = Tr.rotation();
        Eigen::Matrix<T,3,1> t = Tr.translation();

        for (uint32_t i = 0; i < this->n_clusters_; ++i)
        {
          Eigen::Map< Eigen::Matrix<T,3,3> > cov(this->covs_.col(i).data(),3,3);
          Eigen::Matrix<T,3,3> cov_rotated = R * cov * R.transpose();
          this->covs_.col(i) << Eigen::Map< Eigen::Matrix<T,9,1> > (cov_rotated.data(), cov_rotated.size());
          this->means_.col(i) = R * this->means_.col(i) + t;
        }
      }

      inline
      void makeCovsIsoplanar()
      {
	Eigen::Matrix<float, 9, K> isoplanar_covs = this->covs_;

	for (int i = 0; i < this->covs_.cols(); ++i)
	{
	  Eigen::Matrix<float,9,1> temp = this->covs_.col(i);
	  Eigen::Matrix<float,3,3> cov = Eigen::Map<Eigen::Matrix<float,3,3>>(temp.data(),3,3);
	  Eigen::Matrix<float, 3, 3> D = Eigen::Matrix<float, 3, 3>::Identity();
	  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 3, 3>> eigensolver(cov);
	  D.diagonal() << 0.001, 1.0f, 1.0f;
	  Eigen::Matrix<float, 3, 3> U = (eigensolver.eigenvectors());
	  cov.noalias() = U * D * U.transpose();
	  isoplanar_covs.col(i) = Eigen::Map<Eigen::Matrix<float,9,1>>(cov.data(),9,1);
	}

	this->covs_ = isoplanar_covs;
      }
    };

  typedef GMM3Base<float,-1,-1> GMM3f;
  typedef GMM3Base<double,-1,-1> GMM3d;
}



#endif // GMM_3_BASE_H
