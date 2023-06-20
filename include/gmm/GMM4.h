#ifndef GMM_4_BASE_H
#define GMM_4_BASE_H

#include <Eigen/Core>
#include <Eigen/Dense>

#include <gmm/GMM3.h>
#include <gmm/GMMNBase.h>

namespace gmm_utils
{
  template <typename T, int K=-1, int S=-1>
  class GMM4Base : public GMMNBase<T,4,K,S>
    {
      public:

      using Ptr = std::shared_ptr<GMM4Base<T,K,S> >;
      using ConstPtr = std::shared_ptr<GMM4Base<T,K,S> const>;

      GMM4Base() : GMMNBase<T,4,K,S>() {}
      GMM4Base(const GMM4Base &d) : GMMNBase<T,4,K,S>(d) {}
      GMM4Base(const GMMNBase<T,4,K,S> &d) : GMMNBase<T,4,K,S>(d) {}
      ~GMM4Base() {}

      inline
      gmm_utils::GMM3Base<T> get3DGMM()
      {
	gmm_utils::GMM3Base<T> gmm3;

	gmm3.setMeans(this->getMeans().block(0,0,3,this->getNClusters()));
	gmm3.setWeights(this->getWeights());
	gmm3.setNClusters(this->getNClusters());
	gmm3.setSupportSize(this->getSupportSize());

	Eigen::Matrix<float, 9, K> covs = Eigen::Matrix<float,9,-1>::Zero(9, this->getNClusters());
	Eigen::Matrix<float, 9, K> precs = Eigen::Matrix<float,9,-1>::Zero(9, this->getNClusters());
	for (int i = 0; i < this->getNClusters(); ++i)
	{
	  // Covariance Matrices
	  Eigen::Matrix<float,16,1> cov = this->getCovs().col(i);
	  Eigen::Matrix<float,4,4> c;
	  c << Eigen::Map<Eigen::Matrix<float,4,4> >(cov.data(), 4, 4);

	  Eigen::Matrix<float,3,3,Eigen::ColMajor> c2 = c.block(0,0,3,3);
	  covs.col(i) << Eigen::Map<Eigen::Matrix<float,9,1,Eigen::ColMajor> >(c2.data(), c2.size());
	}

	for (int i = 0; i < this->getNClusters(); ++i)
	{
	  // Precision Matrices
	  Eigen::Matrix<float,16,1> prec = this->getPrecs().col(i);
	  Eigen::Matrix<float,4,4> p;
	  p << Eigen::Map<Eigen::Matrix<float,4,4> >(prec.data(), 4, 4);

	  Eigen::Matrix<float,3,3,Eigen::ColMajor> p2 = p.block(0,0,3,3);
	  precs.col(i) << Eigen::Map<Eigen::Matrix<float,9,1,Eigen::ColMajor> >(p2.data(), p2.size());
	}

	gmm3.setCovs(covs);
	gmm3.setPrecs(precs);
	gmm3.eigenDecomposition();

	return gmm3;
      }

      // Input:  in   = 3 x N matrix of spatial coordinates
      // Output: m_j  = N x K matrix where N is the number of input points and K is the number of GMM components
      //         s2_j = 1 x K vector where K is the number of GMM components. The sigma value is indepent of the
      //                input spatial point. It depends only on the covariance.

      inline
      void conditionalMeanVar(const Eigen::Matrix<T,3,-1>& in,

			      // Equation 4.7 in Sung thesis http://www.stat.rice.edu/~hgsung/thesis.pdf
			      Eigen::Matrix<T,-1,K>& m_j,

			      // Equation 4.8 in Sung thesis http://www.stat.rice.edu/~hgsung/thesis.pdf
			      Eigen::Matrix<T,-1,K>& s2_j)
      {
	std::cerr << "in.size(): " << in.rows() << ", " << in.cols() << std::endl;
	m_j = Eigen::Matrix<T,-1,K>::Zero(in.cols(),this->getNClusters());
	Eigen::Matrix<T,1,-1> mu_jY = this->getMeans().row(3);
	Eigen::Matrix<T,3,-1> mu_jX = this->getMeans().block(0,0,3,this->getNClusters());

	Eigen::Matrix<T,1,K> s_j = Eigen::Matrix<T,1,K>::Zero(1, this->getNClusters());
	s2_j = Eigen::Matrix<T,S,K>::Zero(in.cols(), this->getNClusters());

	for (int j = 0; j < this->getNClusters(); ++j)
	{
	  Eigen::Matrix<T,4,4>  S2 = Eigen::Map<Eigen::Matrix<T,4,4> > (this->getCovs().col(j).data(), 4, 4);
	  Eigen::Matrix<T,1,3>  S_jYX = S2.block(3,0,4,3);
	  Eigen::Matrix<T,3,1>  S_jXY = S2.block(0,3,3,4);
	  Eigen::Matrix<T,1,-1> S_jYY = this->getCovs().row(15);

	  Eigen::Matrix<T,4,4> P = Eigen::Map<Eigen::Matrix<T,4,4> > (this->getPrecs().col(j).data(), 4, 4);
	  Eigen::Matrix<T,3,3> P_jX = P.block(0,0,3,3).transpose() * P.block(0,0,3,3);

	  Eigen::Matrix<T,3,-1> in_mujX = in - mu_jX.col(j).replicate(1, in.cols());
	  Eigen::Matrix<T,1,-1> mu_jY_arr = mu_jY.col(j).replicate(1, in.cols());
	  m_j.col(j) << (mu_jY.col(j).replicate(1, in.cols()) +  S_jYX*P_jX*in_mujX).transpose();
	  s_j(j) = S_jYY(j) - S_jYX*P_jX*S_jXY;
	}

	s2_j = s_j.replicate(in.cols(), 1);
      }

      inline
      void conditionalWeights(const Eigen::Matrix<T,3,-1>& in,

			      // Equation 4.11 in Sung thesis
			      Eigen::Matrix<T,-1,K>& w_j)
      {
	gmm_utils::GMM3Base<T> gmm3 = this->get3DGMM();
	w_j = gmm3.posterior(in);
      }


      inline
      void regress(const Eigen::Matrix<T,3,-1>& in,

		   // Equation 4.12 in Sung thesis
		   Eigen::Matrix<T,-1,1>& m,

		   // Equation 4.13 in Sung thesis
		   Eigen::Matrix<T,-1,1>& v)
      {
	Eigen::Matrix<T,-1,K> w_j;
	this->conditionalWeights(in, w_j);

	Eigen::Matrix<T,-1,K> m_j;
	Eigen::Matrix<T,-1,K> s2_j;
	this->conditionalMeanVar(in, m_j, s2_j);

	// perform a coefficient-wise product instead of matrix multiplication
	// https://eigen.tuxfamily.org/dox/group__TutorialArrayClass.html
	Eigen::Matrix<T,-1,K>  m2 = w_j.array() * m_j.array();

	//sum along columns to get a N x 1 vector following 4.12 of Sung
	m = m2.array().rowwise().sum();

	// term 1 Equation 4.13 of Sung thesis
	Eigen::Matrix<T,-1,K> term1 = w_j.array() * (m_j.array() * m_j.array() + s2_j.array());

	// term 2 Equation 4.13 of Sung thesis
	Eigen::Matrix<T,-1,K> term2 = w_j.array() * m_j.array();

	// Equation 4.13
	v = term1.array().rowwise().sum() - term2.array().rowwise().sum().square();
      }

    protected:

    };

  typedef GMM4Base<float,-1,-1> GMM4f;
  typedef GMM4Base<double,-1,-1> GMM4d;
}

#endif // GMM_4_BASE_H
