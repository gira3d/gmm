#ifndef GMM_N_BASE_H
#define GMM_N_BASE_H

#include <iostream>
#include <fstream>
#include <random>

#include <boost/filesystem.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <gmm/SKLearn.h>

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
namespace gmm_utils {

  template <typename T, uint32_t N, int K=-1, int S=-1>
    class GMMNBase
    {
      public:

      using Ptr = std::shared_ptr<GMMNBase>;
      using ConstPtr = std::shared_ptr<GMMNBase const>;

      static constexpr uint32_t C = N*N;

      GMMNBase()
      {
        support_size_ = 0;
        n_clusters_ = 0;
      }

      GMMNBase(const GMMNBase& d)
      {
        support_size_ = d.getSupportSize();
        n_clusters_ = d.getNClusters();
        weights_ = d.getWeights();
        means_ = d.getMeans();
        covs_ = d.getCovs();
        precs_ = d.getPrecs();
        eigenvalues_ = d.getEigenvalues();
        eigenvectors_ = d.getEigenvectors();
      }
      ~GMMNBase() { }

      /* number of components in distribution */
      inline uint32_t getNClusters() const
      {
        return n_clusters_;
      }

      inline void setNClusters(uint32_t n_clusters)
      {
        n_clusters_ = n_clusters;
      }

      /* vector of weights for each density */
      inline Eigen::Matrix<T, K, 1> getWeights() const
      {
        return weights_;
      }

      inline void setWeights(const Eigen::Matrix<T,K,1>& weights)
      {
        weights_ = weights;
      }

      /* matrix of covariances for the distribution */
      inline Eigen::Matrix<T, C, K> getCovs() const
      {
        return covs_;
      }

      /* matrix of covariances for the distribution */
      inline Eigen::Matrix<T, C, K> getPrecs() const
      {
        return precs_;
      }

      inline void setCovs(const Eigen::Matrix<T,C,K>& covs)
      {
        covs_ = covs;
      }

      inline void setPrecs(const Eigen::Matrix<T,C,K>& precs)
      {
        precs_ = precs;
      }

      inline void setPrecs()
      {
	if (covs_.size())
	{
	  precs_ = Eigen::Matrix<T,C,K>::Zero(covs_.rows(), covs_.cols());
	  Eigen::Matrix<T, N, N> cov = Eigen::Matrix<T, N, N>::Zero(N,N);
	  Eigen::Matrix<T, N, N> choleskyL = Eigen::Matrix<T,N,N>::Zero(N,N);
	  Eigen::Matrix<T, N, N> prec = Eigen::Matrix<T,N,N>::Zero(N,N);
	  Eigen::Matrix<T, C, 1> c = Eigen::Matrix<T, C, 1>::Zero(C,1);

	  for (int k = 0; k < covs_.cols(); ++k)
	  {
	    c = covs_.col(k);
	    cov << Eigen::Map<Eigen::Matrix<T,N,N> >(c.data(), N, N);
	    choleskyL = cov.llt().matrixL();
	    prec.noalias() = (choleskyL.transpose() * choleskyL).ldlt().solve(choleskyL.transpose() * Eigen::Matrix<T,N,N>::Identity(N, N));
	    precs_.col(k) << Eigen::Map<Eigen::Matrix<T,C,1>>(prec.data(), prec.size());
	  }
	}
      }

      /* matrix of means for the distribution */
      inline Eigen::Matrix<T, N, K> getMeans() const
      {
        return means_;
      }

      inline void setMeans(const Eigen::Matrix<T,N,K>& means)
      {
        means_ = means;
      }

      /* n_features x n_clusters matrix of eigen values for the distribution */
      inline Eigen::Matrix<T, N, K> getEigenvalues() const
      {
        return eigenvalues_;
      }

      inline void setEigenvalues(const Eigen::Matrix<T,N,K>& evals)
      {
        eigenvalues_ = evals;
      }

      /* n_features^2 x n_clusters matrix of eigenvectors for the distribution */
      inline Eigen::Matrix<T, C, K> getEigenvectors() const
      {
        return eigenvectors_;
      }

      inline void setEigenvectors(const Eigen::Matrix<T,C,K>& evecs)
      {
        eigenvectors_ = evecs;
      }

      inline double getSupportSize() const
      {
        return support_size_;
      }

      inline void setSupportSize(const double support_size)
      {
        support_size_ = support_size;
      }

      /* set eigen vectors and eigen values */
      inline bool eigenDecomposition()
      {
        eigenvalues_ = Eigen::Matrix<T, N, -1>::Zero(N,n_clusters_);
        eigenvectors_ = Eigen::Matrix<T, C, -1>::Zero(C,n_clusters_);

        bool is_psd = true;
        for (uint32_t i = 0; i < n_clusters_; ++i)
          {
            Eigen::Matrix<T, N, N> cov(Eigen::Map<Eigen::Matrix<T, N, N> >(covs_.col(i).data(),N,N));
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, N, N> > eigensolver(cov);
            Eigen::Matrix<T, N, 1> D = eigensolver.eigenvalues();
            Eigen::Matrix<T, N, N> U = (eigensolver.eigenvectors());
            eigenvalues_.col(i) = D;
            eigenvectors_.col(i) = Eigen::Map<Eigen::Matrix<T,C,1> > (U.data(), C, 1);
            if (eigensolver.info() == Eigen::NumericalIssue)
              is_psd = false;
          }
        return is_psd;
      }

      /* checks if covariances are positive semi-definite */
      bool checkPSD()
      {
        bool is_psd = true;

        if (!eigenvalues_.size())
          eigenDecomposition();

        for (uint32_t i = 0; i < n_clusters_; ++i)
          {
            Eigen::Matrix<T,N,N> U(Eigen::Map<Eigen::Matrix<T,N,N> >(eigenvectors_.col(i).data(),N,N));
            Eigen::Matrix<T,N,1> eval = eigenvalues_.col(i);
            Eigen::Matrix<T,N,N> D = Eigen::Matrix<T,N,N>::Identity();

            D.diagonal() << eval;

            Eigen::Matrix<T,N,N> cov(Eigen::Map<Eigen::Matrix<T,N,N> >(covs_.col(i).data(),N,N));

            if ( ((U * D * U.transpose()) - cov ).array().abs().maxCoeff() > 1e-5 )
              {
                return false;
              }
          }
        return true;
      }

      // Sample from the distribution according to weight
      // Input: n_samples: number of points to sample
      //        samples: sample values
      //        n_samples_comp: components where samples come from
      void sample(const uint32_t n_samples,
                  Eigen::Matrix<T,N,-1>& samples,
                  std::vector<uint32_t>& n_samples_comp ) const
      {
        multinomial(n_samples, n_samples_comp);
        std::sort(n_samples_comp.begin(), n_samples_comp.end());

        std::default_random_engine generator;
        std::normal_distribution<float> distribution(0.0,1.0);

        std::vector<float> x;
        x.reserve(n_samples*N);

        while (x.size() < n_samples*N)
          {
            float rand_val = distribution(generator);

            // keeps points that are within 3 sigma of distribution
            // throws out all other points
            // TODO: may need to revisit this design decision
            if (std::abs(rand_val) < 3.0)
              x.push_back(rand_val);
          }

        // reformat x into 3 x n_samples vector
        samples = Eigen::Map<Eigen::Matrix<T,N,-1> >(x.data(), N, n_samples);

        for (uint32_t i = 0; i < n_samples_comp.size(); ++i)
          {
            Eigen::Matrix<T,C,1> C = eigenvectors_.col(n_samples_comp[i]);
            Eigen::Matrix<T,N,N> V = Eigen::Map<Eigen::Matrix<T,N,N> >(C.data(), N, N);
            Eigen::Matrix<T,N,N> E = eigenvalues_.col(n_samples_comp[i]).cwiseSqrt().asDiagonal();
            samples.col(i) << V * E * samples.col(i) + means_.col(n_samples_comp[i]);
          }
      }

      // Sample from the distribution according to weight
      // Input: n_samples: number of points to sample
      //        samples: sample values
      //        n_samples_comp: components where samples come from
      void fastSample(const uint32_t n_samples,
                      Eigen::Matrix<T,N,-1>& samples,
                      std::vector<uint32_t>& n_samples_comp ) const
      {
        fastMultinomial(n_samples, n_samples_comp);

        std::default_random_engine generator;
        std::normal_distribution<float> distribution(0.0,1.0);

        std::vector<float> x;
        x.reserve(n_samples*N);

        while (x.size() < n_samples*N)
        {
          float rand_val = distribution(generator);

          // keeps points that are within 3 sigma of distribution
          // throws out all other points
          // TODO: may need to revisit this design decision
          if (std::abs(rand_val) < 3.0)
            x.push_back(rand_val);
        }

        // reformat x into 3 x n_samples vector
        samples = Eigen::Map<Eigen::Matrix<T,N,-1> >(x.data(), N, n_samples);

        for (uint32_t i = 0; i < n_samples_comp.size(); ++i)
        {
          Eigen::Matrix<T,C,1> C = eigenvectors_.col(n_samples_comp[i]);
          Eigen::Matrix<T,N,N> V = Eigen::Map<Eigen::Matrix<T,N,N> >(C.data(), N, N);
          Eigen::Matrix<T,N,N> E = eigenvalues_.col(n_samples_comp[i]).cwiseSqrt().asDiagonal();
          samples.col(i) << V * E * samples.col(i) + means_.col(n_samples_comp[i]);
        }
      }

      // Sample from the distribution according to weight
      // Input: n_samples: number of points to sample
      //        samples: sample values
      //        n_samples_comp: components where samples come from
      void veryFastSample(const uint32_t n_samples,
			  const std::vector<float>& random_numbers,
			  Eigen::Matrix<T,N,-1>& samples,
			  std::vector<uint32_t>& n_samples_comp ) const
      {
        fastMultinomial(n_samples, n_samples_comp);

        std::vector<float> x;
        x.reserve(n_samples*N);

	int count = 0;
        while (x.size() < n_samples*N)
        {
          float rand_val = random_numbers[count % random_numbers.size()]; ++count;

          // keeps points that are within 3 sigma of distribution
          // throws out all other points
          // TODO: may need to revisit this design decision
          if (std::abs(rand_val) < 3.0)
            x.push_back(rand_val);
        }

        // reformat x into 3 x n_samples vector
        samples = Eigen::Map<Eigen::Matrix<T,N,-1> >(x.data(), N, n_samples);

        for (uint32_t i = 0; i < n_samples_comp.size(); ++i)
        {
          Eigen::Matrix<T,C,1> C = eigenvectors_.col(n_samples_comp[i]);
          Eigen::Matrix<T,N,N> V = Eigen::Map<Eigen::Matrix<T,N,N> >(C.data(), N, N);
          Eigen::Matrix<T,N,N> E = eigenvalues_.col(n_samples_comp[i]).cwiseSqrt().asDiagonal();
          samples.col(i) << V * E * samples.col(i) + means_.col(n_samples_comp[i]);
        }
      }

      /* kmeans and expectation maximization */
      template<typename EM_TYPE = SKLearn<T,N,K,-1> >
      void fit(const Eigen::Matrix<T,N,-1>& X, int n_clusters = K)
      {
        EM_TYPE em = EM_TYPE();
        em.fit(X, n_clusters);

        setWeights(em.getWeights());
        setMeans(em.getMeans());
        setCovs(em.getCovs());
        setPrecs(em.getPrecs());
        setNClusters(weights_.size());
        eigenDecomposition();
        setSupportSize(X.cols());
      }

      /* kmeans and expectation maximization */
      template<typename EM_TYPE = SKLearn<T,N,K,-1> >
      void fit(const Eigen::Matrix<T,N,-1>& X, Eigen::Matrix<T,-1, 1>& labels, int n_clusters = K)
      {
        EM_TYPE em = EM_TYPE();
        em.fit(X, labels, n_clusters);

        setWeights(em.getWeights());
        setMeans(em.getMeans());
        setCovs(em.getCovs());
        setPrecs(em.getPrecs());
        setNClusters(weights_.size());
        eigenDecomposition();
        setSupportSize(X.cols());
      }

      /* kmeans and expectation maximization */
      template<typename EM_TYPE = SKLearn<T,N,K,-1> >
      void fit(const Eigen::Matrix<T,N,-1>& X,
               const float mahalanobis_distance,
               int n_clusters = K)
      {
        EM_TYPE em = EM_TYPE();
        em.fit(X, mahalanobis_distance, n_clusters);

        setWeights(em.getWeights());
        setMeans(em.getMeans());
        setCovs(em.getCovs());
        setPrecs(em.getPrecs());
        setNClusters(weights_.size());
        eigenDecomposition();
        setSupportSize(X.cols());
      }

      /* kmeans and expectation maximization */
      template<typename EM_TYPE = SKLearn<T,N,K,-1> >
      void fit(const Eigen::Matrix<T,N,-1>& X,
               Eigen::Matrix<T,-1, 1>& labels,
               const float mahalanobis_distance,
               int n_clusters = K)
      {
        EM_TYPE em = EM_TYPE();
        em.fit(X, labels, mahalanobis_distance, n_clusters);

        setWeights(em.getWeights());
        setMeans(em.getMeans());
        setCovs(em.getCovs());
        setPrecs(em.getPrecs());
        setNClusters(weights_.size());
        eigenDecomposition();
        setSupportSize(X.cols());
      }

      /* save GMM to file with data as < mean, covs, weights> */
      //TODO: add precs here
      void save(const std::string& filepath) const
      {
        /*
        if (!boost::filesystem::exists(filepath))
        {
          std::cerr << "GMMNBase: file path does not exist." << std::endl;
          return;
        }
        */

        const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision,
                                               Eigen::DontAlignCols,
                                               ", ", "\n");

        Eigen::Matrix<T,-1,N> means = means_.transpose();
        Eigen::Matrix<T,-1,C> covs = covs_.transpose();
        Eigen::Matrix<T,-1,1> weights = weights_;

        Eigen::Matrix<T,-1,-1> data(means.rows(),
                                    means.cols()+covs.cols()+weights.cols());

       data << means, covs, weights;

       std::ofstream file(filepath.c_str());
       file << data.format(CSVFormat);
      }

      /* load GMM from file */
      // TODO: add precs here
      void load(const std::string& filepath)
      {
        if (!boost::filesystem::exists(filepath))
        {
          std::cerr << "GMMNBase: file path does not exist." << std::endl;
          return;
        }

        Eigen::Matrix<T,-1,-1> data = loadCSV<Eigen::Matrix<T,-1,-1>, T>(filepath);
        data.transposeInPlace();

        if (data.cols() != (1+N+C))
          std::cerr << "GMMNBase: data has incorrect size" << std::endl;

        means_ = data.leftCols(N).transpose();
        Eigen::Matrix<T,-1,-1> covsweights = data.rightCols(C+1);
        covs_ = covsweights.leftCols(C).transpose();
        weights_ = covsweights.rightCols(1);

        n_clusters_ = weights_.size();
      }


      // we can only merge into a GMM that has Eigen::Dynamic clusters
      //TODO: check this function for correctness.
      template<int A, int B>
      void merge(const GMMNBase<T, N, A, B>& gmm)
      {
        static_assert((K == -1), "merge undefined for K >= 0");

        if (n_clusters_ <= 0)
        {
          means_ = gmm.getMeans();
          covs_ = gmm.getCovs();
          precs_ = gmm.getPrecs();
          eigenvalues_ = gmm.getEigenvalues();
          eigenvectors_ = gmm.getEigenvectors();
          weights_ = gmm.getWeights();
          n_clusters_ = gmm.getNClusters();
          support_size_ = gmm.getSupportSize();
        }
        else
        {
          double total_support_size = gmm.getSupportSize() + support_size_;

          Eigen::Matrix<T, N, -1> means_temp = Eigen::Matrix<T,N,-1>::Zero(N, means_.cols()+gmm.getMeans().cols());
          means_temp.block(0,0,N,means_.cols()) = means_;
          means_temp.block(0,means_.cols(),N,gmm.getMeans().cols()) = gmm.getMeans();
          means_ = means_temp;

          Eigen::Matrix<T, C, -1> covs_temp = Eigen::Matrix<T,C,-1>::Zero(C, covs_.cols()+gmm.getCovs().cols());
          covs_temp.block(0,0,C,covs_.cols()) = covs_;
          covs_temp.block(0,covs_.cols(),C,gmm.getCovs().cols()) = gmm.getCovs();
          covs_ = covs_temp;

          Eigen::Matrix<T, C, -1> precs_temp = Eigen::Matrix<T,C,-1>::Zero(C, precs_.cols()+gmm.getPrecs().cols());
          precs_temp.block(0,0,C,precs_.cols()) = precs_;
          precs_temp.block(0,precs_.cols(),C,gmm.getPrecs().cols()) = gmm.getPrecs();
          precs_ = precs_temp;

          Eigen::Matrix<T, N, -1> eigenvalues_temp = Eigen::Matrix<T,N,-1>::Zero(N, eigenvalues_.cols()+gmm.getEigenvalues().cols());
          eigenvalues_temp.block(0,0,N,eigenvalues_.cols()) = eigenvalues_;
          eigenvalues_temp.block(0,eigenvalues_.cols(),N,gmm.getEigenvalues().cols()) = gmm.getEigenvalues();
          eigenvalues_ = eigenvalues_temp;

          Eigen::Matrix<T, C, -1> eigenvectors_temp = Eigen::Matrix<T,C,-1>::Zero(C, eigenvectors_.cols()+gmm.getEigenvectors().cols());
          eigenvectors_temp.block(0,0,C,eigenvectors_.cols()) = eigenvectors_;
          eigenvectors_temp.block(0,eigenvectors_.cols(),C,gmm.getEigenvectors().cols()) = gmm.getEigenvectors();
          eigenvectors_ = eigenvectors_temp;

          Eigen::Matrix<T,-1,1> weights_temp = Eigen::Matrix<T,-1,1>::Zero(weights_.rows() + gmm.getWeights().rows(), 1);
          weights_temp.block(0,0,weights_.rows(),1) = (weights_ * support_size_) / total_support_size;
          weights_temp.block(weights_.rows(),0,gmm.getWeights().rows(),1) = (gmm.getWeights() * gmm.getSupportSize()) / total_support_size;
          weights_ = weights_temp;

          n_clusters_ += gmm.getNClusters();
          support_size_ += gmm.getSupportSize();
        }
      }

      typename GMMNBase<T,N,-1,S>::Ptr exciseComponents(const std::vector<uint32_t>& component_ids)
      {
	std::vector<uint32_t> submap_ids;
	for (uint32_t key = 0; key < getNClusters(); ++key)
	  if (!std::count(component_ids.begin(), component_ids.end(), key))
	    submap_ids.push_back(key);
	return getSubmap(submap_ids);
      }

      // extract submap based on component ids
      typename GMMNBase<T,N,-1,S>::Ptr getSubmap(const std::vector<uint32_t>& component_ids)
      {
        typename GMMNBase<T,N,-1,S>::Ptr gmm = typename GMMNBase<T,N,-1,S>::Ptr(new GMMNBase<T,N,-1,S>());

        uint32_t size = component_ids.size();

        Eigen::Matrix<float, -1, 1> weights;
        weights.resize(size, 1);

        Eigen::Matrix<float, N, -1> means;
        means.resize(N, size);

        Eigen::Matrix<float, C, -1> covs;
        covs.resize(C, size);

        Eigen::Matrix<float, C, -1> precs;
        precs.resize(C, size);

        Eigen::Matrix<float, N, -1> evals;
        evals.resize(N, size);

        Eigen::Matrix<float, C, -1> evecs;
        evecs.resize(C, size);

        for (uint32_t i = 0; i < size; ++i)
        {
          if (component_ids[i] > n_clusters_) continue;

          weights.row(i) = weights_.row(component_ids[i]);
          means.col(i) << means_.col(component_ids[i]);
          covs.col(i) << covs_.col(component_ids[i]);
          precs.col(i) << precs_.col(component_ids[i]);
          evals.col(i) << eigenvalues_.col(component_ids[i]);
          evecs.col(i) << eigenvectors_.col(component_ids[i]);
        }

        gmm->setMeans(means);
        gmm->setCovs(covs);
        gmm->setPrecs(precs);
        gmm->setEigenvalues(evals);
        gmm->setEigenvectors(evecs);
        gmm->setNClusters(size);
        gmm->setSupportSize(support_size_ * weights.array().sum());
        gmm->setWeights(weights / weights.array().sum());

        return gmm;
      }

      /* helper function to sample from the distribution */
      void fastMultinomial(const uint32_t n_samples,
                           std::vector<uint32_t>& n_samples_comp) const
      {
        Eigen::Matrix<T,-1,1> weights = weights_ * n_samples;

        n_samples_comp.clear();
        n_samples_comp.reserve(n_samples);

        for (uint32_t i = 0; i < weights.size(); ++i)
        {
          for (uint32_t j = 0; j < std::floor(weights[i]); ++j)
          {
            n_samples_comp.push_back(i);
          }
        }

        for (uint32_t i = n_samples_comp.size(); i < n_samples; ++i)
        {
          n_samples_comp.push_back(i % weights_.size());
        }
      }

      /* helper function to sample from the distribution */
      void multinomial(const uint32_t n_samples,
                       std::vector<uint32_t>& n_samples_comp) const
      {
        std::random_device rd;
        std::mt19937 rng(rd());
        std::discrete_distribution<uint32_t>
        dist(weights_.data(), weights_.data() + weights_.size());

        n_samples_comp.clear();
        n_samples_comp.reserve(n_samples);

        for (uint32_t i = 0; i < n_samples; ++i)
          n_samples_comp.push_back(dist(rng));
      }

      float bic(const Eigen::Matrix<T,N,-1>& X)
      {
        float cov_params = n_clusters_ * N * (N+1)/2;
        float mean_params = N * n_clusters_;
        int n_parameters = int(cov_params + mean_params + n_clusters_ - 1);
        return (-2 * score(X) * X.cols() + n_parameters * std::log(X.cols()));
      }

      float aic(const Eigen::Matrix<T,N,-1>& X)
      {
        float cov_params = n_clusters_ * N * (N+1)/2;
        float mean_params = N * n_clusters_;
        int n_parameters = int(cov_params + mean_params + n_clusters_ - 1);
        return (-2 * score(X) * X.cols() + 2 * n_parameters);
      }

      double pri(const GMMNBase& that) const
      {
        T DEN = std::pow(2 * M_PI, N / 2.0);
        T DEN_INV = 1.0 / DEN;

        Eigen::Matrix<T, N, K> mu = means_;
        Eigen::Matrix<T, C, K> Linv = covs_;
        Eigen::Matrix<T, K, 1> pi = weights_;

        Eigen::Matrix<T, N, K> nu = that.getMeans();
        Eigen::Matrix<T, C, K> Oinv = that.getCovs();
        Eigen::Matrix<T, K, 1> tau = that.getWeights();

        double term1 = 0.0;

        //term 1
        for (uint32_t m = 0; m < pi.rows(); ++m)
          for (uint32_t k = 0; k < tau.rows(); ++k)
            term1 += pi(m)*tau(k)*zmk(mu.col(m) - nu.col(k), Linv.col(m) + Oinv.col(k));

        return term1;
      }

      // Based on http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.643.4088&rep=rep1&type=pdf
      double dcs(const GMMNBase& that) const
      {
        T DEN = std::pow(2 * M_PI, N / 2.0);
        T DEN_INV = 1.0 / DEN;

        Eigen::Matrix<T, N, K> mu = means_;
        Eigen::Matrix<T, C, K> Linv = covs_;
        Eigen::Matrix<T, K, 1> pi = weights_;

        Eigen::Matrix<T, N, K> nu = that.getMeans();
        Eigen::Matrix<T, C, K> Oinv = that.getCovs();
        Eigen::Matrix<T, K, 1> tau = that.getWeights();

        double term1 = 0.0, term2 = 0.0, term3 = 0.0;

        //term 1
        for (uint32_t m = 0; m < pi.rows(); ++m)
          for (uint32_t k = 0; k < tau.rows(); ++k)
            term1 += pi(m)*tau(k)*zmk(mu.col(m) - nu.col(k), Linv.col(m) + Oinv.col(k));

        //term 2
        for (uint32_t m = 0; m < pi.rows(); ++m)
        {
          Eigen::Matrix<T,C,1> cov = Linv.col(m) + Linv.col(m); // Note: this line differs from the paper but is correct
          Eigen::Matrix<T,N,N> Lm_inv = Eigen::Map<Eigen::Matrix<T,N,N> > (cov.data(), N, N);
          Eigen::Matrix<T,N,N> Lm = Lm_inv.inverse();
          term2 += pi(m)*pi(m)*std::sqrt(Lm.determinant()) * DEN_INV ;
        }

        for (uint32_t m = 0; m < pi.rows(); ++m)
          for (uint32_t k = 0; k < m; ++k)
            term2 += 2*pi(m)*pi(k)*(zmk(mu.col(m) - mu.col(k), Linv.col(m) + Linv.col(k)));

        //term 3
        for (uint32_t k = 0; k < tau.rows(); ++k)
        {
          Eigen::Matrix<T,C,1> cov = Oinv.col(k) + Oinv.col(k); // Note: this line differs from the paper but is correct
          Eigen::Matrix<T,N,N> Ok_inv = Eigen::Map<Eigen::Matrix<T,N,N> > (cov.data(), N, N);
          Eigen::Matrix<T,N,N> Ok = Ok_inv.inverse();
          term3 += tau(k)*tau(k)*std::sqrt(Ok.determinant()) * DEN_INV ;
        }

        for (uint32_t k = 0; k < tau.rows(); ++k)
          for (uint32_t m = 0; m < k; ++m)
            term3 += 2*tau(k)*tau(m)*(zmk(nu.col(k) - nu.col(m), Oinv.col(k) + Oinv.col(m)));

        return -std::log(term1) + 0.5*std::log(term2) + 0.5*std::log(term3);
      }

      // This is the logLikelihood function according to (3.5) in
      // http://reports-archive.adm.cs.cmu.edu/anon/2019/CMU-CS-19-108.pdf
      T logLikelihood(const Eigen::Matrix<T,N,-1>& X)
      {
	return weightedLogProbability(X).sum();
      }

      // This function calculates everything to the right of and including the natural
      // log inside of Equation (3.5) in  http://reports-archive.adm.cs.cmu.edu/anon/2019/CMU-CS-19-108.pdf
      // This function matches the logLikelihood function in Shobhit's codebase.
      Eigen::Matrix<T,-1,1> weightedLogProbability(const Eigen::Matrix<T,N,-1>& X)
      {
	return scoreSamples(X);
      }

      // This function calculates (3.7) in http://reports-archive.adm.cs.cmu.edu/anon/2019/CMU-CS-19-108.pdf#page=38
      Eigen::Matrix<T,S,K> posterior(const Eigen::Matrix<T,N,-1>& X)
      {
	// This is the first term in (3.7)
        Eigen::Matrix<T,K,1> log_weights = weights_.template cast<double>().array().log().template cast<T>();

	// This is the second term in (3.7)
	Eigen::Matrix<T,K,S> log_gauss_prob = logGaussProb(X);

	// This is the third term in (3.7)
        Eigen::Matrix<T,K,S> weighted_log_prob = log_gauss_prob.array().colwise() + log_weights.array();
        Eigen::Matrix<double,1,S> weighted_prob_sum = weighted_log_prob.array().template cast<double>().exp().colwise().sum();
        Eigen::Matrix<T,S,1> weighted_log_prob_sum = weighted_prob_sum.array().log().template cast<T>().transpose();

	// Sum the terms
	Eigen::Matrix<T,S,K> weighted_log_prob_T = weighted_log_prob.transpose();
	Eigen::Matrix<T,S,K> log_posterior = weighted_log_prob_T.array().colwise() - weighted_log_prob_sum.array();
	Eigen::Matrix<T,S,K> posterior = log_posterior.template cast<double>().array().exp().template cast<T>();

	return posterior;
      }

      protected:

      // This function calculates (3.11) for every point and density, which results in a matrix of size K x S
      // http://reports-archive.adm.cs.cmu.edu/anon/2019/CMU-CS-19-108.pdf#page=38
      Eigen::Matrix<T,K,S> logGaussProb(const Eigen::Matrix<T,N,-1>& X)
      {
        Eigen::Matrix<T,N,N> prec_chol = Eigen::Matrix<T,N,N>::Zero(N,N);
        Eigen::Matrix<T,C,1> p = Eigen::Matrix<T,C,1>::Zero(C,1);
        Eigen::Matrix<T,S,K> log_prob = Eigen::Matrix<T,-1, K>::Zero(X.cols(), n_clusters_); // S x K

        for (int k = 0; k < n_clusters_; ++k)
        {
          p = precs_.col(k);
          prec_chol << Eigen::Map<Eigen::Matrix<T,N,N> >(p.data(), N, N);
          log_prob.col(k) << ((prec_chol * X).colwise() - (prec_chol * means_.col(k))).array().square().colwise().sum().transpose();
        }
	Eigen::Matrix<T,K,1> log_det_chol = Eigen::Matrix<T,K,1>::Zero(n_clusters_,1);
        for (int k = 0; k < n_clusters_; ++k)
        {
          Eigen::Matrix<T,N,N> chol = Eigen::Map<Eigen::Matrix<T,N,N> > (precs_.col(k).data(), N, N);
          log_det_chol(k) = (T) (chol.diagonal().template cast<double>().array().log().sum());
        }
        Eigen::Matrix<T,K,S> log_gauss_prob = (-0.5 * ((N * LOG_2_M_PI) + log_prob.array())).transpose().colwise() + log_det_chol.array();

	return log_gauss_prob;
      }

      // This is the weighted log probability.
      Eigen::Matrix<T,S,1> scoreSamples(const Eigen::Matrix<T,N,S>& X)
      {
	Eigen::Matrix<T,K,S> log_gauss_prob = logGaussProb(X);
        Eigen::Matrix<T,K,S> weighted_log_prob = log_gauss_prob.array().colwise() + weights_.template cast<double>().array().log().template cast<T>();
        if (weighted_log_prob.array().isInf().sum())
        {
	  //Note: Because of the template cast<double>() throughout this function, this case should almost never happen unless
	  // the points in X are very, very far away from the distribution.
          std::cerr << "WARNING: weighted_log_prob in scoreSamples is infinite due to lack of precision." << std::endl;
        }
        Eigen::Matrix<double,1,S> weighted_log_prob_sum = weighted_log_prob.array().template cast<double>().exp().colwise().sum();
        Eigen::Matrix<T,S,1> score_samples = weighted_log_prob_sum.array().log().template cast<T>().transpose();

        return score_samples;
      }

      float score(const Eigen::Matrix<T,N,-1>& X)
      {
        return scoreSamples(X).mean();
      }

      T zmk(const Eigen::Matrix<T, N, 1>& mean, const Eigen::Matrix<T, C, 1>& cov) const
      {
        Eigen::Matrix<T,C,1> c2 = cov;
        Eigen::Matrix<T,N,N> c = Eigen::Map<Eigen::Matrix<T,N,N>>(c2.data(), N, N);
        Eigen::Matrix<T,N,N> prec = c.inverse();
        T term = std::sqrt(prec.determinant()) * (1 / std::pow((2*M_PI), N / 2.0)) * exp ( -0.5 * mean.transpose() * prec * mean );
        return term;
      }

      /* load comma-separated file of the gmm */
      template<typename A, typename M>
      A loadCSV (const std::string & path) {
        std::ifstream indata;
        indata.open(path);
        std::string line; std::vector<M> values;
        uint rows = 0;
        while (std::getline(indata, line)) {
          std::stringstream lineStream(line);
          std::string cell;
          while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
          }
          ++rows;
        }
        return Eigen::Map<const Eigen::Matrix<typename A::Scalar, A::RowsAtCompileTime, A::ColsAtCompileTime, Eigen::ColMajor>>(values.data(), values.size()/rows, rows);
      }

      Eigen::Matrix<T, N, K> means_;
      Eigen::Matrix<T, C, K> covs_;

      // WARNING: precs_ is NOT the precision matrix.
      // It stores the inverse of the lower triangular cholesky
      // decomposition of the covariance matrix.
      //
      // See (3.8 - 3.10) of http://reports-archive.adm.cs.cmu.edu/anon/2019/CMU-CS-19-108.pdf#page=38
      // for detailed explanation.
      //
      // To retrieve the kth precision matrix, use the following:
      //
      // Eigen::Matrix<T,N,N> Linv = Eigen::Map<Eigen::Matrix<T,N,N> > (precs_.col(k).data(), N, N);
      // Eigen::Matrix<T,N,N> precision_k = Linv.transpose() * Linv;
      Eigen::Matrix<T, C, K> precs_;
      Eigen::Matrix<T, K, 1> weights_;

      Eigen::Matrix<T, N, K> eigenvalues_;
      Eigen::Matrix<T, C, K> eigenvectors_;

      uint32_t n_clusters_;
      double support_size_;
    };
}


#endif //GMM_N_BASE
