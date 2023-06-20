/*
  gmm_mex: transforms a data to a GMM via MATLAB
  Copyright (C) 2018 Wennie Tabib

  This program is free software; you can redistribute it and/or
  modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation; either version 2
  of the License, or (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*/
#include<mex_interface_matlab/mex_class_handle.hpp>
#include<mex_interface_matlab/ConversionUtils.h>
#include<mex_interface_matlab/PrintToScreen.h>
#include <Eigen/Geometry>

#include <gmm/GMM3.h>
#include <gmm/SKLearn.h>

namespace cu=conversion_utils;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::PointXYZ Point;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatrixX;
typedef gmm_utils::GMM3Base<float> GMM;
typedef SKLearn<float,3,-1,-1> SK;

void
mexFunction(int nlhs, mxArray *plhs[],
            int nrhs, const mxArray *prhs[])
{
  // Get the command string
  char cmd[128];
  if (nrhs < 1 || mxGetString(prhs[0], cmd, sizeof(cmd)))
    mexErrMsgTxt("First input should be a command string less than 64 characters long.");

  if (!strcmp("new", cmd))
  {

    if (nrhs != 3)
    {
      mexErrMsgTxt("New: inputs required: <\"new\">, data, n_clusters");
      return;
    }

    uint32_t idx = 1;

    MatrixX X;
    cu::matlabToEigen(prhs, idx, X); ++idx;

    if (X.rows() != 3)
    {
      mexErrMsgTxt("input data must be 3xn");
      return;
    }

    int n_clusters = cu::matlabToScalar(prhs, idx); ++idx;

    GMM gmm;
    gmm.fit<SK>(X, n_clusters);

    cu::eigenToMATLAB(gmm.getWeights(), 0, plhs);
    cu::eigenToMATLAB(gmm.getMeans(), 1, plhs);
    cu::eigenToMATLAB(gmm.getCovs(), 2, plhs);
    plhs[3] = convertPtr2Mat<GMM>(new GMM(gmm));
    return;
  }

  if (!strcmp("create_mex_object", cmd))
  {

    if (nrhs != 5)
    {
      mexErrMsgTxt("CreateMexObject: inputs required: <\"create_mex_object\">, means, covs, weights, support_size");
      return;
    }

    uint32_t idx = 1;

    MatrixX means;
    MatrixX covs;
    MatrixX weights;
    cu::matlabToEigen(prhs, idx, means); ++idx;
    cu::matlabToEigen(prhs, idx, covs); ++idx;
    cu::matlabToEigen(prhs, idx, weights); ++idx;
    float support_size = cu::matlabToScalar(prhs, idx); ++idx;

    if (means.rows() != 3)
    {
      mexErrMsgTxt("input data must be 3xn");
      return;
    }

    if (covs.rows() != 9)
    {
      mexErrMsgTxt("input data must be 9xn");
      return;
    }

    if (weights.cols() != 1)
    {
      mexErrMsgTxt("input data must be nx1");
      return;
    }

    GMM gmm;
    gmm.setNClusters(weights.rows());
    gmm.setWeights(weights);
    gmm.setMeans(means);
    gmm.setCovs(covs);
    gmm.setPrecs();
    gmm.eigenDecomposition();
    gmm.setSupportSize(support_size);

    plhs[0] = convertPtr2Mat<GMM>(new GMM(gmm));
    return;
  }

  if (!strcmp("get_parameters", cmd))
  {
    if (nrhs != 2)
    {
      mexErrMsgTxt("getParameters: inputs required: <\"get_parameters\", this.objectHandle");
      return;
    }

    GMM* gmm = convertMat2Ptr<GMM>(prhs[1]);

    cu::eigenToMATLAB(gmm->getWeights(), 0, plhs);
    cu::eigenToMATLAB(gmm->getMeans(), 1, plhs);
    cu::eigenToMATLAB(gmm->getCovs(), 2, plhs);
    cu::scalarToMATLAB<uint32_t>(gmm->getNClusters(), 3, plhs);
    cu::scalarToMATLAB<double>(gmm->getSupportSize(), 4, plhs);
    return;
  }

  if (!strcmp("get_precs", cmd))
  {
    if (nrhs != 2)
    {
      mexErrMsgTxt("getPrecs: inputs required: <\"get_precs\", this.objectHandle");
      return;
    }

    GMM* gmm = convertMat2Ptr<GMM>(prhs[1]);
    cu::eigenToMATLAB(gmm->getPrecs(), 0, plhs);
    return;
  }

  if (!strcmp("set_precs", cmd))
  {
    if (nrhs != 2)
    {
      mexErrMsgTxt("setPrecs: inputs required: <\"set_precs\", this.objectHandle");
      return;
    }

    GMM* gmm = convertMat2Ptr<GMM>(prhs[1]);
    gmm->setPrecs();
    return;
  }

  if (!strcmp("aic", cmd))
  {
    if (nrhs != 3)
    {
      mexErrMsgTxt("aic: inputs required: <\"get_parameters\", this.objectHandle, X>");
      return;
    }

    GMM* gmm = convertMat2Ptr<GMM>(prhs[1]);
    Eigen::MatrixXf X;
    cu::matlabToEigen(prhs, 2, X);

    float score = gmm->aic(X);

    cu::scalarToMATLAB<float>(score, 0, plhs);
    return;
  }

  if (!strcmp("bic", cmd))
  {
    if (nrhs != 3)
    {
      mexErrMsgTxt("bic: inputs required: <\"get_parameters\", this.objectHandle, X>");
      return;
    }

    GMM* gmm = convertMat2Ptr<GMM>(prhs[1]);
    Eigen::MatrixXf X;
    cu::matlabToEigen(prhs, 2, X);
    float score = gmm->bic(X);

    cu::scalarToMATLAB<float>(score, 0, plhs);
    return;
  }

  if (!strcmp("fast_sample", cmd))
  {
    if (nrhs != 3)
    {
      mexErrMsgTxt("fast_sample: inputs required: <\"get_parameters\", this.objectHandle, n>");
      return;
    }

    GMM* gmm = convertMat2Ptr<GMM>(prhs[1]);
    uint32_t n = cu::matlabToScalar(prhs, 2);
    Eigen::Matrix<float, 3, -1> samples = Eigen::Matrix<float,3,-1>::Zero(3, n);
    std::vector<uint32_t> n_samples_comp;
    gmm->sample(n, samples, n_samples_comp);

    cu::eigenToMATLAB(samples, 0, plhs);
    return;
  }

  if (!strcmp("transform", cmd))
  {
    if (nrhs != 3)
    {
      mexErrMsgTxt("transform: inputs required: <\"transform\", this.objectHandle, T>");
      return;
    }

    GMM* gmm = convertMat2Ptr<GMM>(prhs[1]);
    Eigen::MatrixXf Tr;
    cu::matlabToEigen(prhs, 2, Tr);

    Eigen::Affine3f T;
    T.linear() = Tr.block(0,0,3,3);
    T.translation() = Tr.block(0,3,3,1);

    gmm->transform(T);
    return;
  }

  if (!strcmp("load_from_file", cmd))
  {
    if (nrhs != 2)
    {
      mexErrMsgTxt("transform: inputs required: <\"load_from_file\", filepath>");
      return;
    }

    char filestr[2048];
    mxGetString(prhs[1], filestr, sizeof(filestr));
    std::string filepath(filestr);

    plhs[0] = convertPtr2Mat<GMM>(new GMM());
    GMM* gmm = convertMat2Ptr<GMM>(plhs[0]);
    gmm->load(filepath);
    gmm->eigenDecomposition();
    return;
  }

  if (!strcmp("dcs", cmd))
  {
    if (nrhs != 3)
    {
      mexErrMsgTxt("transform: inputs required: <\"dcs\", gmm1.objectHandle, gmm2.objectHandle>");
      return;
    }
    GMM* gmm1 = convertMat2Ptr<GMM>(prhs[1]);
    GMM* gmm2 = convertMat2Ptr<GMM>(prhs[2]);

    // dcs is symmetric so gmm2->dcs(*gmm1) gives same result
    double dcs_score = gmm1->dcs(*gmm2);
    cu::scalarToMATLAB<double>(dcs_score, 0, plhs);
    return;
  }

  if (!strcmp("log_likelihood", cmd))
  {
    if (nrhs != 3)
    {
      mexErrMsgTxt("log_likelihood: inputs required: <\"log_likelihood\", objectHandle, data>");
      return;
    }
    GMM* gmm = convertMat2Ptr<GMM>(prhs[1]);

    Eigen::MatrixXf X;
    cu::matlabToEigen(prhs, 2, X);

    float ll = gmm->logLikelihood(X);
    cu::scalarToMATLAB(ll, 0, plhs);
    return;
  }

  if (!strcmp("weighted_log_probability", cmd))
  {
    if (nrhs != 3)
    {
      mexErrMsgTxt("weighted_log_probability: inputs required: <\"weighted_log_probability\", objectHandle, data>");
      return;
    }
    GMM* gmm = convertMat2Ptr<GMM>(prhs[1]);

    Eigen::MatrixXf X;
    cu::matlabToEigen(prhs, 2, X);

    Eigen::MatrixXf wp = gmm->weightedLogProbability(X);
    cu::eigenToMATLAB(wp, 0, plhs);
    return;
  }

  if (!strcmp("posterior", cmd))
  {
    if (nrhs != 3)
    {
      mexErrMsgTxt("posterior: inputs required: <\"posterior\", objectHandle, data>");
      return;
    }
    GMM* gmm = convertMat2Ptr<GMM>(prhs[1]);

    Eigen::MatrixXf X;
    cu::matlabToEigen(prhs, 2, X);

    Eigen::MatrixXf p = gmm->posterior(X);
    cu::eigenToMATLAB(p, 0, plhs);
    return;
  }

  if (!strcmp("merge", cmd))
  {
    if (nrhs != 3)
    {
      mexErrMsgTxt("merge: inputs required: <\"merge\", gmm1.objectHandle, gmm2.objectHandle>");
      return;
    }
    GMM* gmm1 = convertMat2Ptr<GMM>(prhs[1]);
    GMM* gmm2 = convertMat2Ptr<GMM>(prhs[2]);
    gmm1->merge(*gmm2);

    return;
  }

  if (!strcmp("set_support_size", cmd))
  {
    if (nrhs != 3)
    {
      mexErrMsgTxt("set_support_size: inputs required: <\"set_support_size\", gmm.objectHandle, support_size>");
      return;
    }
    GMM* gmm = convertMat2Ptr<GMM>(prhs[1]);
    uint32_t n = cu::matlabToScalar(prhs, 2);
    gmm->setSupportSize(n);

    return;
  }

  if (!strcmp("make_covs_isoplanar", cmd))
  {
    if (nrhs != 2)
    {
      mexErrMsgTxt("make_covs_isoplanar: inputs required: <\"make_covs_isoplanar\", gmm.objectHandle>");
      return;
    }
    GMM* gmm = convertMat2Ptr<GMM>(prhs[1]);
    gmm->makeCovsIsoplanar();

    return;
  }

  //Delete
  if (!strcmp("delete", cmd))
  {
    if (nlhs != 0)
      {
        mexErrMsgTxt("Delete: outputs required: none");
        return;
        }

    if (nrhs != 2)
      {
        mexErrMsgTxt("Delete: inputs required : <\"delete\", objectHandle>");
        return;
      }

    destroyObject<GMM>(prhs[1]);

    return;
  }

  mexErrMsgTxt("Command not recognized.");
  return;
}
