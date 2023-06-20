^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package gmm
^^^^^^^^^^^^^^^^^^^^^^^^^

0.0.9 (2023-06-19)
------------------
* Removes lfs objects
* Removes unit tests
* Add BSD-3 license
* Remove fast GMM implementations
* Remove files used to speed up EM
* Contributors: Wennie Tabib

0.0.8 (2023-06-19)
------------------
* Update the GMM plotting script so that it doesn't take an alpha
* Minor updates to visualization interface
* Update loading function to conform to comments, clean up plotting code
* Remove ros and catkin
* Contributors: Wennie Tabib

0.0.7 (2023-06-19)
------------------
* remove mex interface matlab even if MATLAB is found
* add a matlab boolean in the cmakelists
* remove mex_interface_matlab as dependence
* adds function to remove components from GMM and output a submap
* pri objective is the first term of dcs
* Contributors: Kshitij Goel, Wennie Tabib

0.0.6 (2023-06-19)
------------------
* Debugging strange failure case for dynamic kmeans unit test
* Remove unused src directory and GMM.cc file
* Fix compilation issue in getSubmap function wrt GMMNBase <-> GMM3Base
* labels variable cast to int to fix issue wrt eigen 3.4.0
* gmm2
* Add data to reproduce regression test cases
* Clean up regression file so that it uses the plane
* Removes matlab files that have been absorbed by MATLAB GMM classes
* Incorporate matlab functions directly into GMM class files
* Adds matlab test scripts and gmm4 mex
* Adds regression for 4D GMM and MATLAB interface
* Change typedefs to using so that pointers don't get cast to wrong type
* Adds inline to GMM3 functions transform and makeCovsIsoplanar()
* Adds matlab interface for GMM4
* Removes explicit line to call precs. Use setPrecs() instead
* Adds getters and setters for cholesky precision matrices
* Computes precisions if covariances are set and precs\_ are empty
* Adds comments to posterior, loglikelihood functions
* Adds warning so that users know that precs\_ is not the precision matrix
* Adds posterior function
* Casts to double to retain sufficient resolution for weighted log probability calculation
* Adds functions for loglikelihood and weightedLogProbability
* Allow scoreSamples function to return -Inf due to lack of precision. Add comment to this effect.
* Removes publisher_utils from cmakelists and package.xml since unit test no longer requires it
* gitignore ccls and compile commands
* GMM4.h
* Contributors: Kshitij Goel, Wennie Tabib

0.0.5 (2023-06-19)
------------------
* adds comment for the fastSample matlab function
* Add boost filesystem and remove catkin libraries in cmakelists.txt
* Check that covariances are PSD and correct if they are not
* Adds function handles to the MATLAB GMM class
* Adds function to create a mex object from matlab class data
* Adds function handle to matlab scripts to make isoplanar covariances
* Adds function to make covariances isoplanar
* Removes print message for MATLAB include path
* Contributors: Kshitij Goel, Wennie Tabib

0.0.4 (2023-06-19)
------------------
* Adds very fast point sampler by providing randomly generated numbers as input
* Adds setter for precisions using the covariance matrices
* Contributors: Wennie Tabib

0.0.3 (2023-06-19)
------------------
* Removes constexpr to enable compilation with -frounding-math flag
* Removes print statements
* Removes typedef std::shared_ptr in GMM3.h class to prevent error about no viable conversion between GMMNBase and GMM3Base.
  Adds unit test to prevent this from happening again.
* Contributors: Wennie Tabib

0.0.2 (2023-06-19)
------------------
* correct interface for SigmaPointEM so kmeans isn't run separately
* modify interface to take input for mahalanobis distance
* add dcs function, load_from_file, and set_support_size functions for FSR paper
* sample from MATLAB properly
* matlab file for wrapper around SigmaPointEM
* new EM based on std::vector instead of sparse matrix
* vary the EM type
* mex for fast GMM calculation
* working mahalanobis distance version
* remove extraneous resp_T\_ reference
* sparse matrix throughout code
* comment for reg covar
* estimate_gaussian_parameters updated with sparse matrix
* sparse matrix through EM
* sparse matrix in estimate_log_prob_resp
* huge speedup when templating number of clusters
* sparse matrix replaced in estimate_weighted_log_prob
* added matrix for sparse responsibility
* copy EM into new class SigmaPointEM.h
* add deepCopy function
* don't check for existence of save filepath
* member variables updated storage necessitates updates to matlab functions
* add transform to mex interface
* change linear call to rotation
* add Ptr and ConstPtr to GMM3Base
* Ptr and ConstPtr changed from boost to std
* save gmms to file via matlab
* load gmms from file properly
* remove eigen_catkin as dependency
* C++ sampling function added to matlab bindings
* increase number of allowed elements to 12e7
* fixed inf return value for aic and bic
* bic
* added precision matrices to compute AIC
* Contributors: Wennie Tabib

0.0.1 (2019-01-13)
------------------
* remove test_gmm unit test
* add roslib dependency and enable gtests
* fast sample method
* replace system-level eigen with one compiled in dry
* clean up comments
* eigenvalue copy corrected to N instead of C
* compile mex file with -Wall -Wextra -Werror -Wno-int-in-bool-context -Wno-unused-parameter, but gtests do not compile with these flags
* correct gtest entry in README.md
* force derived class to initialize partition(), getCenterID(), and getRandomVector() methods
* removed unused variables in gmm mex getParameters()
* removing extraneous files and libraries from CMakeLists.txt
* migrate to package.xml version 2.0
* initialize base class virtual functions
* fstream for compilation on linux
* single color for gmm matlab visualization
* updated interface of sample function to return which component each point is
  sampled from
* added test_sample test, updated README
* mex file works now and renamed to GMM3
* class for N-base GMMs where N is the dimension of the data
* swap number of samples with number of clusters in templated parameters
* temporarily remove multivariate_normal_sample
* pass by reference for kInit and KMeansElkan
* pass by reference in sklearn
* Dynamic parameters with EM working
* update template parameters for SKLearn EM to remove n_local_trials
* updated kmeans timing test to remove parameter for n_local_trials
* move common functions into test/include/Common.h
* additional kmeans test with dynamically allocated parameters
* uncomment kmeans gtest
* cleaned up kmeans and replaced k_init and k_means_elkan functions with classes
* extra test with Eigen::Dynamic templated parameters
* kmeans_elkan separated into its own class
* Eigen::Dynamic changed back to N<K>() so comment is now removed.
* change < 0 to <= 0 in KInit
* throw error if n_clusters is less than 0
* constant expression function to determine n_local_trials, remove template
  parameter for n_local_trials
* move random number generator to bottom of file and forward declare
* all typedef matrices declared inside test functions
* refactored k_init function into its own class with mock'ed random number
  generator for unit testing
* convert kmeans template values to int and set default -1 values
* replaced templated values with variables
* check that time taken to execute EM gtest is always less than 600 ms
* README.md includes information about how to run gtests
* working small example for templating kmeans
* updated kmeans class with constexpr for different numbers of components
* mex wrappers to call GMM
* application of transform to GMMs corrected
* methods to sample from distribution
* remove printouts
* remove template from GMM file and don't extend SKLearn from GMM
* compiler flags for macos
* additional flags to enable faster runtime
* compiler flags to run faster on linux
* remove extraneous executables and unit tests
* remoe debug mode
* fixed problems failing in debug mode
* failure on scan 450 when running test_gmm.cc
* remove unused unit tests
* add interface to GMM with changeable EM types
* replaced uint32_t template parameter for number of samples with int to enable dynamic allocation
* fixed broken unit test in kmeans
* remove labels as member variable and inline all functions
* changed number of components back to 29
* regression test
* much faster operation
* working copy but needs optimization
* resp -> resp\_ in estimate_gaussian_parameters
* fixed bug in compute_precisions and put together the entire pipeline
* m step working
* working e step
* working estimate_log_prob_resp
* estimate_weighted_log_prob complete and tested
* completed estimate_log_gaussian_prob function. works with doubles.
* half of the estimate_log_gaussian_prob function developed and tested.  Overflow occurs when using floats
* Class for EM and compute_log_det_cholesky tested
* very fast runs on macos but many more interations of elkan on linux
* relative paths to unit test data files for portability
* reset to unit testing mode
* static variables for extractCandidates and searchSorted functions
* static dimensions in fit and elkan functions
* static matrix sizes in centers_dense function
* static initialization for upper_bounds variable
* added S parameter for number of samples (or points)
* sped up searchSorted
* significant speed increase by creating fast euclidean distance method
* rename header directive
* function to write matrix to file
* correct matrix sizes on variance and tolerance functions
* parameterize unit tests by whether or not comparison to correct answers should occur
* remove print statement in fit()
* cleaned up print outs for faster speed
* fit() passes unit tests
* partition function return value changed from (incorrect) VectorkX to (correct) Matrixkk
* added new typedef for VectorfX
* updated types of bounds_tight, center_shift, and slb
* created Matrixkk for KxK matrices and updated the type of center_half_distances
* changed upper_bounds to vector
* converted labels variable from MatrixX to VectorX
* explicitly sized matrix for Kxunknown
* k_init partially templated
* compilation flags to speed up code significantly
* remove extraneous typdefs
* noalias() to euclidean_distance function
* correct dimensions of YY in euclidean_distances
* unit testing data
* cleaned up unit tests
* clean up printouts
* fit produces correct result from unit tests
* added checks for lower_bound and upper bound
* fixed bug in k_init call
* finished rewrite of kmeans elkan
* centers_dense completed and tested
* working update_labels_distances_inplace
* added candidate IDs to check against. there's an off by one error due to rounding causing differences in the outputs
* kmeans working for small datasets but not large ones
* initial commit
* Contributors: Wennie Tabib
