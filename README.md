# GMM

This package provides provides functions for training GMMs on pointclouds of arbitrary dimensions.

## To run Gtests
Gtests are listed in the CMakeLists.txt file.  In order to run them, the tests must first be compiled using the following command:

  `catkin build --make-args tests -- gmm -DCMAKE_BUILD_TYPE=Release`

The tests may be run using the following commands:

```rosrun gmm test_k_init
   rosrun gmm test_kmeans
   rosrun gmm test_k_means_elkan
   rosrun gmm test_kmeans_time
   rosrun gmm test_sample
   rosrun gmm test_em
   rosrun gmm test_get_submap```

`test_gmm` is a ros node so it spins until it receives SIGINT.
