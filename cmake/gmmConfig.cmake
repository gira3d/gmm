# - Try to find gmm header files and libraries
#
# Once done this will define
#
# GMM

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)
set(gmm_INCLUDE_DIR "${PACKAGE_PREFIX_DIR}/gmm/include")
