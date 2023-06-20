# - this module looks for Matlab
# Defines:
#  MATLAB_INCLUDE_DIR: include path for mex.h
#  MATLAB_LIBRARIES:   required libraries: libmex, libmx
#  MATLAB_MEX_LIBRARY: path to libmex
#  MATLAB_MX_LIBRARY:  path to libmx

# Check if environment variable MATLAB_ROOT is set
IF( "$ENV{MATLAB_ROOT}" STREQUAL "" )
  # if it's not set, then we look in the standard locations
  if (APPLE)
    file(GLOB matlab_bin_directories "/Applications/MATLAB*/bin")
    set(mex_program_name "mex")
  else()
    file(GLOB matlab_bin_directories "/usr/local/MATLAB/*/bin")
    set(mex_program_name "mex")
  endif()

  #Reverse list so the highest version (sorted alphabetically) is preferred
  list(REVERSE matlab_bin_directories)
  find_program(MEX_COMMAND ${mex_program_name}
    PATHS ${matlab_bin_directories} ENV PATH
    NO_DEFAULT_PATH)
  mark_as_advanced(FORCE MEX_COMMAND)

  get_filename_component(MEX_COMMAND "${MEX_COMMAND}" REALPATH)
  get_filename_component(mex_path "${MEX_COMMAND}" PATH)
  get_filename_component(MATLAB_ROOT2 "${mex_path}/.." ABSOLUTE)
  set(ENV{MATLAB_ROOT} "${MATLAB_ROOT2}" CACHE PATH "Path to MATLAB installation root (e.g. /usr/local/MATLAB/R2012a)")
endif()

SET(MATLAB_FOUND 0)
if ( "$ENV{MATLAB_ROOT}" STREQUAL "" )
  MESSAGE(STATUS "MATLAB_ROOT environment variable not set." )
  MESSAGE(STATUS "In Linux this can be done in your user .bashrc file by appending the corresponding line, e.g:" )
  MESSAGE(STATUS "export MATLAB_ROOT=/usr/local/MATLAB/R2012b" )
  MESSAGE(STATUS "In Windows this can be done by adding system variable, e.g:" )
  MESSAGE(STATUS "MATLAB_ROOT=D:\\Program Files\\MATLAB\\R2011a" )
else()
  set(MATLAB_INCLUDE_DIR $ENV{MATLAB_ROOT}/extern/include)
  INCLUDE_DIRECTORIES(${MATLAB_INCLUDE_DIR})

  FIND_LIBRARY( MATLAB_MEX_LIBRARY
    NAMES libmex mex
    PATHS $ENV{MATLAB_ROOT}/bin
    PATH_SUFFIXES maci64 glnxa64 glnx86 win64/microsoft win32/microsoft)

  FIND_LIBRARY( MATLAB_MX_LIBRARY
    NAMES libmx mx
    PATHS $ENV{MATLAB_ROOT}/bin
    PATH_SUFFIXES maci64 glnxa64 glnx86 win64/microsoft win32/microsoft)

  if (APPLE)
    set(mxLibPath "$ENV{MATLAB_ROOT}/bin/maci64")
    set(MATLAB_MX_LIBRARY "${mxLibPath}/libmx.dylib")
  endif()

  FIND_LIBRARY( MATLAB_MAT_LIBRARY
    NAMES libmat mat
    PATHS $ENV{MATLAB_ROOT}/bin
    PATH_SUFFIXES maci64 glnxa64 glnx86 win64/microsoft win32/microsoft)

ENDIF()

# This is common to UNIX and Win32:
SET(MATLAB_LIBRARIES
  ${MATLAB_MEX_LIBRARY}
  ${MATLAB_MX_LIBRARY}
  ${MATLAB_MAT_LIBRARY}
  )

IF(MATLAB_INCLUDE_DIR AND MATLAB_LIBRARIES)
  SET(MATLAB_FOUND 1)
  MESSAGE(STATUS "Matlab libraries will be used")
ENDIF(MATLAB_INCLUDE_DIR AND MATLAB_LIBRARIES)

MARK_AS_ADVANCED(
  MATLAB_LIBRARIES
  MATLAB_MEX_LIBRARY
  MATLAB_MX_LIBRARY
  MATLAB_INCLUDE_DIR
  MATLAB_FOUND
  MATLAB_ROOT
  )
