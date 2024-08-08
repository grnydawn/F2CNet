cmake_minimum_required(VERSION 3.0)
project(miniWeather Fortran)

enable_testing()

include(utils.cmake)


############################################################
## Set Parameters
############################################################
if ("${NX}" STREQUAL "")
  SET(NX 100)
endif()
if ("${NZ}" STREQUAL "")
  SET(NZ 50)
endif()
if ("${SIM_TIME}" STREQUAL "")
  SET(SIM_TIME 1000)
endif()
if ("${OUT_FREQ}" STREQUAL "")
  SET(OUT_FREQ 10)
endif()
if ("${DATA_SPEC}" STREQUAL "")
  SET(DATA_SPEC DATA_SPEC_THERMAL)
endif()
SET(EXE_DEFS "-D_NX=${NX} -D_NZ=${NZ} -D_SIM_TIME=${SIM_TIME} -D_OUT_FREQ=${OUT_FREQ} -D_DATA_SPEC=${DATA_SPEC}")
SET(TEST_DEFS "-D_NX=100 -D_NZ=50 -D_SIM_TIME=400 -D_OUT_FREQ=400 -D_DATA_SPEC=DATA_SPEC_THERMAL")
if ("${FFLAGS}" MATCHES ".*SINGLE_PREC.*")
  set(tol "1.e-5")
  message(STATUS "Using single precision. Test tolerance ${tol}")
else()
  set(tol "1.e-13")
  message(STATUS "Using double precision. Test tolerance ${tol}")
endif()


############################################################
## Append FFLAGS
############################################################
SET(CMAKE_Fortran_FLAGS "${FFLAGS}")


############################################################
## Compile the MPI version
############################################################
add_executable(mpi miniWeather_mpi.F90)
set_target_properties(mpi PROPERTIES COMPILE_FLAGS "${EXE_DEFS}")

if (NOT ("${LDFLAGS}" STREQUAL "") )
  target_link_libraries(mpi      "${LDFLAGS}")
endif()
if (NOT ("${MPI_LINK_FLAGS}" STREQUAL "") )
  target_link_libraries(mpi      "${MPI_LINK_FLAGS}")
endif()

