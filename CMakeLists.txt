cmake_minimum_required(VERSION 3.15)

set(ENV{FC} "mpif90")

project(accP2P LANGUAGES Fortran)

set(CMAKE_Fortran_FLAGS "-cpp -fast -D_USE_NVTX -gpu=lineinfo,ptxinfo -Minfo=all -cuda -acc -lnvToolsExt")

file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_LIST_DIR}/src/*.f90)
add_executable(${PROJECT_NAME} ${SOURCES})