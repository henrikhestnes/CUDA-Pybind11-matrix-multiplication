cmake_minimum_required(VERSION 3.3)
project(multiply)
find_package(CUDA REQUIRED)
find_package(PythonInterp 3.6 REQUIRED)
find_package(PythonLibs 3.6 REQUIRED)

include_directories(
    ${PYTHON_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ./pybind11/include
    )

link_directories(
    /usr/local/cuda/lib64
    )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

cuda_add_library(gpu_library SHARED
    multiply_matrix.cu)

target_link_libraries(gpu_library
    ${PYTHON_LIBRARIES}
    cudart)

set_target_properties(gpu_library PROPERTIES PREFIX "")

#cuda_add_executable(multiply multiply_matrix.cu)