#---------------------------------
#cmake .. \
#-DPYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")  \
#-DPYTHON_LIBRARY=$(python3 -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
#---------------------------------
cmake_minimum_required(VERSION 3.12)
project(cuda_algo_lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

set(CMAKE_CUDA_ARCHITECTURES 52;60;61;70;75;80;86;89;90)

if(NOT DEFINED PYTHON_INCLUDE_DIR)
  message(FATAL_ERROR "Please set -DPYTHON_INCLUDE_DIR=xxx")
endif()

if(NOT DEFINED PYTHON_LIBRARY)
  message(FATAL_ERROR "Please set -DPYTHON_LIBRARY=xxx")
endif()

message(STATUS "Using PYTHON_INCLUDE_DIR: ${PYTHON_INCLUDE_DIR}")
message(STATUS "Using PYTHON_LIBRARY: ${PYTHON_LIBRARY}")

include_directories(${PYTHON_INCLUDE_DIR})

find_library(CUDA_RUNTIME_LIBRARY cudart HINTS /usr/local/cuda/lib64)

link_directories(${PYTHON_LIBRARY})
# pybind11
add_subdirectory(third_party/pybind11)

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/src/util)


file(GLOB_RECURSE CUDA_SRC ${PROJECT_SOURCE_DIR}/src/cuda/*.cu)
file(GLOB_RECURSE CPU_SRC ${PROJECT_SOURCE_DIR}/src/cpu/*.cpp)
file(GLOB_RECURSE BIND_SRC ${PROJECT_SOURCE_DIR}/src/binding/*.cu)


add_library(cuda_kernels STATIC ${CUDA_SRC})
add_library(cpu_kernels STATIC ${CPU_SRC})

pybind11_add_module(binding MODULE ${BIND_SRC})

set_target_properties(binding PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)

target_link_libraries(binding PRIVATE cuda_kernels cpu_kernels ${CUDA_RUNTIME_LIBRARY})

set_target_properties(binding PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
