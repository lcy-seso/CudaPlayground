cmake_minimum_required(VERSION 3.25.0)
project(test_vectorization LANGUAGES CXX CUDA)

set(TARGET "vec_test")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/../cmake")

include(generic)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

set(CUDA_NVCC_FLAGS
    "${CUDA_NVCC_FLAGS} --extended-lambda  --keep --source-in-ptx")
message(STATUS "NVCC FLAGS = ${CUDA_NVCC_FLAGS}")

add_executable(${TARGET} main.cu)
