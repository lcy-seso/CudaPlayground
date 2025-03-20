cmake_minimum_required(VERSION 3.25.0)
project(test_gemv LANGUAGES CXX CUDA)

set(TARGET "gemv_test")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/../cmake")

include(generic)

set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS} --extended-lambda  --keep --source-in-ptx")

set(CUDA_NVCC_FLAGS
    "${CUDA_NVCC_FLAGS} --extended-lambda  --keep --source-in-ptx")
message(STATUS "NVCC FLAGS = ${CUDA_NVCC_FLAGS}")

add_executable(${TARGET} main.cu)
