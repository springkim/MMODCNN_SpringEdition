cmake_minimum_required(VERSION 3.4)
project(MMODCNN_SpringEdition)


find_package(CUDA QUIET REQUIRED)
include(FindCUDA)

set(CMAKE_CXX_STANDARD 11)
list(APPEND CMAKE_C_FLAGS "-std=c11 -O3 -ffast-math -Wno-unused-result")
list(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50)    #for maxwell
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52)    #for maxwell
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_60,code=sm_60)    #for pascal
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_61,code=sm_61)    #for pascal
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_62,code=sm_62)    #for jetson tx2
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_75,code=sm_75)    #for turing
#https://developer.nvidia.com/cuda-gpus

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(SOURCE_FILES
        src/tinyxml2.cpp
)

add_library(cuda INTERFACE)

include_directories(3rdparty/include)
set_target_properties(cuda PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${CUDA_INCLUDE_DIRS}
        INTERFACE_LINK_LIBRARIES "${CUDA_LIBRARIES};${CUDA_curand_LIBRARY};${CUDA_CUFFT_LIBRARIES};${CUDA_CUBLAS_LIBRARIES};${CUDA_CUSOLVER_LIBRARIES}"
        )

if(MSVC)
    add_compile_options("/openmp")
    set(cudnn "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cudnn.lib")
    set(cusolver "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cusolver.lib")
    include_directories(${PROJECT_SOURCE_DIR}/3rdparty/include)
    cuda_add_executable(MMODCNNSE_Train ${SOURCE_FILES} src/MMODCNNSE_train.cpp)
    cuda_add_library(libMMODCNNSE SHARED ${SOURCE_FILES} src/MMODCNNSE_test.cpp)       #Make dll file
    target_link_libraries(MMODCNNSE_Train ${PROJECT_SOURCE_DIR}/3rdparty/staticlib/x64/Release/dlib19.16.0_release_64bit_msvc1900.lib
            cuda ${cudnn} ${cusolver})
    target_link_libraries(libMMODCNNSE
            ${PROJECT_SOURCE_DIR}/3rdparty/staticlib/x64/Release/dlib19.16.0_release_64bit_msvc1900.lib
            ${PROJECT_SOURCE_DIR}/3rdparty/staticlib/x64/Release/opencv_world400.lib
            cuda ${cudnn} ${cusolver})
endif(MSVC)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")