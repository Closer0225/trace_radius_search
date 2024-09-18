cmake_minimum_required(VERSION 3.22)

project(OptixNormals)

# 设置编译选项
set(CMAKE_CXX_STANDARD 11)

set(OptiX_INSTALL_DIR $ENV{OPTIX_PATH})

set(CUDA_INSTALL_DIR $ENV{CUDA_PATH})

find_package(CUDA REQUIRED)

# 包含头文件目录
include_directories(${CMAKE_SOURCE_DIR}/include)

include_directories(${OptiX_INSTALL_DIR}/include)

include_directories(${CUDA_INSTALL_DIR}/include)

add_compile_definitions(NDEBUG)

add_compile_definitions(_CONSOLE)

add_compile_definitions(_CRT_SECURE_NO_WARNINGS)

add_compile_definitions(NOMINMAX)

# 查找当前目录下的所有源文件
file(GLOB SRC_LIST "src/*.cpp")

# 添加可执行文件
add_executable(OptixNormals ${SRC_LIST})

# 链接必要的库
target_link_libraries(OptixNormals ${CUDA_LIBRARIES})


