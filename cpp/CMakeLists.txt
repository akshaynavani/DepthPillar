cmake_minimum_required(VERSION 3.18)
project(depthpillar_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

option(DEPTHPILLAR_BUILD_TESTS "Build C++ unit tests" ON)

# LibTorch: pass -DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')
# or point it at an extracted libtorch distribution.
find_package(Torch REQUIRED)

include(FetchContent)
FetchContent_Declare(
  json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

add_library(depthpillar
  src/config.cpp
  src/point_cloud.cpp
  src/voxelizer.cpp
  src/anchors.cpp
  src/geometry.cpp
  src/postprocess.cpp
  src/detector.cpp
)
target_include_directories(depthpillar PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(depthpillar PUBLIC ${TORCH_LIBRARIES} nlohmann_json::nlohmann_json)
target_compile_options(depthpillar PRIVATE -Wall -Wextra)

add_executable(depthpillar_infer apps/infer.cpp)
target_link_libraries(depthpillar_infer PRIVATE depthpillar)

add_executable(depthpillar_bench apps/benchmark.cpp)
target_link_libraries(depthpillar_bench PRIVATE depthpillar)

if(DEPTHPILLAR_BUILD_TESTS)
  enable_testing()
  add_executable(depthpillar_tests tests/test_main.cpp)
  target_link_libraries(depthpillar_tests PRIVATE depthpillar)
  add_test(NAME depthpillar_tests COMMAND depthpillar_tests)
endif()
