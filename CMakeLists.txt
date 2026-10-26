cmake_minimum_required(VERSION 3.16)
project(SuperScienceInference LANGUAGES CXX)

add_library(super_science INTERFACE)
target_include_directories(super_science INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(super_science INTERFACE cxx_std_17)

# Use available CPU cores for large inference matrix-vector operations.
# Optional and portable: builds without OpenMP retain the serial kernel.
option(SUPER_SCIENCE_USE_OPENMP "Enable OpenMP CPU inference parallelism" ON)
if(SUPER_SCIENCE_USE_OPENMP)
  find_package(OpenMP QUIET COMPONENTS CXX)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(super_science INTERFACE OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP CPU inference: enabled")
  else()
    message(STATUS "OpenMP unavailable; using single-threaded CPU inference")
  endif()
endif()

add_executable(qwen3_chat apps/qwen3_chat.cpp)
target_link_libraries(qwen3_chat PRIVATE super_science)

add_executable(qwen3_cli apps/qwen3_cli.cpp)
target_link_libraries(qwen3_cli PRIVATE super_science)

add_executable(tokenizer_cli apps/tokenizer_cli.cpp)
target_link_libraries(tokenizer_cli PRIVATE super_science)

add_executable(inference_tests tests/main.cpp)
target_link_libraries(inference_tests PRIVATE super_science)
add_executable(safetensors_tests tests/safetensors_test.cpp)
target_link_libraries(safetensors_tests PRIVATE super_science)
add_executable(validation_tests tests/validation_test.cpp)
target_link_libraries(validation_tests PRIVATE super_science)
add_executable(qwen3_cpu_tests tests/qwen3_cpu_test.cpp)
target_link_libraries(qwen3_cpu_tests PRIVATE super_science)
add_executable(qwen3_file_tests tests/qwen3_file_test.cpp)
target_link_libraries(qwen3_file_tests PRIVATE super_science)
# CMake Release configurations normally define NDEBUG. The tests below use
# assert(), so explicitly keep assertions enabled on each supported compiler.
# This applies to test executables only, never production programs.
function(super_science_enable_test_assertions target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /UNDEBUG)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE -UNDEBUG)
  endif()
endfunction()

super_science_enable_test_assertions(inference_tests)
super_science_enable_test_assertions(qwen3_cpu_tests)

enable_testing()
add_test(NAME inference_tests COMMAND inference_tests)
add_test(NAME safetensors_tests COMMAND safetensors_tests)
add_test(NAME validation_tests COMMAND validation_tests)
add_test(NAME qwen3_cpu_tests COMMAND qwen3_cpu_tests)
add_test(NAME qwen3_file_tests COMMAND qwen3_file_tests)

add_executable(tokenizer_tests tests/tokenizer_test.cpp)
target_link_libraries(tokenizer_tests PRIVATE super_science)
super_science_enable_test_assertions(tokenizer_tests)
add_test(NAME tokenizer_tests COMMAND tokenizer_tests)

# Optional native desktop application. The inference library remains Qt-free.
option(SUPER_SCIENCE_BUILD_DESKTOP "Build Qt desktop chat application" OFF)
if(SUPER_SCIENCE_BUILD_DESKTOP)
  find_package(Qt6 REQUIRED COMPONENTS Widgets)
  add_executable(super_science_desktop apps/desktop/main.cpp)
  set_target_properties(super_science_desktop PROPERTIES AUTOMOC ON)
  target_link_libraries(super_science_desktop PRIVATE super_science Qt6::Widgets)
endif()

# Educational microbenchmarks are explicit, not part of the default test suite.
option(SUPER_SCIENCE_BUILD_BENCHMARKS "Build optional numerical benchmarks" OFF)
if(SUPER_SCIENCE_BUILD_BENCHMARKS)
  add_executable(matmul_bench benchmarks/matmul_bench.cpp)
  target_link_libraries(matmul_bench PRIVATE super_science)
endif()
