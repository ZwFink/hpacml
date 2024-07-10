cmake_minimum_required(VERSION 3.14)
project(my_project)

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

set(HPAC_LIBRARY_LOCATION $ENV{HPAC_LIBRARY_LOCATION})

# add link directory
link_directories(
   ${HDF_LIBRARY_LOCATION}
    ${HPAC_LIBRARY_LOCATION}
    )

add_compile_options(-fapprox)

add_executable(
  hpacml_test
  tests/test_input1d_ovlp.cpp
  tests/test_input1d.cpp
  tests/test_input2d.cpp
  tests/test_broadcast.cpp
  tests/input_1d_input_more_than1.cpp
)
target_link_libraries(
  hpacml_test
  approx
  GTest::gtest_main
  hdf5
)

include(GoogleTest)
gtest_add_tests(TARGET hpacml_test)
