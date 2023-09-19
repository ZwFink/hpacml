configure_file(${LIBAPPROX_INC_DIR}/approx.h approx.h COPYONLY)
configure_file(${LIBAPPROX_INC_DIR}/approx_debug.h approx_debug.h COPYONLY)
message("libapprox is ${LIBAPPROX_INC_DIR}") 

set(CLANG_SOURCE_INCLUDE "${LLVM_EXTERNAL_CLANG_SOURCE_DIR}/include/")
message("libapprox is ${CLANG_SOURCE_INCLUDE}") 
message("approx include is ${LIBAPPROX_INC_DIR}")

set(LIBAPPROX_SOURCE_FILES
    event.cpp
    approx_runtime.cpp
    approx_data_util.cpp
    approx_memoize_out.cpp
    approx_memoize_iact_in.cpp
    approx_perforate.cpp
    approx_petrubate.cpp
    approx_surrogate_rti.cpp
    memory_pool/memory_pool.cpp
    database/database.cpp
)

include_directories(
    ${CMAKE_CURRENT_BINARY_DIR}
    ${LIBAPPROX_INC_DIR}
    ${LIBAPPROX_BASE_DIR}
    ${CLANG_SOURCE_INCLUDE}
)

set(JSON_BuildTests OFF CACHE INTERNAL "")

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
#We might need to use CXX as linker.
set(LIBAPPROX_LINKER_LANGUAGE C)
add_library(approx ${LIBAPPROX_LIBRARY_KIND} ${LIBAPPROX_SOURCE_FILES})
find_package(Torch REQUIRED)
set(CMAKE_CUDA_FLAGS "")
set(CMAKE_CUDA_ARCHITECTURES ON)
# link with openmp
find_package(OpenMP REQUIRED)
target_link_libraries(approx PRIVATE OpenMP::OpenMP_CXX)

# set to RelWithDebInfo
set(CMAKE_BUILD_TYPE RelWithDebInfo)

find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(HDF5 NAMES hdf5 COMPONENTS C shared REQUIRED NO_DEFAULT_PATH PATHS ${HDF5_Dir} ${HDF5_Dir}/share/cmake)

set_target_properties(approx PROPERTIES
    PREFIX "" SUFFIX "" OUTPUT_NAME "${LIBAPPROX_LIB_FILE}"
    LINK_FLAGS ""
    LINKER_LANGUAGE ${LIBAPPROX_LINKER_LANGUAGE})
set_target_properties(approx PROPERTIES
    PREFIX "" SUFFIX "" OUTPUT_NAME "${LIBAPPROX_LIB_FILE}"
    LINK_FLAGS ""
    LINKER_LANGUAGE ${LIBAPPROX_LINKER_LANGUAGE})

include_directories(${HDF5_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})

target_link_libraries(approx PRIVATE ${CUDA_LIBRARIES} cuda)
target_link_libraries(approx PRIVATE ${CMAKE_DL_LIBS})
target_link_libraries(approx PRIVATE ${HDF5_LIBRARIES})
target_link_libraries(approx PRIVATE ${TORCH_LIBRARIES})
message(STATUS "Torch libraries dir: ${TORCH_LIBRARIES}")

target_link_libraries(approx PRIVATE nlohmann_json::nlohmann_json)

message("PACKAGE VERSION is: ${PACKAGE_VERSION}")
set(LIBAPPROX_HEADERS_INSTALL_PATH "${APPROX_INSTALL_LIBDIR}/clang/${PACKAGE_VERSION}/include")

install(TARGETS approx ${LIBAPPROX_INSTALL_KIND} DESTINATION "${APPROX_INSTALL_LIBDIR}")
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/approx.h DESTINATION ${LIBAPPROX_HEADERS_INSTALL_PATH})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/approx_debug.h DESTINATION ${LIBAPPROX_HEADERS_INSTALL_PATH})
