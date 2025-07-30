cmake_minimum_required(VERSION 3.13)
project(Optimization)
set (CMAKE_EXPORT_COMPILE_COMMANDS True) 

# Configure Eigen3 with BLAS support
if(APPLE)
    # Configure Eigen3 with Apple Accelerate BLAS on macOS
    find_library(ACCELERATE_LIBRARY Accelerate)
    if(ACCELERATE_LIBRARY)
        message(STATUS "Found Apple Accelerate framework: ${ACCELERATE_LIBRARY}")
        # Set Eigen3 to use Accelerate BLAS
        set(EIGEN_USE_BLAS ON)
        # Add compiler definitions for Eigen3 to use Accelerate
        add_compile_definitions(EIGEN_USE_BLAS=1)
        # Add Accelerate include directories
        include_directories(SYSTEM /System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Headers)
    else()
        message(WARNING "Apple Accelerate framework not found, using default Eigen3 configuration")
    endif()
elseif(UNIX AND NOT APPLE)
    # Configure BLAS for Linux systems
    # Try to find BLAS libraries in order of preference: OpenBLAS, Intel MKL, ATLAS, generic BLAS
    
    # First try OpenBLAS
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(OPENBLAS QUIET openblas)
    endif()
    
    if(OPENBLAS_FOUND)
        message(STATUS "Found OpenBLAS: ${OPENBLAS_LIBRARIES}")
        set(BLAS_LIBRARIES ${OPENBLAS_LIBRARIES})
        set(BLAS_INCLUDE_DIRS ${OPENBLAS_INCLUDE_DIRS})
        set(BLAS_LINK_FLAGS ${OPENBLAS_LDFLAGS})
        set(EIGEN_USE_BLAS ON)
        add_compile_definitions(EIGEN_USE_BLAS=1)
        message(STATUS "Using OpenBLAS for Eigen3")
    else()
        # Try Intel MKL
        find_package(MKL QUIET)
        if(MKL_FOUND)
            message(STATUS "Found Intel MKL: ${MKL_LIBRARIES}")
            set(BLAS_LIBRARIES ${MKL_LIBRARIES})
            set(BLAS_INCLUDE_DIRS ${MKL_INCLUDE_DIRS})
            set(EIGEN_USE_BLAS ON)
            add_compile_definitions(EIGEN_USE_BLAS=1)
            message(STATUS "Using Intel MKL for Eigen3")
        else()
            # Try ATLAS
            find_library(ATLAS_LIBRARY atlas)
            find_library(CBLAS_LIBRARY cblas)
            if(ATLAS_LIBRARY AND CBLAS_LIBRARY)
                message(STATUS "Found ATLAS: ${ATLAS_LIBRARY}, ${CBLAS_LIBRARY}")
                set(BLAS_LIBRARIES ${ATLAS_LIBRARY} ${CBLAS_LIBRARY})
                set(EIGEN_USE_BLAS ON)
                add_compile_definitions(EIGEN_USE_BLAS=1)
                message(STATUS "Using ATLAS for Eigen3")
            else()
                # Try generic BLAS
                find_library(BLAS_LIBRARY blas)
                find_library(LAPACK_LIBRARY lapack)
                if(BLAS_LIBRARY AND LAPACK_LIBRARY)
                    message(STATUS "Found generic BLAS/LAPACK: ${BLAS_LIBRARY}, ${LAPACK_LIBRARY}")
                    set(BLAS_LIBRARIES ${BLAS_LIBRARY} ${LAPACK_LIBRARY})
                    set(EIGEN_USE_BLAS ON)
                    add_compile_definitions(EIGEN_USE_BLAS=1)
                    message(STATUS "Using generic BLAS/LAPACK for Eigen3")
                else()
                    message(WARNING "No BLAS library found. Eigen3 will use its own implementations.")
                    message(WARNING "For better performance, install OpenBLAS: sudo apt-get install libopenblas-dev")
                endif()
            endif()
        endif()
    endif()
endif()

find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-g)
add_compile_options(-Wall -Wextra -pedantic )
if(APPLE)
    add_compile_options(-O3 -march=native -ffast-math -Wno-deprecated-declarations)
elseif(UNIX AND NOT APPLE)
    add_compile_options(-O3 -march=native -ffast-math)
endif()


#add_executable(LPSolver lp_solver.cpp)
#target_link_libraries(LPSolver PUBLIC mps_io ${FLEX_LIBRARIES} LinearPrograming)

#add_executable(${PROJECT_NAME}_test optimization_test.cpp reduce.cc)
#target_link_libraries(${PROJECT_NAME}_test PUBLIC
#GTest::GTest
#Eigen3::Eigen
#glog::glog
#)


#add_executable(Linear_Solver_test linear_solver_test.cpp)

#target_include_directories(Linear_Solver_test PUBLIC
#                           "${PROJECT_BINARY_DIR}"
#                           )

#target_link_libraries(Linear_Solver_test PUBLIC
#GTest::GTest
#Eigen3::Eigen
#)

#add_executable(Auto_Diff_Test auto_diff_test.cc)
#target_link_libraries(Auto_Diff_Test PUBLIC
#GTest::GTest
#Eigen3::Eigen
#)


#add_executable(Sparse_Matrix_Test sparse_matrix_test.cpp )
#target_link_libraries(Sparse_Matrix_Test GTest::GTest mmio
#Eigen3::Eigen)
#target_compile_definitions(Sparse_Matrix_Test
#    PRIVATE -DTest_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

#add_executable(IterativeMethodTest itertive_method_test.cpp)
#target_link_libraries(IterativeMethodTest GTest::GTest
#Eigen3::Eigen)


# Link BLAS libraries if available
if(APPLE AND ACCELERATE_LIBRARY)
    # Create a global target for Accelerate that can be linked by subprojects
    add_library(Accelerate::Accelerate INTERFACE IMPORTED)
    
    # Prepare link libraries list
    set(ACCELERATE_LINK_LIBRARIES ${ACCELERATE_LIBRARY})
    
    set_target_properties(Accelerate::Accelerate PROPERTIES
        INTERFACE_LINK_LIBRARIES "${ACCELERATE_LINK_LIBRARIES}"
        INTERFACE_COMPILE_DEFINITIONS "EIGEN_USE_BLAS=1"
    )
    message(STATUS "Accelerate framework configured for use with Eigen3")
elseif(UNIX AND NOT APPLE AND BLAS_LIBRARIES)
    # Create a global target for BLAS that can be linked by subprojects
    add_library(BLAS::BLAS INTERFACE IMPORTED)
    
    set_target_properties(BLAS::BLAS PROPERTIES
        INTERFACE_LINK_LIBRARIES "${BLAS_LIBRARIES}"
        INTERFACE_COMPILE_DEFINITIONS "EIGEN_USE_BLAS=1"
    )
    
    if(BLAS_INCLUDE_DIRS)
        set_target_properties(BLAS::BLAS PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${BLAS_INCLUDE_DIRS}"
        )
    endif()
    
    if(BLAS_LINK_FLAGS)
        set_target_properties(BLAS::BLAS PROPERTIES
            INTERFACE_LINK_FLAGS "${BLAS_LINK_FLAGS}"
        )
    endif()
    
    message(STATUS "BLAS libraries configured for use with Eigen3")
endif()

add_subdirectory(bal)
add_subdirectory(Graph)
add_subdirectory(RGD)
add_subdirectory(solver)
add_subdirectory(first_order_methods)
add_subdirectory(programing)
