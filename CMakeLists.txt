cmake_minimum_required(VERSION 3.13)
project(Optimization)
set (CMAKE_EXPORT_COMPILE_COMMANDS True) 

# Configure Eigen3 with Apple Accelerate BLAS on macOS
if(APPLE)
    # Find Accelerate framework
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
endif()

find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-g)
add_compile_options(-Wall -Wextra -pedantic )
if(APPLE)
    add_compile_options(-O3 -march=native -ffast-math -Wno-deprecated-declarations)
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


# Link Accelerate framework if available
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
endif()

add_subdirectory(bal)
add_subdirectory(Graph)
add_subdirectory(RGD)
add_subdirectory(solver)
add_subdirectory(first_order_methods)
add_subdirectory(programing)
