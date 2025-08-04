cmake_minimum_required(VERSION 3.18)
project(CudaFinanceTool VERSION 1.0.0 LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA settings
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)

# Set CUDA architecture (adjust based on your GPU)
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler -Wall,-Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Source files
set(CUDA_SOURCES
    main.cu
    option_pricing.cu
    monte_carlo.cu
    risk_analysis.cu
    utils.cu
)

# Create main executable
cuda_add_executable(cuda_finance_tool ${CUDA_SOURCES})

# Set output directory
set_target_properties(cuda_finance_tool PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Test executable (if test files exist)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_suite.cu")
    cuda_add_executable(test_suite test_suite.cu)
    set_target_properties(test_suite PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endif()

# Python interface (if pycuda is available)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
if(Python3_FOUND)
    add_subdirectory(python_interface)
endif()

# Installation
install(TARGETS cuda_finance_tool
    RUNTIME DESTINATION bin
)

# Package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/CudaFinanceToolConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

# Export targets
install(EXPORT CudaFinanceToolTargets
    FILE CudaFinanceToolTargets.cmake
    NAMESPACE CudaFinanceTool::
    DESTINATION lib/cmake/CudaFinanceTool
)

# Configure package config file
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CudaFinanceToolConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/CudaFinanceToolConfig.cmake"
    INSTALL_DESTINATION lib/cmake/CudaFinanceTool
)

# Install config files
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/CudaFinanceToolConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/CudaFinanceToolConfigVersion.cmake"
    DESTINATION lib/cmake/CudaFinanceTool
)

# Custom targets
add_custom_target(check-cuda
    COMMAND ${CMAKE_COMMAND} -E echo "Checking CUDA installation..."
    COMMAND nvcc --version
    COMMENT "Checking CUDA installation"
)

add_custom_target(check-gpu
    COMMAND ${CMAKE_COMMAND} -E echo "Checking GPU compatibility..."
    COMMAND nvidia-smi
    COMMENT "Checking GPU compatibility"
)

add_custom_target(setup
    DEPENDS check-cuda check-gpu
    COMMAND ${CMAKE_COMMAND} -E echo "Setup complete. Ready to build."
)

# Documentation target
add_custom_target(docs
    COMMAND ${CMAKE_COMMAND} -E echo "Generating documentation..."
    COMMENT "Generate documentation"
)

# Clean target
add_custom_target(clean-all
    COMMAND ${CMAKE_BUILD_TOOL} clean
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/bin
    COMMENT "Clean all build artifacts"
) 