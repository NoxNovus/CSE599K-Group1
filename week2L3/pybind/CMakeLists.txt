cmake_minimum_required(VERSION 3.26.1)
project(Nanoflow-python LANGUAGES CXX CUDA)

# Language standards and compiler settings
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 90)

if(CMAKE_CUDA_ARCHITECTURES STREQUAL "90")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -DNDEBUG --generate-code=arch=compute_90a,code=[compute_90a,sm_90a] -Xcompiler=-fPIE -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

# Find required packages
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_program(PYBIND11_STUBGEN_EXECUTABLE NAMES pybind11-stubgen)

# Get Torch CMake path
execute_process(
  COMMAND python -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TorchCMakePrefixPath
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get pybind11 CMake directory
execute_process(
  COMMAND python -m pybind11 --cmakedir
  OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Set paths
list(APPEND CMAKE_PREFIX_PATH "${TorchCMakePrefixPath}")
get_filename_component(PARENT_DIR ${CMAKE_SOURCE_DIR} DIRECTORY)


# Find libraries
set(Torch_KINETO_LIBRARIES "")
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

add_compile_options(-Wall)
add_compile_options(-Wunused-variable)


# Helper function to create Python modules
function(add_pybind_module TARGET_NAME)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "" "SOURCES;LIBS")
    
    # Create the Python module
    pybind11_add_module(${TARGET_NAME} ${ARG_SOURCES})
    target_link_libraries(${TARGET_NAME} PRIVATE 
        Python::Python 
        ${TORCH_LIBRARIES} 
        ${TORCH_PYTHON_LIBRARY} 
        ${ARG_LIBS}
    )
    target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include)
    
    # Stub generation
    if(PYBIND11_STUBGEN_EXECUTABLE)
        add_custom_command(
            TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E env 
                PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}:$ENV{PYTHONPATH}
                ${PYBIND11_STUBGEN_EXECUTABLE}
                    --output-dir ${CMAKE_CURRENT_BINARY_DIR}
                    ${TARGET_NAME}
            COMMENT "Generating pybind11 stubs for ${TARGET_NAME}"
        )
    endif()
endfunction()

# Add modules
add_pybind_module(my_addition SOURCES addition.cu)
