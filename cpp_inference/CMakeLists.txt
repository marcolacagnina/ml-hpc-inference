cmake_minimum_required(VERSION 3.20)
project(ONNXInference CXX)

# ------------------ Standard C++ ------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------ ONNX Runtime ------------------
# Check if ONNXRUNTIME_DIR is defined (passed from CI)
if(DEFINED ONNXRUNTIME_DIR)
    message(STATUS "Using ONNX Runtime from ONNXRUNTIME_DIR=${ONNXRUNTIME_DIR}")
    set(ONNXRUNTIME_ROOT "${ONNXRUNTIME_DIR}")
else()
    # Default for macOS Homebrew
    set(ONNXRUNTIME_ROOT "/opt/homebrew/opt/onnxruntime")
    message(STATUS "Using default ONNX Runtime path: ${ONNXRUNTIME_ROOT}")
endif()

include_directories(${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/lib)

# ------------------ YAML-CPP ------------------
find_package(yaml-cpp REQUIRED)

# ------------------ Executable ------------------
add_executable(onnx_inference
    src/main.cpp
    src/benchmark.cpp
)

# ------------------ Link libraries ------------------
target_link_libraries(onnx_inference
    onnxruntime
    yaml-cpp::yaml-cpp
)
