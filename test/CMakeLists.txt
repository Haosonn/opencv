cmake_minimum_required(VERSION 3.5)

# Define the test project
project(DNNFuncTest)
set(SOURCES test.cpp)
add_executable(${PROJECT_NAME} ${SOURCES})

# Use opencv debug version
set(OpenCV_DIR "../../build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


# set(RENDERDOC_PATH "C:\\Program Files\\RenderDoc")
if (DEFINED RENDERDOC_PATH)
  message("RENDERDOC_PATH is defined: ${RENDERDOC_PATH}")
  include_directories(${RENDERDOC_PATH})
else()
  message(WARNING "RENDERDOC_PATH is not defined. Set this variable to enable RENDERDOC_PATH")
endif()

# Add opencv lib
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
# message(${OpenCV_LIBS})

# Set compiler flags
set(CPP_FLAGS -g)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CPP_FLAGS}")
