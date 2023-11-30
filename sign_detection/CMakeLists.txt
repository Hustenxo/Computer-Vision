cmake_minimum_required(VERSION 3.5)
project(sign_detection)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(libpsaf REQUIRED)
find_package(libpsaf_msgs REQUIRED)
find_package(OpenCV REQUIRED)

# find libtorch library
set(Torch_DIR ../../lib/libtorch/share/cmake/Torch)
find_package(Torch PATHS ${Torch_DIR} NO_DEFAULT REQUIRED)

# define sign model directory
add_compile_definitions(SIGN_MODEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/models")

# Add executable for sign detection node
add_executable(sign_detection
        src/sign_detection.cpp
        src/sign_detection_node.cpp
        include/sign_detection/definitions.hpp)

target_include_directories(sign_detection PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${OpenCV_INCLUDE_DIRS})
target_link_libraries(sign_detection
        ${OpenCV_LIBS}
        ${TORCH_LIBRARIES})
ament_target_dependencies(sign_detection
        "rclcpp"
        "libpsaf"
        "libpsaf_msgs"
        "OpenCV"
        "Torch")

# Install executable
install(TARGETS
        sign_detection
        DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  set(ament_cmake_copyright_FOUND TRUE)

  # GTest
  find_package(ament_cmake_gtest)

  # Test sample
  ament_add_gtest(sign_detection_test test/sign_detection_test.cpp src/sign_detection_node.cpp)

  set_tests_properties(sign_detection_test PROPERTIES
          ENVIRONMENT
          "TEST_DATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/test/test_data")

  target_include_directories(sign_detection_test PUBLIC
          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
          $<INSTALL_INTERFACE:include>
          ${OpenCV_INCLUDE_DIRS})

  target_link_libraries(sign_detection_test
          ${OpenCV_LIBS}
          ${TORCH_LIBRARIES})

  ament_target_dependencies(sign_detection_test
          "rclcpp"
          "libpsaf"
          "libpsaf_msgs"
          "OpenCV"
          "Torch")

  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
