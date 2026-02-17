cmake_minimum_required(VERSION 2.8.3)
project(livox_laser_simulation)

## Compile as C++11, supported in ROS Kinetic and newer
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  tf
  sensor_msgs
  std_msgs
  message_generation
)

find_package(gazebo REQUIRED)
find_package(PCL REQUIRED)
include_directories(${GAZEBO_INCLUDE_DIRS})
add_message_files(FILES
    CustomPoint.msg
    CustomMsg.msg
  #   Message2.msg
  )

generate_messages(DEPENDENCIES
  std_msgs
)
#To solve the error which gazebo does not handle the include file well, we must add this line.
include_directories(/usr/include/)

link_directories(${GAZEBO_LIBRARY_DIRS})

#This is designed for whose proto installed in system is not 2.6. We can install the version of proto in local dir
#include_directories(/home/lfc/proto/include/)
#link_directories(/home/lfc/proto/lib/)

include_directories(
         include
        ${catkin_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
)

catkin_package(
        INCLUDE_DIRS include
        LIBRARIES livox_laser_simulation
        CATKIN_DEPENDS tf
)

add_library(livox_laser_simulation SHARED src/livox_points_plugin.cpp src/livox_ode_multiray_shape.cpp)
target_link_libraries(livox_laser_simulation ${catkin_LIBRARIES} RayPlugin)

target_link_libraries(livox_laser_simulation libprotobuf.so.9)

# Install Python scripts with executable permissions
catkin_install_python(PROGRAMS
  scripts/object_detection.py
  scripts/person_detector.py
  scripts/wall_detector.py
  scripts/truck_detector.py
  scripts/tray_detector.py
  scripts/oscillate_person.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install Python dependencies
find_package(PythonInterp 3 REQUIRED)
add_custom_target(install_python_deps ALL
  COMMAND ${PYTHON_EXECUTABLE} -m pip install -r ${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt
  COMMENT "Installing Python dependencies from requirements.txt"
)
