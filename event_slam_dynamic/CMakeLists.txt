cmake_minimum_required(VERSION 3.0.2)
project(event_slam_dynamic)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  visualization_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/dynamic_filter_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY src/event_slam_dynamic
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
)
