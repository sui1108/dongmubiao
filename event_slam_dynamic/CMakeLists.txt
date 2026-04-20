cmake_minimum_required(VERSION 3.0.2)
project(event_slam_dynamic)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  visualization_msgs
  geometry_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/dynamic_filter_node.py
  scripts/result_visualizer_node.py
  scripts/raw_bridge_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

catkin_python_setup()
