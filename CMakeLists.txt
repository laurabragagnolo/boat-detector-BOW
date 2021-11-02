cmake_minimum_required (VERSION 2.8)

project(Laura_Bragagnolo_boat_detector)

find_package (OpenCV REQUIRED)

include_directories(
	${OpenCV_INCLUDE_DIRS}
	Detector_Utils 
)

add_executable(
	${PROJECT_NAME}
	src/Laura_Bragagnolo_boat_detector.cpp
)

add_library (
	Detector_Utils
	Detector_Utils/Detector_Utils.h
	Detector_Utils/Detector_Utils.cpp
)

target_link_libraries(
	${PROJECT_NAME}
	${OpenCV_LIBS}
	Detector_Utils
)
