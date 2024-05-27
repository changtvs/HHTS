cmake_minimum_required(VERSION 3.26.0)
get_filename_component(EXE_NAME ${CMAKE_CURRENT_LIST_DIR} NAME)
project(${EXE_NAME} VERSION 0.1.0)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

find_package(OpenCV CONFIG REQUIRED)
find_package(Boost COMPONENTS system timer REQUIRED)

file(GLOB src_files
    "**/*.c"
    "**/*.h"
    "**/*.cpp"
    "*.c"
    "*.h"
    "*.cpp"
)
add_executable(${EXE_NAME} ${src_files})

include_directories(${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
target_link_libraries(${EXE_NAME} PRIVATE ${OpenCV_LIBS} ${Boost_LIBRARIES})
set(OPENCV_ENABLE_NONFREE true)