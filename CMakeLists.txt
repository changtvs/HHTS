cmake_minimum_required (VERSION 2.8)
project (superpixel_benchmark)

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
add_library(hhts hhts.cpp)
target_link_libraries(hhts ${OpenCV_LIBRARIES})
