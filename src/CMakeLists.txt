########################################################################

#### User settings #### * Modify here *

set(OPENCV_INC_DIR "C:/opencv3/build/include")
set(OPENCV_LIB_DIR "C:/opencv3/build/x64/vc12/lib")
set(OPENCV_LIBS_DEBUG debug opencv_ts300d.lib debug opencv_world300d.lib)
set(OPENCV_LIBS_RELEASE optimized opencv_ts300.lib optimized opencv_world300.lib)
set(OPENCV_LIBS ${OPENCV_LIBS_DEBUG} ${OPENCV_LIBS_RELEASE})

########################################################################



#### CMake Settings ####

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
cmake_policy(SET CMP0015 OLD)	#// Do not add relative path if the designated path is not exist
add_definitions(-DUNICODE -D_UNICODE)	#// Set Unicode, not MultiBytes


#### Solution Settings ####

project(Label2D)

if(MSVC AND NOT PLATFORM EQUAL 64)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LARGEADDRESSAWARE")
  message(STATUS "MSVC: Enabled large address awareness")
endif()


#### Includes and Libraries ####

include_directories(${OPENCV_INC_DIR})
link_directories(${OPENCV_LIB_DIR})


#### Required files ####

set(LABEL2D label2d.hpp label2d.cpp)

#### Projects Settings ####

add_executable(Label2D main.cpp ${LABEL2D})
target_link_libraries(Label2D ${OPENCV_LIBS})