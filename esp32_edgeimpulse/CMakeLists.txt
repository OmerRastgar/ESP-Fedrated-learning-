cmake_minimum_required(VERSION 3.16)

set(EXTRA_CXX_FLAGS "-Wno-error=missing-field-initializers -Wno-error=deprecated-declarations -fpermissive")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_CXX_FLAGS}")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(fl_client_edgeimpulse)
