cmake_minimum_required(VERSION 3.27)
project(Truck_Platooning CUDA CXX)

set(CMAKE_CXX_STANDARD 17)

set(THREADS_PREFER_PTHREAD_FLAG ON)
if (WIN32)
    set(WSOCK_LIBRARY "ws2_32.lib")
endif()

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_75,code=sm_75")

add_executable(Truck_Platooning main.cpp
        Truck/Truck.cuh
        Truck/Truck.cu
        Truck/LeaderTruck/LeaderTruck.h
        Truck/LeaderTruck/LeaderTruck.cpp
        Truck/FollowerTruck/FollowerTruck.cpp
        Truck/FollowerTruck/FollowerTruck.h
)


target_include_directories(Truck_Platooning PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_link_libraries(Truck_Platooning PRIVATE ${WSOCK_LIBRARY} ${CUDA_LIBRARIES})



