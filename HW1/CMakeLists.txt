cmake_minimum_required(VERSION 2.8)			#Specify the minimum CMake version
project(hello)								#Specify the name of the project
find_package(CUDA REQUIRED)					#find the CUDA libraries
include_directories(${CUDA_INCLUDE_DIRS})	#Specify the CUDA include directories for the project
cuda_add_executable(hello main.cu)			#create an executable from the given CUDA files

#specify any additional libraries here (CUFFT and CUBLAS can be useful)
target_link_libraries(hello
					  )