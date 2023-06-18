cmake_minimum_required(VERSION 3.10)

project(gemm)
set(CMAKE_CXX_COMPILER icpx)
set(PRECISIONS         "s" "d" "c" "z")
get_filename_component(TOOLS_PATH                "${CMAKE_SOURCE_DIR}/../../tools" ABSOLUTE)
get_filename_component(CPP_SYCL_FPGA_PATH        "${CMAKE_SOURCE_DIR}/../../../.." ABSOLUTE)
get_filename_component(PRODUCTIVE_LIBRARIES_PATH "${CMAKE_SOURCE_DIR}/../.."       ABSOLUTE)
get_filename_component(TEST_PATH                 "${CMAKE_SOURCE_DIR}/../../test"  ABSOLUTE)

if(NOT DEFINED ENV{MKLROOT})
    message(FATAL_ERROR "MKLROOT is not set")
endif()

# Create targets for generating OneAPI files, reporting of resource usage, and synthesizing bitstreams.
set(ENV{KERNEL} "gemm")
include(../reconfigurable_matmul/CMakeInterface.txt)

# Generate correctness tests. They assume a tiny systolic array on an FPGA emulator.
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
foreach(N RANGE 0 3)
    add_custom_command(OUTPUT  ${CMAKE_SOURCE_DIR}/bin/test_${N}
                       COMMAND ${CMAKE_CXX_COMPILER}
                               -I${TOOLS_PATH}/Halide/include -I$ENV{MKLROOT}/include -I${PRODUCTIVE_LIBRARIES_PATH}/include -I${RECONFIGURABLE_MATMUL_DIR}
                               -I${RECONFIGURABLE_MATMUL_DIR}/oneapi -I${CPP_SYCL_FPGA_PATH}/include
                               -DTINY -DMKL_ILP64 -DT2SP_NDEBUG -DT2SP_TEST_${N} -DFPGA_EMULATOR
                               -I${TEST_PATH}/include -I${TEST_PATH}/googletest/include
                               -fsycl -fintelfpga
                               ${CMAKE_SOURCE_DIR}/test.cpp
                               ${RECONFIGURABLE_MATMUL_DIR}/oneapi/smatmul_tiny_a10.cpp
                               ${RECONFIGURABLE_MATMUL_DIR}/oneapi/dmatmul_tiny_a10.cpp
                               # ${RECONFIGURABLE_MATMUL_DIR}/oneapi/cmatmul_tiny_a10.cpp removed as we cannot generate code for complex type.
                               # ${RECONFIGURABLE_MATMUL_DIR}/oneapi/zmatmul_tiny_a10.cpp
                               -L${TOOLS_PATH}/Halide/lib -L$ENV{MKLROOT}/lib/intel64
                               -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lsycl -lOpenCL -lpthread -lm -ldl
                               -L${TEST_PATH}/googletest/lib
                               -lgtest -lgtest_main
                               -lHalide
                               -o ${CMAKE_SOURCE_DIR}/bin/test_${N}
                       DEPENDS oneapi_sgemm_tiny_a10
                               oneapi_dgemm_tiny_a10
                               #oneapi_cgemm_tiny_a10 removed as we cannot generate code for complex type.
                               #oneapi_zgemm_tiny_a10
                               ${CMAKE_SOURCE_DIR}/test.cpp
                       WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                       COMMENT "Generate tests"
                       VERBATIM)
endforeach()
add_custom_target(tests DEPENDS ${CMAKE_SOURCE_DIR}/bin/test_0
                                ${CMAKE_SOURCE_DIR}/bin/test_1
                                ${CMAKE_SOURCE_DIR}/bin/test_2
                                ${CMAKE_SOURCE_DIR}/bin/test_3)

# CMake does not allow multiple custom commands whose targets are all the same (e.g. demo.o).

# Create targets for demonstrating performance running on real hardware.
#foreach(PRECISION ${PRECISIONS})
#    string(TOUPPER ${PRECISIONS} UPPER_PRECISION)
#    foreach(SIZE "tiny" "large")
#        string(TOUPPER ${SIZE} UPPER_SIZE)
#        foreach(HW "a10" "s10")
#            string(TOUPPER ${HW} UPPER_HW)
#            # Generate a demo application that will run on real hardware
#            add_custom_command(OUTPUT  ${CMAKE_BINARY_DIR}/demo.o
#                               COMMAND ${CMAKE_CXX_COMPILER}
#                                       -I${TOOLS_PATH}/Halide/include -I${PRODUCTIVE_LIBRARIES_PATH}/include
#                                       -DT2SP_${UPPER_PRECISION}MATMUL -D${UPPER_SIZE} -D${UPPER_HW}
#                                       -fsycl -fintelfpga
#                                       -c ${CMAKE_SOURCE_DIR}/demo.cpp
#                                       -o ${CMAKE_BINARY_DIR}/demo.o
#                               DEPENDS ${CMAKE_SOURCE_DIR}/demo.cpp
#                               WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
#                               COMMENT "Generating a demo application"
#                               VERBATIM)
#            add_custom_command(OUTPUT  ${CMAKE_SOURCE_DIR}/bin/demo_${PRECISION}$ENV{KERNEL}_${SIZE}_${HW}
#                               COMMAND ${CMAKE_CXX_COMPILER}
#                                       -fsycl -fintelfpga
#                                       ${CMAKE_BINARY_DIR}/demo.o ${RECONFIGURABLE_MATMUL_DIR}/bin/${PRECISION}matmul_${SIZE}_${HW}.a
#                                       -o ${CMAKE_SOURCE_DIR}/bin/demo_${PRECISION}$ENV{KERNEL}_${SIZE}_${HW}
#                               DEPENDS ${CMAKE_BINARY_DIR}/demo.o
#                                       synthesize_${PRECISION}$ENV{KERNEL}_${SIZE}_${HW}
#                               WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
#                               COMMENT "Linking the demo application with the device image"
#                               VERBATIM)
#            add_custom_target(demo_${PRECISION}$ENV{KERNEL}_${SIZE}_${HW} DEPENDS ${CMAKE_SOURCE_DIR}/bin/demo_${PRECISION}$ENV{KERNEL}_${SIZE}_${HW})
#        endforeach()
#    endforeach()
#endforeach()
