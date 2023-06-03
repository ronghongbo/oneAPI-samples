 cmake_minimum_required(VERSION 3.10)

 project(gemm)
set(CMAKE_CXX_COMPILER icpx)
set(PRECISIONS         "s" "d" "c" "z")
set(TOOLS_PATH         ${CMAKE_SOURCE_DIR}/../../tools)
set(CPP_SYCL_FPGA_PATH ${CMAKE_SOURCE_DIR}/../../../..)

if(NOT DEFINED ENV{MKLROOT})
    message(FATAL_ERROR "MKLROOT is not set")
endif()

# Create targets for generating OneAPI files, reporting of resource usage, and synthesizing bitstreams.
set(ENV{KERNEL} "gemm")
include(../reconfigurable_matmul/CMakeInterface.txt)

# Generate correctness tests. They assume a tiny systolic array on an FPGA emulator.
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
foreach(N RANGE 0 4)
    add_custom_command(OUTPUT  ${CMAKE_SOURCE_DIR}/bin/test_${N}
                       COMMAND ${CMAKE_CXX_COMPILER}
                               -I${TOOLS_PATH}/Halide/include -I$ENV{MKLROOT}/include -I${RECONFIGURABLE_MATMUL_DIR}
                               -I${RECONFIGURABLE_MATMUL_DIR}/oneapi -I${CPP_SYCL_FPGA_PATH}/include
                               -D$TINY -DMKL_ILP64 -DT2SP_NDEBUG -DT2SP_TEST_${N} -DFPGA_EMULATOR
                               -fsycl -fintelfpga
                               ${CMAKE_SOURCE_DIR}/test.cpp
                               -L${TOOLS_PATH}/Halide/lib -L$ENV{MKLROOT}/lib/intel64
                               -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core sycl -lOpenCL -lpthread -lm -ldl
                               -o ${CMAKE_SOURCE_DIR}/bin/test_${N}
                       DEPENDS oneapi_sgemm_tiny_a10
                               oneapi_dgemm_tiny_a10
                               oneapi_cgemm_tiny_a10
                               oneapi_zgemm_tiny_a10
                               ${CMAKE_SOURCE_DIR}/test.cpp
                       WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                       COMMENT "Generate tests"
                       VERBATIM)
endforeach()
add_custom_target(tests DEPENDS ${CMAKE_SOURCE_DIR}/bin/test_0
                                ${CMAKE_SOURCE_DIR}/bin/test_1
                                ${CMAKE_SOURCE_DIR}/bin/test_2
                                ${CMAKE_SOURCE_DIR}/bin/test_3
                                ${CMAKE_SOURCE_DIR}/bin/test_4)

# Create targets for demonstrating performance running on real hardware.
# TODO: modify hardware_demo.cpp so that it can work with all precisions, sizes, and hardwares.
foreach(PRECISION ${PRECISIONS})
    foreach(SIZE "tiny" "large")
        foreach(HW "a10" "s10")
            string(TOUPPER ${SIZE} UPPER_SIZE)
            # Generate a demo application that will run on real hardware
            add_custom_command(OUTPUT  ${CMAKE_SOURCE_DIR}/bin/demo_${PRECISION}$ENV{KERNEL}_${SIZE}_${HW}
                               COMMAND ${CMAKE_CXX_COMPILER}
                                       -reuse-exe=${RECONFIGURABLE_MATMUL_DIR}/bin/${PRECISION}matmul_${SIZE}_${HW}.image
                                       -I${TOOLS_PATH}/Halide/include -I$ENV{MKLROOT}/include -I${RECONFIGURABLE_MATMUL_DIR}
                                       -I${RECONFIGURABLE_MATMUL_DIR}/oneapi -I${CPP_SYCL_FPGA_PATH}/include
                                       -fsycl -fintelfpga
                                       ${CMAKE_SOURCE_DIR}/hardware_demo.cpp
                                       -o ${CMAKE_SOURCE_DIR}/bin/demo_${PRECISION}$ENV{KERNEL}_${SIZE}_${HW}
                               DEPENDS synthesize_${PRECISION}$ENV{KERNEL}_${SIZE}_${HW}
                               WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                               COMMENT "Start to generate a demo application using the image"
                               VERBATIM)
            add_custom_target(demo_${PRECISION}$ENV{KERNEL}_${SIZE}_${HW} DEPENDS ${CMAKE_SOURCE_DIR}/bin/demo_${PRECISION}$ENV{KERNEL}_${SIZE}_${HW})
        endforeach()
    endforeach()
endforeach()
