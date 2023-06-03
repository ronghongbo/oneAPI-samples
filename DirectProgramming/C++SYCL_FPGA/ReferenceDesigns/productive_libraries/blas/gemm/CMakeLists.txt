 cmake_minimum_required(VERSION 3.16)

 project(gemm)
set(KERNELS "sgemm" "dgemm" "cgemm" "zgemm")
set(CMAKE_CXX_COMPILER icpx)
set(TOOLS_PATH "${CMAKE_SOURCE_DIR}/../../tools")
set(CPP_SYCL_FPGA_PATH "${CMAKE_SOURCE_DIR}/../../../..")

if(NOT DEFINED ENV{MKLROOT})
    message(FATAL_ERROR "MKLROOT is not set")
endif()

# Create targets for generating OneAPI files, reporting of resource usage, and synthesizing bitstreams.
include(../reconfigurable_matmul/CMakeInterface.txt)

# Create targets for testing correctness of the kernels, and demonstrating performance running on real hardware.
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
foreach(KERNEL ${KERNELS})
    string(SUBSTRING ${KERNEL} 0 1 PRECISION)
    foreach(SIZE "tiny" "large")
        foreach(HW "a10" "s10")
            string(TOUPPER ${SIZE} UPPER_SIZE)

            # Generate tests
            foreach(N RANGE 0 4)
                add_custom_command(OUTPUT  ${CMAKE_SOURCE_DIR}/bin/test_${N}
                                   COMMAND ${CMAKE_CXX_COMPILER}
                                           -I${TOOLS_PATH}/Halide/include -I$ENV{MKLROOT}/include -I${RECONFIGURABLE_MATMUL_DIR}
                                           -I${RECONFIGURABLE_MATMUL_DIR}/oneapi -I${CPP_SYCL_FPGA_PATH}/include
                                           -D${UPPER_SIZE} -DMKL_ILP64 -DT2SP_NDEBUG -DT2SP_TEST_${N} -DFPGA_EMULATOR
                                           -fsycl -fintelfpga
                                           -L${TOOLS_PATH}/Halide/lib -L$ENV{MKLROOT}/lib/intel64
                                           -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core sycl -lOpenCL -lpthread -lm -ldl
                                           ${CMAKE_SOURCE_DIR}/test.cpp
                                           -o ${CMAKE_SOURCE_DIR}/bin/test_${N}
                                   DEPENDS gen_oneapi_${KERNEL}_${SIZE}_${HW}
                                           ${CMAKE_SOURCE_DIR}/test.cpp
                                   WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                                   COMMENT "Generate tests"
                                   VERBATIM)
            endforeach()
            add_custom_target(test_${KERNEL}_${SIZE}_${HW} DEPENDS ${CMAKE_SOURCE_DIR}/bin/test_0
                                                                   ${CMAKE_SOURCE_DIR}/bin/test_1
                                                                   ${CMAKE_SOURCE_DIR}/bin/test_2
                                                                   ${CMAKE_SOURCE_DIR}/bin/test_3
                                                                   ${CMAKE_SOURCE_DIR}/bin/test_4)

            # Generate a demo application that will run on real hardware
            add_custom_command(OUTPUT  ${RECONFIGURABLE_MATMUL_DIR}/bin/${PRECISION}matmul_${SIZE}_${HW}.image
                                       ${RECONFIGURABLE_MATMUL_DIR}/reports/${PRECISION}matmul_${SIZE}_${HW}/report.html
                                       ${RECONFIGURABLE_MATMUL_DIR}/reports/${PRECISION}matmul_${SIZE}_${HW}/resources
                                       ${CMAKE_SOURCE_DIR}/bin/demo
                               COMMAND ${CMAKE_CXX_COMPILER}
                                       -reuse-exe=${RECONFIGURABLE_MATMUL_DIR}/bin/${PRECISION}matmul_${SIZE}_${HW}.image
                                       -I${TOOLS_PATH}/Halide/include -I$ENV{MKLROOT}/include -I${RECONFIGURABLE_MATMUL_DIR}
                                       -I${RECONFIGURABLE_MATMUL_DIR}/oneapi -I${CPP_SYCL_FPGA_PATH}/include
                                       -fsycl -fintelfpga
                                       ${CMAKE_SOURCE_DIR}/hardware_demo.cpp
                                       -o ${CMAKE_SOURCE_DIR}/bin/demo
                               DEPENDS synthesize_${KERNEL}_${SIZE}_${HW}
                               WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                               COMMENT "Start to generate a demo application using the image"
                               VERBATIM)
            add_custom_target(demo_${KERNEL}_${SIZE}_${HW} DEPENDS ${RECONFIGURABLE_MATMUL_DIR}/bin/${PRECISION}matmul_${SIZE}_${HW}.image
                                                                   ${CMAKE_SOURCE_DIR}/bin/demo)
        endforeach()
    endforeach()
endforeach()


