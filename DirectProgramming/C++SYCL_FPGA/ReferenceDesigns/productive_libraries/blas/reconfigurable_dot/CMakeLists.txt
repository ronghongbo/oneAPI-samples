cmake_minimum_required(VERSION 3.10)

project(reconfigurable_dot)
set(CMAKE_CXX_COMPILER icpx)
set(KERNEL "dot")
set(PRECISIONS "s" "d" "c" "z")
set(OPENCL_1_2_1_BSP /opt/a10/inteldevstack/a10_gx_pac_ias_1_2_1_pv/opencl/opencl_bsp)

# Due to restrictions of file size, we have split libHalide.a into parts. Join them into the original binary
if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/../../tools/Halide/lib/libHalide.a")
    execute_process(COMMAND cat libHalide.part.aa libHalide.part.ab libHalide.part.ac libHalide.part.ad libHalide.part.ae libHalide.part.af libHalide.part.ag
                    OUTPUT_FILE libHalide.a
                    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../../tools/Halide/lib)
endif()
get_filename_component(TOOLS_PATH         "${CMAKE_SOURCE_DIR}/../../tools"         ABSOLUTE)
get_filename_component(CPP_SYCL_FPGA_PATH "${CMAKE_SOURCE_DIR}/../../../../include" ABSOLUTE)
get_filename_component(PRODUCTIVE_LIBRARIES_PATH "${CMAKE_SOURCE_DIR}/../.."       ABSOLUTE)

# FPGA board selection
if(NOT DEFINED FPGA_DEVICE)
    # For whatever reason, it is faster to synthesize with OpenCL 1.2.1 BSP.
    # set(FPGA_DEVICE "intel_a10gx_pac:pac_a10")
    set(FPGA_DEVICE "${OPENCL_1_2_1_BSP}:pac_a10")
    message(STATUS "FPGA_DEVICE was not specified. "
                   "Configuring the design to run on the default FPGA board ${FPGA_DEVICE} (Intel(R) PAC with Intel Arria(R) 10 GX FPGA). "
                   "Please refer to the README for information on board selection.")
else()
    if((NOT("${FPGA_DEVICE}" STREQUAL "intel_s10sx_pac:pac_s10")) AND (NOT("${FPGA_DEVICE}" STREQUAL "${OPENCL_1_2_1_BSP}:pac_a10")))
        message(FATAL_ERROR "Unsupported device ${FPGA_DEVICE}")
    endif()
    message(STATUS "Configuring the design to run on FPGA board ${FPGA_DEVICE}")
endif()

# Seed for synthesis
if(NOT DEFINED SEED)
    set(SEED 1)
endif()

###############################################################################
### Create results directories
###############################################################################
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/oneapi)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/reports)

# Below we use add_custom_command and add_custom_target to make sure independent targets can be built concurrently without conflict,
# even if they depend on the same file. This is inspired by an example from
# https://cmake.org/cmake/help/latest/command/add_custom_command.html#example-generating-files-for-multiple-targets
foreach(PRECISION ${PRECISIONS})
    foreach(SIZE "tiny" "large")
        foreach(HW "a10" "s10")
            string(TOUPPER ${PRECISION} UPPER_PRECISION)
            string(TOUPPER ${KERNEL}    UPPER_KERNEL)
            string(TOUPPER ${SIZE}      UPPER_SIZE)
            string(TOUPPER ${HW}        UPPER_HW)

            # Generate OneAPI files
            add_custom_command(OUTPUT  ${CMAKE_SOURCE_DIR}/oneapi/${PRECISION}${KERNEL}_${SIZE}_${HW}.cpp
                               COMMAND ${CMAKE_CXX_COMPILER} -DT2SP_${UPPER_PRECISION}${UPPER_KERNEL} -D${UPPER_SIZE} -D${UPPER_HW}
                               -DKERNEL="t2sp::blas::row_major::${PRECISION}${KERNEL}::${PRECISION}${KERNEL}" -DOUTPUT_FILE="${PRECISION}${KERNEL}_${SIZE}_${HW}.cpp"
                                       -I${TOOLS_PATH}/Halide/include -I${CMAKE_SOURCE_DIR}
                                       -std=gnu++11
                                       ${CMAKE_SOURCE_DIR}/dot.cpp
                                       -L${TOOLS_PATH}/Halide/lib -lpthread -lz -ldl -lHalide
                                       -o ${CMAKE_BINARY_DIR}/oneapi_${PRECISION}${KERNEL}_${SIZE}_${HW}
                               COMMAND env CLEARCODE=1 ${CMAKE_BINARY_DIR}/oneapi_${PRECISION}${KERNEL}_${SIZE}_${HW}
                               COMMAND cp ${CMAKE_BINARY_DIR}/${PRECISION}${KERNEL}_${SIZE}_${HW}.cpp ${CMAKE_SOURCE_DIR}/oneapi
                               DEPENDS ${CMAKE_SOURCE_DIR}/dot.cpp
                                       ${CMAKE_SOURCE_DIR}/parameters.h
                                       ${TOOLS_PATH}/Halide/include/Halide.h
                               WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                               COMMENT "Generating OneAPI files"
                               VERBATIM)
            add_custom_target(oneapi_${PRECISION}${KERNEL}_${SIZE}_${HW} DEPENDS ${CMAKE_SOURCE_DIR}/oneapi/${PRECISION}${KERNEL}_${SIZE}_${HW}.cpp)

            # Generate a report
            file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/reports/${PRECISION}${KERNEL}_${SIZE}_${HW})
            add_custom_command(OUTPUT  ${CMAKE_SOURCE_DIR}/reports/${PRECISION}${KERNEL}_${SIZE}_${HW}/report.html
                                       ${CMAKE_SOURCE_DIR}/reports/${PRECISION}${KERNEL}_${SIZE}_${HW}/resources
                               COMMAND ${CMAKE_CXX_COMPILER}
                                       -I${CMAKE_SOURCE_DIR}/oneapi -I${CPP_SYCL_FPGA_PATH} -I${PRODUCTIVE_LIBRARIES_PATH}/include -I${TOOLS_PATH}/Halide/include
                                       ${CMAKE_SOURCE_DIR}/oneapi/${PRECISION}${KERNEL}_${SIZE}_${HW}.cpp
                                       ${PRODUCTIVE_LIBRARIES_PATH}/include/halide_runtime_etc.cpp
                                       -Xshardware -Xstarget=${FPGA_DEVICE}
                                       -Xsffp-reassociate -Xsffp-contract=fast -Xsdsp-mode=prefer-dsp -Xsprofile -Xsclock=360MHz
                                       -fsycl -fintelfpga -fsycl-link=early -o ${PRECISION}${KERNEL}_${SIZE}_${HW}.early
                               COMMAND cp -rf ${PRECISION}${KERNEL}_${SIZE}_${HW}.early.prj/reports/report.html ${CMAKE_SOURCE_DIR}/reports/${PRECISION}${KERNEL}_${SIZE}_${HW}
                               COMMAND cp -rf ${PRECISION}${KERNEL}_${SIZE}_${HW}.early.prj/reports/resources   ${CMAKE_SOURCE_DIR}/reports/${PRECISION}${KERNEL}_${SIZE}_${HW}
                               DEPENDS ${PRODUCTIVE_LIBRARIES_PATH}/include/halide_runtime_etc.cpp
                                       ${PRODUCTIVE_LIBRARIES_PATH}/include/halide_runtime_etc.hpp
                                       ${PRODUCTIVE_LIBRARIES_PATH}/include/complex_helper.hpp
                                       ${PRODUCTIVE_LIBRARIES_PATH}/include/pipe_wrapper.hpp
                                       oneapi_${PRECISION}${KERNEL}_${SIZE}_${HW}
                               WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                               COMMENT "Generating a report"
                               VERBATIM)
            add_custom_target(report_${PRECISION}${KERNEL}_${SIZE}_${HW} DEPENDS ${CMAKE_SOURCE_DIR}/reports/${PRECISION}${KERNEL}_${SIZE}_${HW}/report.html
                                                                                 ${CMAKE_SOURCE_DIR}/reports/${PRECISION}${KERNEL}_${SIZE}_${HW}/resources)

            # Generate an image
            add_custom_command(OUTPUT  ${CMAKE_SOURCE_DIR}/bin/${PRECISION}${KERNEL}_${SIZE}_${HW}.a
                                       ${CMAKE_SOURCE_DIR}/reports/${PRECISION}${KERNEL}_${SIZE}_${HW}/acl_quartus_report.txt
                               COMMAND ${CMAKE_CXX_COMPILER}
                                       -I${CMAKE_SOURCE_DIR}/oneapi -I${CPP_SYCL_FPGA_PATH} -I${PRODUCTIVE_LIBRARIES_PATH}/include -I${TOOLS_PATH}/Halide/include
                                       ${CMAKE_SOURCE_DIR}/oneapi/${PRECISION}${KERNEL}_${SIZE}_${HW}.cpp
                                       ${PRODUCTIVE_LIBRARIES_PATH}/include/halide_runtime_etc.cpp
                                       -Xshardware -Xstarget=${FPGA_DEVICE}
                                       -Xsffp-reassociate -Xsffp-contract=fast -Xsdsp-mode=prefer-dsp -Xsprofile -Xsclock=360MHz -Xsseed=${SEED}
                                       -fsycl -fintelfpga -fsycl-link=image -o ${PRECISION}${KERNEL}_${SIZE}_${HW}.a
                               COMMAND cp -rf ${PRECISION}${KERNEL}_${SIZE}_${HW}.a                          ${CMAKE_SOURCE_DIR}/bin
                               COMMAND cp -rf ${PRECISION}${KERNEL}_${SIZE}_${HW}.prj/reports/report.html    ${CMAKE_SOURCE_DIR}/reports/${PRECISION}${KERNEL}_${SIZE}_${HW}
                               COMMAND cp -rf ${PRECISION}${KERNEL}_${SIZE}_${HW}.prj/reports/resources      ${CMAKE_SOURCE_DIR}/reports/${PRECISION}${KERNEL}_${SIZE}_${HW}
                               COMMAND cp -rf ${PRECISION}${KERNEL}_${SIZE}_${HW}.prj/acl_quartus_report.txt ${CMAKE_SOURCE_DIR}/reports/${PRECISION}${KERNEL}_${SIZE}_${HW}
                               DEPENDS ${PRODUCTIVE_LIBRARIES_PATH}/include/halide_runtime_etc.cpp
                                       ${PRODUCTIVE_LIBRARIES_PATH}/include/halide_runtime_etc.hpp
                                       ${PRODUCTIVE_LIBRARIES_PATH}/include/complex_helper.hpp
                                       ${PRODUCTIVE_LIBRARIES_PATH}/include/pipe_wrapper.hpp
                                       oneapi_${PRECISION}${KERNEL}_${SIZE}_${HW}
                               WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                               COMMENT "Generating an image"
                               VERBATIM)
            add_custom_target(synthesize_${PRECISION}${KERNEL}_${SIZE}_${HW} DEPENDS ${CMAKE_SOURCE_DIR}/bin/${PRECISION}${KERNEL}_${SIZE}_${HW}.a)
        endforeach()
    endforeach()
endforeach()
