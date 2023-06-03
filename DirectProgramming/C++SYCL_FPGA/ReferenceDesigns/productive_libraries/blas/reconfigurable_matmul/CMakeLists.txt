 cmake_minimum_required(VERSION 3.16)

 project(reconfigurable_matmul)
set(CMAKE_CXX_COMPILER icpx)
set(KERNELS "smatmul" "dmatmul" "cmatmul" "zmatmul")
set(OPENCL_1_2_1_BSP /opt/a10/inteldevstack/a10_gx_pac_ias_1_2_1_pv/opencl/opencl_bsp)

# Due to restrictions of file size, we have split libHalide.a into parts. Join them into the original binary
if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/../../tools/Halide/lib/libHalide.a")
    execute_process(COMMAND cat libHalide.part.aa libHalide.part.ab libHalide.part.ac libHalide.part.ad libHalide.part.ae libHalide.part.af libHalide.part.ag
                    OUTPUT_FILE libHalide.a
                    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../../tools/Halide/lib)
endif()
set(TOOLS_PATH "${CMAKE_CURRENT_LIST_DIR}/../../tools")

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

###############################################################################
### Create results directories
###############################################################################
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/oneapi)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/reports)

# Below we use add_custom_command and add_custom_target to make sure independent targets can be built concurrently without conflict,
# even if they depend on the same file. This is inspired by an example from
# https://cmake.org/cmake/help/latest/command/add_custom_command.html#example-generating-files-for-multiple-targets
foreach(KERNEL ${KERNELS})
    string(SUBSTRING ${KERNEL} 0 1 PRECISION)
    foreach(SIZE "tiny" "large")
        foreach(HW "a10" "s10")
            string(TOUPPER ${PRECISION} UPPER_PRECISION)
            string(TOUPPER ${KERNEL}    UPPER_KERNEL)
            string(TOUPPER ${SIZE}      UPPER_SIZE)
            string(TOUPPER ${HW}        UPPER_HW)

            # Generate OneAPI files
            add_custom_command(OUTPUT  ${CMAKE_SOURCE_DIR}/oneapi/${KERNEL}_${SIZE}_${HW}.sycl.h
                                       ${CMAKE_SOURCE_DIR}/oneapi/complex_helper.hpp
                                       ${CMAKE_SOURCE_DIR}/oneapi/halide_runtime_etc.h
                                       ${CMAKE_SOURCE_DIR}/oneapi/pipe_wrapper.hpp
                               COMMAND ${CMAKE_CXX_COMPILER} -DT2SP_${UPPER_PRECISION}MATMUL -D${UPPER_SIZE} -D${UPPER_HW}
                                       -DOUTPUT_FILE="${KERNEL}_${SIZE}_${HW}"
                                       -I${TOOLS_PATH}/Halide/include -I${CMAKE_SOURCE_DIR}
                                       -std=gnu++11 ${CMAKE_SOURCE_DIR}/matrix_multiply.cpp
                                       -L${TOOLS_PATH}/Halide/lib -lpthread -lz -ldl -lHalide
                                       -o ${CMAKE_BINARY_DIR}/gen_oneapi_${KERNEL}_${SIZE}_${HW}
                               COMMAND env CLEARCODE=1 ${CMAKE_BINARY_DIR}/gen_oneapi_${KERNEL}_${SIZE}_${HW}
                               COMMAND cp ${CMAKE_BINARY_DIR}/${KERNEL}_${SIZE}_${HW}.sycl.h ${CMAKE_SOURCE_DIR}/oneapi
                               COMMAND cp ${CMAKE_BINARY_DIR}/complex_helper.hpp             ${CMAKE_SOURCE_DIR}/oneapi
                               COMMAND cp ${CMAKE_BINARY_DIR}/halide_runtime_etc.h           ${CMAKE_SOURCE_DIR}/oneapi
                               COMMAND cp ${CMAKE_BINARY_DIR}/pipe_wrapper.hpp               ${CMAKE_SOURCE_DIR}/oneapi
                               DEPENDS ${CMAKE_SOURCE_DIR}/matrix_multiply.cpp
                                       ${CMAKE_SOURCE_DIR}/parameters.h
                               WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                               COMMENT "Start to generate OneAPI files"
                               VERBATIM)
            add_custom_target(gen_oneapi_${KERNEL}_${SIZE}_${HW} DEPENDS ${CMAKE_SOURCE_DIR}/oneapi/${KERNEL}_${SIZE}_${HW}.sycl.h
                                                                       ${CMAKE_SOURCE_DIR}/oneapi/complex_helper.hpp
                                                                       ${CMAKE_SOURCE_DIR}/oneapi/halide_runtime_etc.h
                                                                       ${CMAKE_SOURCE_DIR}/oneapi/pipe_wrapper.hpp)

            # Generate a report
            file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/generated_reports/${KERNEL}_${SIZE}_${HW})
            add_custom_command(OUTPUT  ${CMAKE_SOURCE_DIR}/generated_reports/${KERNEL}_${SIZE}_${HW}/report.html
                                       ${CMAKE_SOURCE_DIR}/generated_reports/${KERNEL}_${SIZE}_${HW}/resources
                               COMMAND ${CMAKE_CXX_COMPILER}
                                       -I${CMAKE_SOURCE_DIR}/oneapi -I${CMAKE_SOURCE_DIR}/../../../../include
                                       -Xshardware -Xstarget=${FPGA_DEVICE}
                                       -Xsffp-reassociate -Xsffp-contract=fast -Xsdsp-mode=prefer-dsp -Xsprofile -Xsclock=360MHz
                                       ${CMAKE_SOURCE_DIR}/oneapi/${KERNEL}_${SIZE}_${HW}.sycl.h
                                       -fsycl -fintelfpga -fsycl-link=early -o ${KERNEL}_${SIZE}_${HW}.early
                               COMMAND cp -rf ${KERNEL}_${SIZE}_${HW}.early.prj/reports/report.html ${CMAKE_SOURCE_DIR}/generated_reports/${KERNEL}_${SIZE}_${HW}
                               COMMAND cp -rf ${KERNEL}_${SIZE}_${HW}.early.prj/reports/resources   ${CMAKE_SOURCE_DIR}/generated_reports/${KERNEL}_${SIZE}_${HW}
                               DEPENDS ${CMAKE_SOURCE_DIR}/oneapi/${KERNEL}_${SIZE}_${HW}.sycl.h # File-level dependency
                                       gen_oneapi_${KERNEL}_${SIZE}_${HW}                                 # Target-level dependency
                               WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                               COMMENT "Start to generate a report"
                               VERBATIM)
            add_custom_target(report_${KERNEL}_${SIZE}_${HW} DEPENDS ${CMAKE_SOURCE_DIR}/generated_reports/${KERNEL}_${SIZE}_${HW}/report.html
                                                                     ${CMAKE_SOURCE_DIR}/generated_reports/${KERNEL}_${SIZE}_${HW}/resources)

            # Generate an image
            add_custom_command(OUTPUT  ${CMAKE_SOURCE_DIR}/bin/${KERNEL}_${SIZE}_${HW}.image
                                       ${CMAKE_SOURCE_DIR}/generated_reports/${KERNEL}_${SIZE}_${HW}/report.html
                                       ${CMAKE_SOURCE_DIR}/generated_reports/${KERNEL}_${SIZE}_${HW}/resources
                               COMMAND ${CMAKE_CXX_COMPILER}
                                       -I${CMAKE_SOURCE_DIR}/oneapi -I${CMAKE_SOURCE_DIR}/../../../../include
                                       -Xshardware -Xstarget=${FPGA_DEVICE}
                                       -Xsffp-reassociate -Xsffp-contract=fast -Xsdsp-mode=prefer-dsp -Xsprofile -Xsclock=360MHz
                                       ${CMAKE_SOURCE_DIR}/oneapi/${KERNEL}_${SIZE}_${HW}.sycl.h
                                       -fsycl -fintelfpga -fsycl-link=image -o ${KERNEL}_${SIZE}_${HW}.image
                               COMMAND cp -rf ${KERNEL}_${SIZE}_${HW}.image ${CMAKE_SOURCE_DIR}/bin
                               COMMAND cp -rf ${KERNEL}_${SIZE}_${HW}.image.prj/reports/report.html    ${CMAKE_SOURCE_DIR}/generated_reports/${KERNEL}_${SIZE}_${HW}
                               COMMAND cp -rf ${KERNEL}_${SIZE}_${HW}.image.prj/reports/resources      ${CMAKE_SOURCE_DIR}/generated_reports/${KERNEL}_${SIZE}_${HW}
                               COMMAND cp -rf ${KERNEL}_${SIZE}_${HW}.image.prj/acl_quartus_report.txt ${CMAKE_SOURCE_DIR}/generated_reports/${KERNEL}_${SIZE}_${HW}
                               DEPENDS ${CMAKE_SOURCE_DIR}/oneapi/${KERNEL}_${SIZE}_${HW}.sycl.h # File-level dependency
                                       gen_oneapi_${KERNEL}_${SIZE}_${HW}                                 # Target-level dependency
                               WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                               COMMENT "Start to generate an image"
                               VERBATIM)
            add_custom_target(synthesize_${KERNEL}_${SIZE}_${HW} DEPENDS ${CMAKE_SOURCE_DIR}/bin/${KERNEL}_${SIZE}_${HW}.image)
        endforeach()
    endforeach()
endforeach()
