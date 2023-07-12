cmake_minimum_required(VERSION 3.10)

set(TYPES "s" "d" "c" "z" "sds")
set(CMAKE_CXX_COMPILER icpx)
set(OPENCL_1_2_1_BSP /opt/a10/inteldevstack/a10_gx_pac_ias_1_2_1_pv/opencl/opencl_bsp)

get_filename_component(HALIDE_TOOLS_PATH          "${RECONFIGURABLE_DOTPROD_PATH}/../../tools/Halide" ABSOLUTE)
get_filename_component(CPP_SYCL_FPGA_PATH         "${RECONFIGURABLE_DOTPROD_PATH}/../../../.." ABSOLUTE)
get_filename_component(PRODUCTIVE_LIBRARIES_PATH  "${RECONFIGURABLE_DOTPROD_PATH}/../.."       ABSOLUTE)
get_filename_component(GOOGLE_TEST_PATH           "${RECONFIGURABLE_DOTPROD_PATH}/../../test"  ABSOLUTE)

# Due to restrictions of file size, we have split libHalide.a into parts. Join them into the original binary
if(NOT EXISTS "${HALIDE_TOOLS_PATH}/lib/libHalide.a")
    execute_process(COMMAND cat libHalide.part.aa libHalide.part.ab libHalide.part.ac libHalide.part.ad libHalide.part.ae libHalide.part.af libHalide.part.ag
                    OUTPUT_FILE libHalide.a
                    WORKING_DIRECTORY ${HALIDE_TOOLS_PATH}/lib)
endif()

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
        message(FATYPEAL_ERROR "Unsupported device ${FPGA_DEVICE}")
    endif()
    message(STYPEATUS "Configuring the design to run on FPGA board ${FPGA_DEVICE}")
endif()

if(NOT DEFINED ENV{MKLROOT})
    message(FATAL_ERROR "MKLROOT is not set")
endif()

# Seed for synthesis
if(NOT DEFINED SEED)
    set(SEED 1)
endif()

###############################################################################
### Create results directories
###############################################################################
file(MAKE_DIRECTORY ${RECONFIGURABLE_DOTPROD_PATH}/oneapi)
file(MAKE_DIRECTORY ${RECONFIGURABLE_DOTPROD_PATH}/bin)
file(MAKE_DIRECTORY ${RECONFIGURABLE_DOTPROD_PATH}/reports)

foreach(TYPEA ${TYPES})
    foreach(SIZE "tiny" "large")
        foreach(HW "a10" "s10")
            string(TOUPPER ${SIZE}   UPPER_SIZE)
            string(TOUPPER ${HW}     UPPER_HW)
            string(TOUPPER ${TYPEA}  UPPER_TYPE)
    
            # Generate OneAPI files
            add_custom_command(OUTPUT  ${RECONFIGURABLE_DOTPROD_PATH}/oneapi/${TYPEA}dotprod_${SIZE}_${HW}.cpp
                               COMMAND ${CMAKE_CXX_COMPILER} 
                                       -DKERNEL="t2sp::blas::row_major::${TYPEA}dotprod::${TYPEA}dotprod"
                                       -DOUTPUT_FILE="${TYPEA}dotprod_${SIZE}_${HW}.cpp"
                                       -I${HALIDE_TOOLS_PATH}/include
                                       -std=gnu++11 -O0 -g
                                       -Wno-tautological-constant-compare
                                       ${RECONFIGURABLE_DOTPROD_PATH}/spec.cpp
                                       -L${HALIDE_TOOLS_PATH}/lib -lpthread -lz -ldl -lHalide
                                       -D${UPPER_SIZE} -DT2SP_${UPPER_TYPE}DOTPROD
                                       -o ${RECONFIGURABLE_DOTPROD_PATH}/bin/oneapi_${TYPEA}dotprod_${SIZE}_${HW}
                               COMMAND env CLEARCODE=1 ${RECONFIGURABLE_DOTPROD_PATH}/bin/oneapi_${TYPEA}dotprod_${SIZE}_${HW}
                               COMMAND mv ${RECONFIGURABLE_DOTPROD_PATH}/bin/${TYPEA}dotprod_${SIZE}_${HW}.cpp ${RECONFIGURABLE_DOTPROD_PATH}/oneapi/
                               DEPENDS ${RECONFIGURABLE_DOTPROD_PATH}/spec.cpp
                                       ${RECONFIGURABLE_DOTPROD_PATH}/parameters.h
                                       ${HALIDE_TOOLS_PATH}/include/Halide.h
                               WORKING_DIRECTORY ${RECONFIGURABLE_DOTPROD_PATH}/bin
                               COMMENT "Generating OneAPI files"
                               VERBATIM)
            add_custom_target(oneapi_${TYPEA}dotprod_${SIZE}_${HW} DEPENDS ${RECONFIGURABLE_DOTPROD_PATH}/oneapi/${TYPEA}dotprod_${SIZE}_${HW}.cpp)
    
            # Generate a report
            file(MAKE_DIRECTORY ${RECONFIGURABLE_DOTPROD_PATH}/reports/${TYPEA}dotprod_${SIZE}_${HW})
            add_custom_command(OUTPUT  ${RECONFIGURABLE_DOTPROD_PATH}/reports/${TYPEA}dotprod_${SIZE}_${HW}/report.html
                                       ${RECONFIGURABLE_DOTPROD_PATH}/reports/${TYPEA}dotprod_${SIZE}_${HW}/resources
                               COMMAND ${CMAKE_CXX_COMPILER}
                                       -I${RECONFIGURABLE_DOTPROD_PATH}/oneapi -I${CPP_SYCL_FPGA_PATH}/include -I${PRODUCTIVE_LIBRARIES_PATH}/include -I${HALIDE_TOOLS_PATH}/include
                                       ${RECONFIGURABLE_DOTPROD_PATH}/oneapi/${TYPEA}dotprod_${SIZE}_${HW}.cpp
                                       ${PRODUCTIVE_LIBRARIES_PATH}/include/halide_runtime_etc.cpp
                                       -Xshardware -Xstarget=${FPGA_DEVICE}
                                       -Xsffp-reassociate -Xsffp-contract=fast -Xsdsp-mode=prefer-dsp -Xsprofile -Xsclock=360MHz
                                       -fsycl -fintelfpga -fsycl-link=early -o ${TYPEA}dotprod_${SIZE}_${HW}.early
                               COMMAND cp -rf ${TYPEA}dotprod_${SIZE}_${HW}.early.prj/reports/report.html ${RECONFIGURABLE_DOTPROD_PATH}/reports/${TYPEA}dotprod_${SIZE}_${HW}
                               COMMAND cp -rf ${TYPEA}dotprod_${SIZE}_${HW}.early.prj/reports/resources   ${RECONFIGURABLE_DOTPROD_PATH}/reports/${TYPEA}dotprod_${SIZE}_${HW}
                               DEPENDS ${PRODUCTIVE_LIBRARIES_PATH}/include/halide_runtime_etc.cpp
                                       ${PRODUCTIVE_LIBRARIES_PATH}/include/halide_runtime_etc.hpp
                                       ${PRODUCTIVE_LIBRARIES_PATH}/include/complex_helper.hpp
                                       ${PRODUCTIVE_LIBRARIES_PATH}/include/pipe_wrapper.hpp
                                       oneapi_${TYPEA}dotprod_${SIZE}_${HW}
                               WORKING_DIRECTORY ${RECONFIGURABLE_DOTPROD_PATH}/bin
                               COMMENT "Generating a report"
                               VERBATIM)
            add_custom_target(report_${TYPEA}dotprod_${SIZE}_${HW} DEPENDS ${RECONFIGURABLE_DOTPROD_PATH}/reports/${TYPEA}dotprod_${SIZE}_${HW}/report.html
                                                                                                  ${RECONFIGURABLE_DOTPROD_PATH}/reports/${TYPEA}dotprod_${SIZE}_${HW}/resources)
    
            # Generate an image
            add_custom_command(OUTPUT  ${RECONFIGURABLE_DOTPROD_PATH}/bin/${TYPEA}dotprod_${SIZE}_${HW}.a
                                       ${RECONFIGURABLE_DOTPROD_PATH}/reports/${TYPEA}dotprod_${SIZE}_${HW}/acl_quartus_report.txt
                               COMMAND ${CMAKE_CXX_COMPILER}
                                       -I${RECONFIGURABLE_DOTPROD_PATH}/oneapi -I${CPP_SYCL_FPGA_PATH}/include -I${PRODUCTIVE_LIBRARIES_PATH}/include -I${HALIDE_TOOLS_PATH}/include
                                       ${RECONFIGURABLE_DOTPROD_PATH}/oneapi/${TYPEA}dotprod_${SIZE}_${HW}.cpp
                                       ${PRODUCTIVE_LIBRARIES_PATH}/include/halide_runtime_etc.cpp
                                       -Xshardware -Xstarget=${FPGA_DEVICE}
                                       -Xsffp-reassociate -Xsffp-contract=fast -Xsdsp-mode=prefer-dsp -Xsprofile -Xsclock=360MHz -Xsseed=${SEED}
                                       -fsycl -fintelfpga -fsycl-link=image -o ${TYPEA}dotprod_${SIZE}_${HW}.a
                               COMMAND cp -rf ${TYPEA}dotprod_${SIZE}_${HW}.a                          ${RECONFIGURABLE_DOTPROD_PATH}/bin
                               COMMAND cp -rf ${TYPEA}dotprod_${SIZE}_${HW}.prj/reports/report.html    ${RECONFIGURABLE_DOTPROD_PATH}/reports/${TYPEA}dotprod_${SIZE}_${HW}
                               COMMAND cp -rf ${TYPEA}dotprod_${SIZE}_${HW}.prj/reports/resources      ${RECONFIGURABLE_DOTPROD_PATH}/reports/${TYPEA}dotprod_${SIZE}_${HW}
                               COMMAND cp -rf ${TYPEA}dotprod_${SIZE}_${HW}.prj/acl_quartus_report.txt ${RECONFIGURABLE_DOTPROD_PATH}/reports/${TYPEA}dotprod_${SIZE}_${HW}
                               DEPENDS ${PRODUCTIVE_LIBRARIES_PATH}/include/halide_runtime_etc.cpp
                                       ${PRODUCTIVE_LIBRARIES_PATH}/include/halide_runtime_etc.hpp
                                       ${PRODUCTIVE_LIBRARIES_PATH}/include/complex_helper.hpp
                                       ${PRODUCTIVE_LIBRARIES_PATH}/include/pipe_wrapper.hpp
                                       oneapi_${TYPEA}dotprod_${SIZE}_${HW}
                               WORKING_DIRECTORY ${RECONFIGURABLE_DOTPROD_PATH}/bin
                               COMMENT "Generating an image"
                               VERBATIM)
            add_custom_target(synthesize_${TYPEA}dotprod_${SIZE}_${HW} DEPENDS ${RECONFIGURABLE_DOTPROD_PATH}/bin/${TYPEA}dotprod_${SIZE}_${HW}.a)
        endforeach()
    endforeach()
endforeach()

# Generate correctness tests for the current kernel. They assume a tiny systolic array on an FPGA emulator.
add_custom_target(oneapi_targets_for_tests)
foreach(PRECISION ${PRECISIONS})
    add_dependencies(oneapi_targets_for_tests oneapi_${PRECISION}dotprod_tiny_a10)
endforeach()

foreach(PRECISION ${PRECISIONS})
    set(GENERATED_ONEAPI_SOURCE_FILES_FOR_TESTS ${GENERATED_ONEAPI_SOURCE_FILES_FOR_TESTS} ${RECONFIGURABLE_DOTPROD_PATH}/oneapi/${PRECISION}dotprod_tiny_a10.cpp)
endforeach()

file(MAKE_DIRECTORY ${CURRENT_KERNEL_PATH}/bin)
add_custom_command(OUTPUT  ${CURRENT_KERNEL_PATH}/bin/test
                   COMMAND ${CMAKE_CXX_COMPILER}
                           -I${HALIDE_TOOLS_PATH}/include -I$ENV{MKLROOT}/include -I${PRODUCTIVE_LIBRARIES_PATH}/include -I${CPP_SYCL_FPGA_PATH}/include
                           -I${GOOGLE_TEST_PATH}/include -I${GOOGLE_TEST_PATH}/googletest/include
                           -DTINY -DMKL_ILP64 -DT2SP_NDEBUG -DFPGA_EMULATOR
                           -fsycl -fintelfpga
                           -Wno-tautological-constant-compare
                           ${CURRENT_KERNEL_PATH}/test.cpp
                           ${PRODUCTIVE_LIBRARIES_PATH}/include/halide_runtime_etc.cpp
                           ${GENERATED_ONEAPI_SOURCE_FILES_FOR_TESTS}
                           -L${HALIDE_TOOLS_PATH}/lib -L$ENV{MKLROOT}/lib/intel64
                           -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lsycl -lOpenCL -lpthread -lm -ldl -lz
                           -L${GOOGLE_TEST_PATH}/googletest/lib
                           -lgtest -lgtest_main
                           -lHalide
                           -o ${CURRENT_KERNEL_PATH}/bin/test
                   DEPENDS oneapi_targets_for_tests
                           ${CURRENT_KERNEL_PATH}/test.cpp
                           ${CURRENT_KERNEL_PATH}/api.hpp
                           ${RECONFIGURABLE_DOTPROD_PATH}/api.hpp
                           ${PRODUCTIVE_LIBRARIES_PATH}/include/halide_runtime_etc.cpp
                           ${PRODUCTIVE_LIBRARIES_PATH}/include/halide_runtime_etc.hpp
                           ${PRODUCTIVE_LIBRARIES_PATH}/include/complex_helper.hpp
                           ${PRODUCTIVE_LIBRARIES_PATH}/include/pipe_wrapper.hpp
                           ${HALIDE_TOOLS_PATH}/include/Halide.h
                           ${HALIDE_TOOLS_PATH}/include/HalideRuntime.h
                           ${HALIDE_TOOLS_PATH}/include/HalideBuffer.h
                   WORKING_DIRECTORY ${CURRENT_KERNEL_PATH}/bin
                   COMMENT "Generate tests"
                   VERBATIM)
add_custom_target(tests DEPENDS ${CURRENT_KERNEL_PATH}/bin/test)

# Create targets for demonstrating how to run on real hardware
foreach(PRECISION ${PRECISIONS})
    foreach(SIZE "tiny" "large")
        string(TOUPPER ${SIZE} UPPER_SIZE)
        foreach(HW "a10" "s10")
            string(TOUPPER ${HW} UPPER_HW)
            add_custom_command(OUTPUT  ${CURRENT_KERNEL_PATH}/bin/demo_${PRECISION}${KERNEL}_${SIZE}_${HW}.o
                               COMMAND ${CMAKE_CXX_COMPILER}
                                       -I${HALIDE_TOOLS_PATH}/include -I$ENV{MKLROOT}/include -I${PRODUCTIVE_LIBRARIES_PATH}/include
                                       -I${GOOGLE_TEST_PATH}/include -I${GOOGLE_TEST_PATH}/googletest/include
                                       -D${UPPER_SIZE} -D${UPPER_HW}
                                       -DMKL_ILP64 -DT2SP_NDEBUG
                                       -fsycl -fintelfpga
                                       -Wno-tautological-constant-compare
                                       -c ${CURRENT_KERNEL_PATH}/demo.cpp
                                       -o ${CURRENT_KERNEL_PATH}/bin/demo_${PRECISION}${KERNEL}_${SIZE}_${HW}.o
                               DEPENDS ${CURRENT_KERNEL_PATH}/demo.cpp
                                       ${CURRENT_KERNEL_PATH}/api.hpp
                                       ${RECONFIGURABLE_DOTPROD_PATH}/api.hpp
                                       ${HALIDE_TOOLS_PATH}/include/Halide.h
                                       ${HALIDE_TOOLS_PATH}/include/HalideRuntime.h
                                       ${HALIDE_TOOLS_PATH}/include/HalideBuffer.h
                               WORKING_DIRECTORY ${CURRENT_KERNEL_PATH}/bin
                               COMMENT "Generating a demo application on ${WHAT}"
                               VERBATIM)
            add_custom_command(OUTPUT  ${CURRENT_KERNEL_PATH}/bin/demo_${PRECISION}${KERNEL}_${SIZE}_${HW}
                               COMMAND ${CMAKE_CXX_COMPILER}
                                       -fsycl -fintelfpga
                                       ${CURRENT_KERNEL_PATH}/bin/demo_${PRECISION}${KERNEL}_${SIZE}_${HW}.o
                                       ${RECONFIGURABLE_DOTPROD_PATH}/bin/${PRECISION}dotprod_${SIZE}_${HW}.a
                                       ${PRODUCTIVE_LIBRARIES_PATH}/include/halide_runtime_etc.cpp
                                       -L${HALIDE_TOOLS_PATH}/lib -lHalide -lpthread -lz
                                       -Wno-tautological-constant-compare
                                       -o ${CURRENT_KERNEL_PATH}/bin/demo_${PRECISION}${KERNEL}_${SIZE}_${HW}
                               DEPENDS ${CURRENT_KERNEL_PATH}/bin/demo_${PRECISION}${KERNEL}_${SIZE}_${HW}.o
                                       ${PRODUCTIVE_LIBRARIES_PATH}/include/halide_runtime_etc.cpp
                                       ${PRODUCTIVE_LIBRARIES_PATH}/include/halide_runtime_etc.hpp
                                       ${PRODUCTIVE_LIBRARIES_PATH}/include/complex_helper.hpp
                                       ${PRODUCTIVE_LIBRARIES_PATH}/include/pipe_wrapper.hpp
                                       synthesize_${PRECISION}dotprod_${SIZE}_${HW}
                               WORKING_DIRECTORY ${CURRENT_KERNEL_PATH}/bin
                               COMMENT "Linking the demo application with the device image"
                               VERBATIM)
            add_custom_target(demo_${PRECISION}${KERNEL}_${SIZE}_${HW} DEPENDS ${CURRENT_KERNEL_PATH}/bin/demo_${PRECISION}${KERNEL}_${SIZE}_${HW})
        endforeach()
    endforeach()
endforeach()

# Create targets for demonstrating how to run on FPGA emulator (for correctness debugging only).
foreach(PRECISION ${PRECISIONS})
    foreach(SIZE "tiny" "large")
        string(TOUPPER ${SIZE} UPPER_SIZE)
        foreach(HW "a10" "s10")
            string(TOUPPER ${HW} UPPER_HW)
            add_custom_command(OUTPUT  ${CURRENT_KERNEL_PATH}/bin/demo_${PRECISION}${KERNEL}_${SIZE}_${HW}_emu
                               COMMAND ${CMAKE_CXX_COMPILER}
                                       -I${HALIDE_TOOLS_PATH}/include -I$ENV{MKLROOT}/include -I${PRODUCTIVE_LIBRARIES_PATH}/include -I${CPP_SYCL_FPGA_PATH}/include
                                       -I${GOOGLE_TEST_PATH}/include -I${GOOGLE_TEST_PATH}/googletest/include
                                       -D${UPPER_SIZE} -D${UPPER_HW}
                                       -DMKL_ILP64 -DT2SP_NDEBUG
                                       -DFPGA_EMULATOR -DCHECK_CORRECTNESS
                                       -fsycl -fintelfpga
                                       -Wno-tautological-constant-compare
                                       ${CURRENT_KERNEL_PATH}/demo.cpp
                                       ${RECONFIGURABLE_DOTPROD_PATH}/oneapi/${PRECISION}dotprod_${SIZE}_${HW}.cpp
                                       ${PRODUCTIVE_LIBRARIES_PATH}/include/halide_runtime_etc.cpp
                                       -L${HALIDE_TOOLS_PATH}/lib -L$ENV{MKLROOT}/lib/intel64
                                       -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lsycl -lOpenCL -lpthread -lm -ldl -lz
                                       -lHalide
                                       -o ${CURRENT_KERNEL_PATH}/bin/demo_${PRECISION}${KERNEL}_${SIZE}_${HW}_emu
                               DEPENDS ${CURRENT_KERNEL_PATH}/demo.cpp
                               ${RECONFIGURABLE_DOTPROD_PATH}/oneapi/${PRECISION}dotprod_${SIZE}_${HW}.cpp
                                       ${CURRENT_KERNEL_PATH}/api.hpp
                                       ${RECONFIGURABLE_DOTPROD_PATH}/api.hpp
                                       ${HALIDE_TOOLS_PATH}/include/Halide.h
                                       ${HALIDE_TOOLS_PATH}/include/HalideRuntime.h
                                       ${HALIDE_TOOLS_PATH}/include/HalideBuffer.h
                                       ${PRODUCTIVE_LIBRARIES_PATH}/include/halide_runtime_etc.cpp
                                       ${PRODUCTIVE_LIBRARIES_PATH}/include/halide_runtime_etc.hpp
                                       ${PRODUCTIVE_LIBRARIES_PATH}/include/complex_helper.hpp
                                       ${PRODUCTIVE_LIBRARIES_PATH}/include/pipe_wrapper.hpp
                                       oneapi_${PRECISION}dotprod_${SIZE}_${HW}
                               WORKING_DIRECTORY ${CURRENT_KERNEL_PATH}/bin
                               COMMENT "Generating a demo application on FPGA emulator"
                               VERBATIM)
            add_custom_target(demo_${PRECISION}${KERNEL}_${SIZE}_${HW}_emu DEPENDS ${CURRENT_KERNEL_PATH}/bin/demo_${PRECISION}${KERNEL}_${SIZE}_${HW}_emu)
        endforeach()
    endforeach()
endforeach()
