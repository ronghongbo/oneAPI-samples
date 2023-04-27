project(gemm_oneapi)
set(CMAKE_CXX_COMPILER icpx)

# Due to restrictions of file size, we have split libHalide.a into parts. Join them into the original binary
if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/../../tools/Halide/lib/libHalide.a")
    execute_process(COMMAND cat libHalide.part.aa libHalide.part.ab libHalide.part.ac libHalide.part.ad libHalide.part.ae libHalide.part.af libHalide.part.ag  
                    OUTPUT_FILE libHalide.a
                    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../../tools/Halide/lib)
endif()
set(TOOLS_PATH "${CMAKE_CURRENT_LIST_DIR}/../../tools")

if(NOT DEFINED ENV{MKLROOT})
    message(FATAL_ERROR "MKLROOT is not set")
endif()

# FPGA board selection
if(NOT DEFINED FPGA_DEVICE)
    set(FPGA_DEVICE "intel_a10gx_pac:pac_a10")
    set(DEVICE_DEFINE "A10")
    message(STATUS "FPGA_DEVICE was not specified.\
                    \nConfiguring the design to run on the default FPGA board ${FPGA_DEVICE} (Intel(R) PAC with Intel Arria(R) 10 GX FPGA). \
                    \nPlease refer to the README for information on board selection.")
else()
    if("${FPGA_DEVICE}" STREQUAL "intel_s10sx_pac:pac_s10")
        set(DEVICE_DEFINE "S10")
    elseif("${FPGA_DEVICE}" STREQUAL "intel_a10gx_pac:pac_a10")
        set(DEVICE_DEFINE "A10")
    else()
        message(FATAL_ERROR "Unsupported device ${FPGA_DEVICE}")
    endif()
    message(STATUS "Configuring the design to run on FPGA board ${FPGA_DEVICE}")
endif()

link_directories("${TOOLS_PATH}/Halide/lib")
link_directories("$ENV{MKLROOT}/lib/intel64")

###############################################################################
### Generate header files
###############################################################################
foreach(SIZE "LARGE" "TINY")
    string(TOLOWER ${SIZE} LOWER_SIZE)
    add_custom_target(generate_${LOWER_SIZE})
    foreach(GEMM_TYPE "SGEMM" "DGEMM" "CGEMM" "ZGEMM")
        string(TOLOWER ${GEMM_TYPE} LOWER_GEMM_TYPE)
        add_executable(generate_${LOWER_SIZE}_${LOWER_GEMM_TYPE} gemm.cpp)
        target_include_directories(generate_${LOWER_SIZE}_${LOWER_GEMM_TYPE} PRIVATE "${TOOLS_PATH}/Halide/include")
        target_include_directories(generate_${LOWER_SIZE}_${LOWER_GEMM_TYPE} PRIVATE "${CMAKE_SOURCE_DIR}/include")
        target_compile_definitions(generate_${LOWER_SIZE}_${LOWER_GEMM_TYPE} PRIVATE "T2SP_${GEMM_TYPE}" "${SIZE}" "${DEVICE_DEFINE}")
        target_link_libraries(generate_${LOWER_SIZE}_${LOWER_GEMM_TYPE} PRIVATE pthread z dl Halide)
        set_target_properties(generate_${LOWER_SIZE}_${LOWER_GEMM_TYPE} PROPERTIES CXX_STANDARD 11)
        add_custom_command(TARGET generate_${LOWER_SIZE}_${LOWER_GEMM_TYPE}
                           POST_BUILD
                           COMMAND env CLEARCODE=1 ${CMAKE_BINARY_DIR}/generate_${LOWER_SIZE}_${LOWER_GEMM_TYPE}
                           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/include
                           COMMENT "${LOWER_GEMM_TYPE} (${LOWER_SIZE} size): Start to generate ${LOWER_GEMM_TYPE}.sycl.h"
                           VERBATIM)
        add_dependencies(generate_${LOWER_SIZE} generate_${LOWER_SIZE}_${LOWER_GEMM_TYPE})
    endforeach()
endforeach()

###############################################################################
### FPGA Emulator
###############################################################################
add_custom_target(test)
foreach(N RANGE 0 4)
    add_executable(test_${N} test.cpp)
    target_include_directories(test_${N} PRIVATE "${TOOLS_PATH}/Halide/include")
    target_include_directories(test_${N} PRIVATE "${CMAKE_SOURCE_DIR}/include")
    target_include_directories(test_${N} PRIVATE "$ENV{MKLROOT}/include")
    target_compile_definitions(test_${N} PRIVATE TINY MKL_ILP64 T2SP_NDEBUG "T2SP_TEST_${N}" FPGA_EMULATOR)
    target_link_libraries(test_${N} mkl_sycl mkl_intel_ilp64 mkl_sequential mkl_core sycl OpenCL pthread m dl)
    set_target_properties(test_${N} PROPERTIES COMPILE_FLAGS "-fsycl -fintelfpga")
    set_target_properties(test_${N} PROPERTIES LINK_FLAGS "-fsycl -fintelfpga")
    add_dependencies(test_${N} generate_tiny_sgemm)
    add_dependencies(test test_${N})
endforeach()

###############################################################################
### Generate Report
###############################################################################
add_executable(report.a hardware_demo.cpp)
add_dependencies(report.a generate_tiny_sgemm)
add_custom_target(report DEPENDS report.a)
target_include_directories(report.a PRIVATE "${TOOLS_PATH}/Halide/include")
target_include_directories(report.a PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_compile_definitions(report.a PRIVATE "${DEVICE_DEFINE}" T2SP_SGEMM TINY)
set_target_properties(report.a PROPERTIES COMPILE_FLAGS "-fsycl -fintelfpga")
set_target_properties(report.a PROPERTIES LINK_FLAGS "-fsycl -fintelfpga -reuse-exe=${CMAKE_BINARY_DIR}/fpga -Xshardware -Xstarget=${FPGA_DEVICE} -Xsffp-reassociate -Xsffp-contract=fast -Xsprofile -Xsclock=360MHz -fsycl-link=early")

###############################################################################
### FPGA Hardware
###############################################################################
add_executable(fpga hardware_demo.cpp)
add_dependencies(fpga generate_tiny_sgemm)
target_include_directories(fpga PRIVATE "${TOOLS_PATH}/Halide/include")
target_include_directories(fpga PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_compile_definitions(fpga PRIVATE "${DEVICE_DEFINE}" T2SP_SGEMM TINY)
set_target_properties(fpga PROPERTIES COMPILE_FLAGS "-fsycl -fintelfpga")
set_target_properties(fpga PROPERTIES LINK_FLAGS "-fsycl -fintelfpga -reuse-exe=${CMAKE_BINARY_DIR}/fpga -Xshardware -Xstarget=${FPGA_DEVICE} -Xsffp-reassociate -Xsffp-contract=fast -Xsprofile -Xsclock=360MHz")
