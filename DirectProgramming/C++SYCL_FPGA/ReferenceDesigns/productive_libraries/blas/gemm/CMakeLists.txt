project(gemm_oneapi)
set(CMAKE_CXX_COMPILER icpx)

if(NOT DEFINED ENV{T2S_PATH})
    message(FATAL_ERROR "T2S_PATH is not set")
elseif(NOT DEFINED ENV{MKLROOT})
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

link_directories("$ENV{T2S_PATH}/Halide/lib")
link_directories("$ENV{MKLROOT}/lib/intel64")

###############################################################################
### Generate header files
###############################################################################
foreach(SIZE "LARGE" "TINY")
    string(TOLOWER ${SIZE} LOWER_SIZE)
    add_custom_target(gemm_generate_${LOWER_SIZE})
    foreach(GEMM_TYPE "SGEMM" "DGEMM" "CGEMM" "ZGEMM")
        string(TOLOWER ${GEMM_TYPE} LOWER_GEMM_TYPE)
        add_executable(gemm_generate_${LOWER_SIZE}_${LOWER_GEMM_TYPE} gemm.cpp)
        target_include_directories(gemm_generate_${LOWER_SIZE}_${LOWER_GEMM_TYPE} PRIVATE "$ENV{T2S_PATH}/Halide/include")
        target_include_directories(gemm_generate_${LOWER_SIZE}_${LOWER_GEMM_TYPE} PRIVATE "${CMAKE_SOURCE_DIR}/include")
        target_compile_definitions(gemm_generate_${LOWER_SIZE}_${LOWER_GEMM_TYPE} PRIVATE "T2SP_${GEMM_TYPE}" "${SIZE}" "${DEVICE_DEFINE}")
        target_link_libraries(gemm_generate_${LOWER_SIZE}_${LOWER_GEMM_TYPE} PRIVATE pthread z dl Halide)
        set_target_properties(gemm_generate_${LOWER_SIZE}_${LOWER_GEMM_TYPE} PROPERTIES CXX_STANDARD 11)
        add_custom_command(TARGET gemm_generate_${LOWER_SIZE}_${LOWER_GEMM_TYPE}
                           POST_BUILD
                           COMMAND env CLEARCODE=1 ${CMAKE_BINARY_DIR}/gemm_generate_${LOWER_SIZE}_${LOWER_GEMM_TYPE}
                           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/include
                           COMMENT "${LOWER_GEMM_TYPE} (${LOWER_SIZE} size): Start to generate ${LOWER_GEMM_TYPE}.sycl.h"
                           VERBATIM)
        add_dependencies(gemm_generate_${LOWER_SIZE} gemm_generate_${LOWER_SIZE}_${LOWER_GEMM_TYPE})
    endforeach()
endforeach()

###############################################################################
### FPGA Emulator
###############################################################################
add_custom_target(gemm_test)
foreach(N RANGE 0 4)
    add_executable(gemm_test_${N} test.cpp)
    target_include_directories(gemm_test_${N} PRIVATE "$ENV{T2S_PATH}/Halide/include")
    target_include_directories(gemm_test_${N} PRIVATE "${CMAKE_SOURCE_DIR}/include")
    target_include_directories(gemm_test_${N} PRIVATE "$ENV{MKLROOT}/include")
    target_compile_definitions(gemm_test_${N} PRIVATE TINY MKL_ILP64 T2SP_NDEBUG "T2SP_TEST_${N}" FPGA_EMULATOR)
    target_link_libraries(gemm_test_${N} mkl_sycl mkl_intel_ilp64 mkl_sequential mkl_core sycl OpenCL pthread m dl)
    set_target_properties(gemm_test_${N} PROPERTIES COMPILE_FLAGS "-fsycl -fintelfpga")
    set_target_properties(gemm_test_${N} PROPERTIES LINK_FLAGS "-fsycl -fintelfpga")
    add_dependencies(gemm_test_${N} gemm_generate_tiny)
    add_dependencies(gemm_test gemm_test_${N})
endforeach()

###############################################################################
### Generate Report
###############################################################################
add_executable(gemm_report.a hardware_demo.cpp)
add_dependencies(gemm_report.a gemm_generate_large)
add_custom_target(gemm_report DEPENDS gemm_report.a)
target_include_directories(gemm_report.a PRIVATE "$ENV{T2S_PATH}/Halide/include")
target_include_directories(gemm_report.a PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_compile_definitions(gemm_report.a PRIVATE "${DEVICE_DEFINE}")
set_target_properties(gemm_report.a PROPERTIES COMPILE_FLAGS "-fsycl -fintelfpga")
set_target_properties(gemm_report.a PROPERTIES LINK_FLAGS "-fsycl -fintelfpga -reuse-exe=${CMAKE_BINARY_DIR}/gemm_fpga -Xshardware -Xstarget=${FPGA_DEVICE} -Xsffp-reassociate -Xsprofile -Xsclock=360MHz -Xsseed=5 -fsycl-link=early")

###############################################################################
### FPGA Hardware
###############################################################################
add_executable(gemm_fpga hardware_demo.cpp)
add_dependencies(gemm_fpga gemm_generate_large)
target_include_directories(gemm_fpga PRIVATE "$ENV{T2S_PATH}/Halide/include")
target_include_directories(gemm_fpga PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_compile_definitions(gemm_fpga PRIVATE "${DEVICE_DEFINE}")
set_target_properties(gemm_fpga PROPERTIES COMPILE_FLAGS "-fsycl -fintelfpga")
set_target_properties(gemm_fpga PROPERTIES LINK_FLAGS "-fsycl -fintelfpga -reuse-exe=${CMAKE_BINARY_DIR}/gemm_fpga -Xshardware -Xstarget=${FPGA_DEVICE} -Xsffp-reassociate -Xsprofile -Xsclock=360MHz -Xsseed=5")
