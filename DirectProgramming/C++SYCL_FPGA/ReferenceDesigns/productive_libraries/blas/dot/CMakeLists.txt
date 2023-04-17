project(dot_oneapi)
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
    add_custom_target(dot_generate_${LOWER_SIZE})
    foreach(DOT_TYPE "SDOT" "DDOT")
        string(TOLOWER ${DOT_TYPE} LOWER_DOT_TYPE)
        add_executable(dot_generate_${LOWER_SIZE}_${LOWER_DOT_TYPE} dot.cpp)
        target_include_directories(dot_generate_${LOWER_SIZE}_${LOWER_DOT_TYPE} PRIVATE "${TOOLS_PATH}/Halide/include")
        target_include_directories(dot_generate_${LOWER_SIZE}_${LOWER_DOT_TYPE} PRIVATE "${CMAKE_SOURCE_DIR}/include")
        target_compile_definitions(dot_generate_${LOWER_SIZE}_${LOWER_DOT_TYPE} PRIVATE "T2SP_${DOT_TYPE}" "${SIZE}" "${DEVICE_DEFINE}")
        target_link_libraries(dot_generate_${LOWER_SIZE}_${LOWER_DOT_TYPE} PRIVATE pthread z dl Halide)
        set_target_properties(dot_generate_${LOWER_SIZE}_${LOWER_DOT_TYPE} PROPERTIES CXX_STANDARD 11)
        add_custom_command(TARGET dot_generate_${LOWER_SIZE}_${LOWER_DOT_TYPE}
                           POST_BUILD
                           COMMAND env CLEARCODE=1 ${CMAKE_BINARY_DIR}/dot_generate_${LOWER_SIZE}_${LOWER_DOT_TYPE}
                                   && mv ${LOWER_DOT_TYPE}.sycl.h ${LOWER_DOT_TYPE}.${LOWER_SIZE}.sycl.h
                           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/include
                           COMMENT "${LOWER_DOT_TYPE} (${LOWER_SIZE} size): Start to generate ${LOWER_DOT_TYPE}.${LOWER_SIZE}.sycl.h"
                           VERBATIM)
        add_dependencies(dot_generate_${LOWER_SIZE} dot_generate_${LOWER_SIZE}_${LOWER_DOT_TYPE})
    endforeach()
endforeach()

###############################################################################
### FPGA Emulator
###############################################################################
add_executable(dot_test test.cpp)
target_include_directories(dot_test PRIVATE "${TOOLS_PATH}/Halide/include")
target_include_directories(dot_test PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_include_directories(dot_test PRIVATE "$ENV{MKLROOT}/include")
target_compile_definitions(dot_test PRIVATE TINY MKL_ILP64 T2SP_NDEBUG FPGA_EMULATOR)
target_link_libraries(dot_test mkl_sycl mkl_intel_ilp64 mkl_sequential mkl_core sycl OpenCL pthread m dl)
set_target_properties(dot_test PROPERTIES COMPILE_FLAGS "-fsycl -fintelfpga")
set_target_properties(dot_test PROPERTIES LINK_FLAGS "-fsycl -fintelfpga")
add_dependencies(dot_test dot_generate_tiny)

###############################################################################
### Generate Report
###############################################################################
add_executable(dot_report.a hardware_demo.cpp)
add_dependencies(dot_report.a dot_generate_tiny_sdot)
add_custom_target(dot_report DEPENDS dot_report.a)
target_include_directories(dot_report.a PRIVATE "${TOOLS_PATH}/Halide/include")
target_include_directories(dot_report.a PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_compile_definitions(dot_report.a PRIVATE "${DEVICE_DEFINE}" T2SP_SGEMM TINY)
set_target_properties(dot_report.a PROPERTIES COMPILE_FLAGS "-fsycl -fintelfpga")
set_target_properties(dot_report.a PROPERTIES LINK_FLAGS "-fsycl -fintelfpga -reuse-exe=${CMAKE_BINARY_DIR}/dot_fpga -Xshardware -Xstarget=${FPGA_DEVICE} -Xsffp-reassociate -Xsprofile -Xsclock=360MHz -Xsseed=5 -fsycl-link=early")

###############################################################################
### FPGA Hardware
###############################################################################
add_executable(dot_fpga hardware_demo.cpp)
add_dependencies(dot_fpga dot_generate_large_sdot)
target_include_directories(dot_fpga PRIVATE "${TOOLS_PATH}/Halide/include")
target_include_directories(dot_fpga PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_compile_definitions(dot_fpga PRIVATE "${DEVICE_DEFINE}" T2SP_SGEMM TINY)
set_target_properties(dot_fpga PROPERTIES COMPILE_FLAGS "-fsycl -fintelfpga")
set_target_properties(dot_fpga PROPERTIES LINK_FLAGS "-fsycl -fintelfpga -reuse-exe=${CMAKE_BINARY_DIR}/dot_fpga -Xshardware -Xstarget=${FPGA_DEVICE} -Xsffp-reassociate -Xsprofile -Xsclock=360MHz -Xsseed=5")
