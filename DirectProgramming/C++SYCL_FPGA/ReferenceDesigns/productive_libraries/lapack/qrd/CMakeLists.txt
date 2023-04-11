project(qrd_oneapi)
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
add_custom_target(t2s_qrd_generate)
add_executable(qrd_generate_float qrd.cpp)
target_include_directories(qrd_generate_float PRIVATE "${TOOLS_PATH}/Halide/include")
target_link_libraries(qrd_generate_float PRIVATE pthread z dl Halide)
set_target_properties(qrd_generate_float PROPERTIES CXX_STANDARD 11)
add_custom_command(TARGET qrd_generate_float
                    POST_BUILD
                    COMMAND env CLEARCODE=1 ${CMAKE_BINARY_DIR}/qrd_generate_float
                    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                    COMMENT "Start to generate qrd.sycl.h"
                    VERBATIM)
add_dependencies(t2s_qrd_generate qrd_generate_float)

###############################################################################
### FPGA Emulator
###############################################################################
add_executable(qrd_test qrd-run-fpga.cpp)
target_include_directories(qrd_test PRIVATE "${TOOLS_PATH}/Halide/include")
target_compile_definitions(qrd_test PRIVATE MKL_ILP64 T2SP_NDEBUG FPGA_EMULATOR)
target_link_libraries(qrd_test mkl_sycl mkl_intel_ilp64 mkl_sequential mkl_core sycl OpenCL pthread m dl)
set_target_properties(qrd_test PROPERTIES COMPILE_FLAGS "-fsycl -fintelfpga")
set_target_properties(qrd_test PROPERTIES LINK_FLAGS "-fsycl -fintelfpga")
add_dependencies(qrd_test t2s_qrd_generate)

###############################################################################
### Generate Report
###############################################################################
add_executable(qrd_report.a qrd-run-fpga.cpp)
add_dependencies(qrd_report.a qrd_generate_large)
add_custom_target(qrd_report DEPENDS qrd_report.a)
target_include_directories(qrd_report.a PRIVATE "${TOOLS_PATH}/Halide/include")
target_compile_definitions(qrd_report.a PRIVATE "${DEVICE_DEFINE}")
set_target_properties(qrd_report.a PROPERTIES COMPILE_FLAGS "-fsycl -fintelfpga")
set_target_properties(qrd_report.a PROPERTIES LINK_FLAGS "-fsycl -fintelfpga -reuse-exe=${CMAKE_BINARY_DIR}/qrd_fpga -Xshardware -Xstarget=${FPGA_DEVICE} -Xsffp-reassociate -Xsprofile -Xsclock=360MHz -Xsseed=5 -fsycl-link=early")

###############################################################################
### FPGA Hardware
###############################################################################
add_executable(qrd_fpga qrd-run-fpga.cpp)
add_dependencies(qrd_fpga qrd_generate_large)
target_include_directories(qrd_fpga PRIVATE "${TOOLS_PATH}/Halide/include")
target_compile_definitions(qrd_fpga PRIVATE "${DEVICE_DEFINE}")
set_target_properties(qrd_fpga PROPERTIES COMPILE_FLAGS "-fsycl -fintelfpga")
set_target_properties(qrd_fpga PROPERTIES LINK_FLAGS "-fsycl -fintelfpga -reuse-exe=${CMAKE_BINARY_DIR}/qrd_fpga -Xshardware -Xstarget=${FPGA_DEVICE} -Xsffp-reassociate -Xsprofile -Xsclock=360MHz -Xsseed=5")
