#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "halide_runtime_etc.hpp"
#include "pipe_wrapper.hpp"
#include "complex_helper.hpp"
#include "constexpr_math.hpp"
#include "tuple.hpp"
#include "unrolled_loop.hpp"

template <typename... Args>
void log(Args &&...args) {
#ifndef T2SP_NDEBUG
  ((std::cout << "[INFO] ") << ... << args) << "\n";
#endif
}

using namespace sycl;
namespace t2sp::blas::row_major::cccsmatmul {

typedef union {
bool __attribute__ ((aligned(2))) s[2];
struct {bool s0,  s1;};
} bool2;
typedef struct {
	bool f0;
	bool f1;
	bool f2;
	bool f3;
} cgs;
using DC_channel = pipe_wrapper<class DC_channel_pipe, complexf2, 256>;
using DA_channel = pipe_wrapper<class DA_channel_pipe, complexf2, 256>;
using SA_channel_array_t = fpga_tools::NTuple<complexf2, 2>;
using SA_channel = pipe_wrapper<class SA_channel_pipe, SA_channel_array_t, 256>;
using DB_channel = pipe_wrapper<class DB_channel_pipe, complexf2, 256>;
using SB_channel_array_t = fpga_tools::NTuple<complexf2, 2>;
using SB_channel = pipe_wrapper<class SB_channel_pipe, SB_channel_array_t, 256>;
using SignalGenerator_channel = pipe_wrapper<class SignalGenerator_channel_pipe, cgs, 256>;
using Product_channel = pipe_wrapper<class Product_channel_pipe, complexf2, 256>;
using Out_channel = pipe_wrapper<class Out_channel_pipe, complexf2, 256>;
sycl::event cccsmatmul(sycl::queue &q_device, struct halide_buffer_t *A_buffer, bool TransposeA, bool ConjugateA, bool SymmetricA, bool HermitianA, bool UpA, struct halide_buffer_t *B_buffer, bool TransposeB, bool ConjugateB, bool SymmetricB, bool HermitianB, bool UpB, struct halide_buffer_t *C_buffer, bool SymmetricC, bool HermitianC, bool UpC, bool HalfSpaceOut, float alpha, float beta, struct halide_buffer_t *Output_buffer) {
  std::vector<sycl::event> oneapi_kernel_events{};
  std::vector<size_t> kernels_used_to_measure_time{};
  auto exception_handler = [](sycl::exception_list exceptions) {
    for (std::exception_ptr const &e : exceptions) {
      try {
        std::rethrow_exception(e);
      } catch (sycl::exception const &e) {
        std::cout << "Caught asynchronous SYCL exception:\n"
                  << e.what() << std::endl;
      }
    }
  };
  log("creating device queues");
  sycl::queue q_host(sycl::cpu_selector_v, exception_handler, sycl::property::queue::enable_profiling());
  log("Host: ", q_host.get_device().get_info<sycl::info::device::name>());
  log("Device: ", q_device.get_device().get_info<sycl::info::device::name>());
  sycl::device dev = q_device.get_device();
  void * const _ucon = nullptr;
  void * A = _halide_buffer_get_host(A_buffer);
  uint32_t A_type = _halide_buffer_get_type(A_buffer);
  int32_t A_dimensions = _halide_buffer_get_dimensions(A_buffer);
  int32_t A_min_0 = _halide_buffer_get_min(A_buffer, 0);
  int32_t A_extent_0 = _halide_buffer_get_extent(A_buffer, 0);
  int32_t A_stride_0 = _halide_buffer_get_stride(A_buffer, 0);
  int32_t A_min_1 = _halide_buffer_get_min(A_buffer, 1);
  int32_t A_extent_1 = _halide_buffer_get_extent(A_buffer, 1);
  int32_t A_stride_1 = _halide_buffer_get_stride(A_buffer, 1);
  void * B = _halide_buffer_get_host(B_buffer);
  uint32_t B_type = _halide_buffer_get_type(B_buffer);
  int32_t B_dimensions = _halide_buffer_get_dimensions(B_buffer);
  int32_t B_min_0 = _halide_buffer_get_min(B_buffer, 0);
  int32_t B_extent_0 = _halide_buffer_get_extent(B_buffer, 0);
  int32_t B_stride_0 = _halide_buffer_get_stride(B_buffer, 0);
  int32_t B_min_1 = _halide_buffer_get_min(B_buffer, 1);
  int32_t B_extent_1 = _halide_buffer_get_extent(B_buffer, 1);
  int32_t B_stride_1 = _halide_buffer_get_stride(B_buffer, 1);
  void * C = _halide_buffer_get_host(C_buffer);
  uint32_t C_type = _halide_buffer_get_type(C_buffer);
  int32_t C_dimensions = _halide_buffer_get_dimensions(C_buffer);
  int32_t C_min_0 = _halide_buffer_get_min(C_buffer, 0);
  int32_t C_extent_0 = _halide_buffer_get_extent(C_buffer, 0);
  int32_t C_stride_0 = _halide_buffer_get_stride(C_buffer, 0);
  int32_t C_min_1 = _halide_buffer_get_min(C_buffer, 1);
  int32_t C_extent_1 = _halide_buffer_get_extent(C_buffer, 1);
  int32_t C_stride_1 = _halide_buffer_get_stride(C_buffer, 1);
  void * Output = _halide_buffer_get_host(Output_buffer);
  uint32_t Output_type = _halide_buffer_get_type(Output_buffer);
  int32_t Output_dimensions = _halide_buffer_get_dimensions(Output_buffer);
  int32_t Output_min_0 = _halide_buffer_get_min(Output_buffer, 0);
  int32_t Output_extent_0 = _halide_buffer_get_extent(Output_buffer, 0);
  int32_t Output_stride_0 = _halide_buffer_get_stride(Output_buffer, 0);
  int32_t Output_min_1 = _halide_buffer_get_min(Output_buffer, 1);
  int32_t Output_extent_1 = _halide_buffer_get_extent(Output_buffer, 1);
  int32_t Output_stride_1 = _halide_buffer_get_stride(Output_buffer, 1);
  int32_t Output_min_2 = _halide_buffer_get_min(Output_buffer, 2);
  int32_t Output_extent_2 = _halide_buffer_get_extent(Output_buffer, 2);
  int32_t Output_stride_2 = _halide_buffer_get_stride(Output_buffer, 2);
  int32_t Output_min_3 = _halide_buffer_get_min(Output_buffer, 3);
  int32_t Output_extent_3 = _halide_buffer_get_extent(Output_buffer, 3);
  int32_t Output_stride_3 = _halide_buffer_get_stride(Output_buffer, 3);
  int32_t Output_min_4 = _halide_buffer_get_min(Output_buffer, 4);
  int32_t Output_extent_4 = _halide_buffer_get_extent(Output_buffer, 4);
  int32_t Output_stride_4 = _halide_buffer_get_stride(Output_buffer, 4);
  int32_t Output_min_5 = _halide_buffer_get_min(Output_buffer, 5);
  int32_t Output_extent_5 = _halide_buffer_get_extent(Output_buffer, 5);
  int32_t Output_stride_5 = _halide_buffer_get_stride(Output_buffer, 5);
  int32_t Output_extent_4_required = (((HalfSpaceOut ? (((C_extent_1 + -1)) / 4) : 0)) + (((C_extent_0 + 3)) / 4));
  if (_halide_buffer_is_bounds_query(A_buffer)) {
    struct halide_dimension_t s0[2] = {
      {A_min_0, A_extent_0, 1, 0},
      {A_min_1, A_extent_1, A_extent_0, 0},
    };
  }
  if (_halide_buffer_is_bounds_query(B_buffer)) {
    struct halide_dimension_t s1[2] = {
      {B_min_0, B_extent_0, 1, 0},
      {B_min_1, B_extent_1, B_extent_0, 0},
    };
  }
  if (_halide_buffer_is_bounds_query(C_buffer)) {
    struct halide_dimension_t s2[2] = {
      {C_min_0, C_extent_0, 1, 0},
      {C_min_1, C_extent_1, C_extent_0, 0},
    };
  }
  if (_halide_buffer_is_bounds_query(Output_buffer)) {
    struct halide_dimension_t s3[6] = {
      {0, 2, 1, 0},
      {0, 2, 2, 0},
      {0, 2, 4, 0},
      {0, 2, 8, 0},
      {0, Output_extent_4_required, 16, 0},
      {0, (((C_extent_1 + 3)) / 4), (Output_extent_4_required * 16), 0},
    };
  }
  if (!((_halide_buffer_is_bounds_query(Output_buffer) || ((_halide_buffer_is_bounds_query(C_buffer) || ((_halide_buffer_is_bounds_query(A_buffer) || _halide_buffer_is_bounds_query(B_buffer)))))))) {
    int64_t A_total_extent_1 = ((int64_t)(A_extent_1) * (int64_t)(A_extent_0));
    int64_t B_total_extent_1 = ((int64_t)(B_extent_1) * (int64_t)(B_extent_0));
    int64_t C_total_extent_1 = ((int64_t)(C_extent_1) * (int64_t)(C_extent_0));
    int64_t Output_total_extent_1 = ((int64_t)(Output_extent_1) * (int64_t)(Output_extent_0));
    int64_t Output_total_extent_2 = (Output_total_extent_1 * (int64_t)(Output_extent_2));
    int64_t Output_total_extent_3 = (Output_total_extent_2 * (int64_t)(Output_extent_3));
    int64_t Output_total_extent_4 = (Output_total_extent_3 * (int64_t)(Output_extent_4));
    int64_t Output_total_extent_5 = (Output_total_extent_4 * (int64_t)(Output_extent_5));
    int32_t serializer_2_j_extent_realized = max((HalfSpaceOut ? ((((C_extent_0 + 3)) / 4) - (((C_extent_1 + 3)) / 4)) : (((C_extent_0 + 3)) / 4)), (((C_extent_0 + 3)) / 4));
    halide_buffer_t b0;
    struct halide_dimension_t s4[6] = {
      {0, 2, 1, 0},
      {0, 2, 2, 0},
      {0, 2, 4, 0},
      {0, 2, 8, 0},
      {0, serializer_2_j_extent_realized, 16, 0},
      {0, (((C_extent_1 + 3)) / 4), (serializer_2_j_extent_realized * 16), 0},
    };
    struct halide_dimension_t s5[6] = {
      {0, 2, 1, 0},
      {0, 2, 2, 0},
      {0, 2, 4, 0},
      {0, 2, 8, 0},
      {0, serializer_2_j_extent_realized, 16, 0},
      {0, (((C_extent_1 + 3)) / 4), (serializer_2_j_extent_realized * 16), 0},
    };
    struct halide_buffer_t * serializer_2_mem_channel_buffer = _halide_buffer_init(&b0, s4, (void *)((uint64_t)(ADD_UINT64_T_SUFFIX(0))), (uint64_t)(ADD_UINT64_T_SUFFIX(0)), (struct halide_device_interface_t *)((uint64_t)(ADD_UINT64_T_SUFFIX(0))), 5, 64, 6, s5, (uint64_t)(ADD_UINT64_T_SUFFIX(0)));
    int32_t halide_device_and_host_malloc_result_3 = 0;
    halide_sycl_device_and_host_malloc(serializer_2_mem_channel_buffer, q_device);
;
    {
      complexf *serializer_2_mem_channel = (complexf *)(_halide_buffer_get_host(serializer_2_mem_channel_buffer));
      if (!serializer_2_mem_channel)
      {
        log("Condition 'serializer_2_mem_channel' failed with error id_msg: None");
        assert(false);
      }
      {
        int32_t addr_temp;
        addr_temp = 0;
        int32_t halide_copy_to_host_result_4 = 0;
        halide_sycl_buffer_copy(C_buffer, 1, q_device);
;
        // kernel_serializer_2
        log("kernel kernel_serializer_2");
        complexf *C = (complexf*)(C_buffer->host);
        serializer_2_mem_channel = (complexf*)(serializer_2_mem_channel_buffer->host);
        {
          for (int i = 0; i < (((C_extent_1 + 3)) / 4); i++) {
            int serializer_2_s0_j_loop_min = (HalfSpaceOut ? i : 0);
            for (int j = serializer_2_s0_j_loop_min; j < (((C_extent_0 + 3)) / 4); j++) {
              for (int iii = 0; iii < 2; iii++) {
                for (int ii = 0; ii < 2; ii++) {
                  for (int jj = 0; jj < 2; jj++) {
                    for (int jjj = 0; jjj < 2; jjj++) {
                      auto _D0 = (((((((HermitianC || SymmetricC)) && (UpC != (((i * 4) + (((ii * 2) + iii))) <= ((j * 4) + (((jj * 2) + jjj)))))) ? ((i * 4) + (((ii * 2) + iii))) : ((j * 4) + (((jj * 2) + jjj))))) + ((((((HermitianC || SymmetricC)) && (UpC != (((i * 4) + (((ii * 2) + iii))) <= ((j * 4) + (((jj * 2) + jjj)))))) ? ((j * 4) + (((jj * 2) + jjj))) : ((i * 4) + (((ii * 2) + iii))))) * C_stride_1)) - (((C_min_1 * C_stride_1) + C_min_0)));
                      auto _D1 = (((((((HermitianC || SymmetricC)) && (UpC != (((i * 4) + (((ii * 2) + iii))) <= ((j * 4) + (((jj * 2) + jjj)))))) ? ((i * 4) + (((ii * 2) + iii))) : ((j * 4) + (((jj * 2) + jjj))))) + ((((((HermitianC || SymmetricC)) && (UpC != (((i * 4) + (((ii * 2) + iii))) <= ((j * 4) + (((jj * 2) + jjj)))))) ? ((j * 4) + (((jj * 2) + jjj))) : ((i * 4) + (((ii * 2) + iii))))) * C_stride_1)) - (((C_min_1 * C_stride_1) + C_min_0)));
                      auto _D2 = (((((((HermitianC || SymmetricC)) && (UpC != (((i * 4) + (((ii * 2) + iii))) <= ((j * 4) + (((jj * 2) + jjj)))))) ? ((i * 4) + (((ii * 2) + iii))) : ((j * 4) + (((jj * 2) + jjj))))) + ((((((HermitianC || SymmetricC)) && (UpC != (((i * 4) + (((ii * 2) + iii))) <= ((j * 4) + (((jj * 2) + jjj)))))) ? ((j * 4) + (((jj * 2) + jjj))) : ((i * 4) + (((ii * 2) + iii))))) * C_stride_1)) - (((C_min_1 * C_stride_1) + C_min_0)));
                      auto _D3 = ((((((i * 4) + (((ii * 2) + iii))) < C_extent_1) && (beta != float_from_bits(0))) && (((((j * 2) + jj)) * 2) < C_extent_0)) ? (((HermitianC && ((((HermitianC || SymmetricC)) && (UpC != (((i * 4) + (((ii * 2) + iii))) <= ((j * 4) + (((jj * 2) + jjj)))))))) ? std::conj(((complexf *)C)[_D1]) : ((complexf *)C)[_D2]) /* conditional_conjugate_c64((HermitianC && ((((HermitianC || SymmetricC)) && (UpC != (((i * 4) + (((ii * 2) + iii))) <= ((j * 4) + (((jj * 2) + jjj)))))))), ((complexf *)C)[_D0]) replaced */ * beta) : (complexf)(ADD_UINT64_T_SUFFIX(0)));
                      serializer_2_mem_channel[addr_temp] = _D3;
                      addr_temp = (addr_temp + 1);
                    }
                  }
                }
              }
            }
          }
        }
        _halide_buffer_set_host_dirty(serializer_2_mem_channel_buffer, (bool)(ADD_UINT64_T_SUFFIX(1)));
      }
      halide_sycl_buffer_copy(serializer_2_mem_channel_buffer, 0, q_device);

      kernels_used_to_measure_time.push_back(oneapi_kernel_events.size());
      // kernel_DC
      log("kernel kernel_DC");
      serializer_2_mem_channel = (complexf*)(((device_handle*) serializer_2_mem_channel_buffer->device)->mem);
      oneapi_kernel_events.push_back(q_device.submit([&](sycl::handler &h){
        h.single_task<class kernel_DC_class>([=](){
          int addr_temp;
          addr_temp = 0;
          for (int i = 0; i < ((C_extent_1 + 3)) >> 2; i++) {
            for (int j = (HalfSpaceOut ? i : 0); j < (((C_extent_0 + 3)) / 4); j++) {
              for (int iii = 0; iii < 2; iii++) {
                for (int ii = 0; ii < 2; ii++) {
                  for (int jj = 0; jj < 2; jj++) {
                    DC_channel::write<>(complexf2{
                      serializer_2_mem_channel[(addr_temp * 2) + 0],
                      serializer_2_mem_channel[(addr_temp * 2) + 1]
                    });
                    addr_temp = (addr_temp + 1);
                  }
                }
              }
            }
          }
        }); //  h.single_task kernel_DC_class
      })); // q_device.submit
      int32_t serializer_k_extent_realized_s = (TransposeA ? A_extent_1 : A_extent_0);
      halide_buffer_t b1;
      struct halide_dimension_t s6[9] = {
        {0, 2, 1, 0},
        {0, 1, 2, 0},
        {0, 2, 2, 0},
        {0, 1, 4, 0},
        {0, 2, 4, 0},
        {0, 2, 8, 0},
        {0, (((serializer_k_extent_realized_s + 3)) / 4), 16, 0},
        {0, 1, ((((serializer_k_extent_realized_s + 3)) / 4) * 16), 0},
        {0, (((C_extent_1 + 3)) / 4), ((((serializer_k_extent_realized_s + 3)) / 4) * 16), 0},
      };
      struct halide_dimension_t s7[9] = {
        {0, 2, 1, 0},
        {0, 1, 2, 0},
        {0, 2, 2, 0},
        {0, 1, 4, 0},
        {0, 2, 4, 0},
        {0, 2, 8, 0},
        {0, (((serializer_k_extent_realized_s + 3)) / 4), 16, 0},
        {0, 1, ((((serializer_k_extent_realized_s + 3)) / 4) * 16), 0},
        {0, (((C_extent_1 + 3)) / 4), ((((serializer_k_extent_realized_s + 3)) / 4) * 16), 0},
      };
      struct halide_buffer_t * serializer_mem_channel_buffer = _halide_buffer_init(&b1, s6, (void *)((uint64_t)(ADD_UINT64_T_SUFFIX(0))), (uint64_t)(ADD_UINT64_T_SUFFIX(0)), (struct halide_device_interface_t *)((uint64_t)(ADD_UINT64_T_SUFFIX(0))), 5, 64, 9, s7, (uint64_t)(ADD_UINT64_T_SUFFIX(0)));
      int32_t halide_device_and_host_malloc_result_2 = 0;
      halide_sycl_device_and_host_malloc(serializer_mem_channel_buffer, q_device);
;
      {
        complexf *serializer_mem_channel = (complexf *)(_halide_buffer_get_host(serializer_mem_channel_buffer));
        if (!serializer_mem_channel)
        {
          log("Condition 'serializer_mem_channel' failed with error id_msg: None");
          assert(false);
        }
        {
          int32_t addr_temp;
          addr_temp = 0;
          int32_t halide_copy_to_host_result_2 = 0;
          halide_sycl_buffer_copy(A_buffer, 1, q_device);
;
          // kernel_serializer
          log("kernel kernel_serializer");
          complexf *A = (complexf*)(A_buffer->host);
          serializer_mem_channel = (complexf*)(serializer_mem_channel_buffer->host);
          {
            for (int i = 0; i < (((C_extent_1 + 3)) / 4); i++) {
              for (int k = 0; k < (((serializer_k_extent_realized_s + 3)) / 4); k++) {
                for (int kk = 0; kk < 2; kk++) {
                  for (int ii = 0; ii < 2; ii++) {
                    for (int iii = 0; iii < 2; iii++) {
                      for (int kkk = 0; kkk < 2; kkk++) {
                        auto _D4 = (((((((HermitianA || SymmetricA)) && (UpA != (((TransposeA ? ((k * 4) + (((kk * 2) + kkk))) : ((i * 4) + (((ii * 2) + iii))))) <= ((TransposeA ? ((i * 4) + (((ii * 2) + iii))) : ((k * 4) + (((kk * 2) + kkk)))))))) ? ((TransposeA ? ((k * 4) + (((kk * 2) + kkk))) : ((i * 4) + (((ii * 2) + iii))))) : ((TransposeA ? ((i * 4) + (((ii * 2) + iii))) : ((k * 4) + (((kk * 2) + kkk))))))) + ((((((HermitianA || SymmetricA)) && (UpA != (((TransposeA ? ((k * 4) + (((kk * 2) + kkk))) : ((i * 4) + (((ii * 2) + iii))))) <= ((TransposeA ? ((i * 4) + (((ii * 2) + iii))) : ((k * 4) + (((kk * 2) + kkk)))))))) ? ((TransposeA ? ((i * 4) + (((ii * 2) + iii))) : ((k * 4) + (((kk * 2) + kkk))))) : ((TransposeA ? ((k * 4) + (((kk * 2) + kkk))) : ((i * 4) + (((ii * 2) + iii))))))) * A_stride_1)) - (((A_min_1 * A_stride_1) + A_min_0)));
                        auto _D5 = (((((((HermitianA || SymmetricA)) && (UpA != (((TransposeA ? ((k * 4) + (((kk * 2) + kkk))) : ((i * 4) + (((ii * 2) + iii))))) <= ((TransposeA ? ((i * 4) + (((ii * 2) + iii))) : ((k * 4) + (((kk * 2) + kkk)))))))) ? ((TransposeA ? ((k * 4) + (((kk * 2) + kkk))) : ((i * 4) + (((ii * 2) + iii))))) : ((TransposeA ? ((i * 4) + (((ii * 2) + iii))) : ((k * 4) + (((kk * 2) + kkk))))))) + ((((((HermitianA || SymmetricA)) && (UpA != (((TransposeA ? ((k * 4) + (((kk * 2) + kkk))) : ((i * 4) + (((ii * 2) + iii))))) <= ((TransposeA ? ((i * 4) + (((ii * 2) + iii))) : ((k * 4) + (((kk * 2) + kkk)))))))) ? ((TransposeA ? ((i * 4) + (((ii * 2) + iii))) : ((k * 4) + (((kk * 2) + kkk))))) : ((TransposeA ? ((k * 4) + (((kk * 2) + kkk))) : ((i * 4) + (((ii * 2) + iii))))))) * A_stride_1)) - (((A_min_1 * A_stride_1) + A_min_0)));
                        auto _D6 = (((((((HermitianA || SymmetricA)) && (UpA != (((TransposeA ? ((k * 4) + (((kk * 2) + kkk))) : ((i * 4) + (((ii * 2) + iii))))) <= ((TransposeA ? ((i * 4) + (((ii * 2) + iii))) : ((k * 4) + (((kk * 2) + kkk)))))))) ? ((TransposeA ? ((k * 4) + (((kk * 2) + kkk))) : ((i * 4) + (((ii * 2) + iii))))) : ((TransposeA ? ((i * 4) + (((ii * 2) + iii))) : ((k * 4) + (((kk * 2) + kkk))))))) + ((((((HermitianA || SymmetricA)) && (UpA != (((TransposeA ? ((k * 4) + (((kk * 2) + kkk))) : ((i * 4) + (((ii * 2) + iii))))) <= ((TransposeA ? ((i * 4) + (((ii * 2) + iii))) : ((k * 4) + (((kk * 2) + kkk)))))))) ? ((TransposeA ? ((i * 4) + (((ii * 2) + iii))) : ((k * 4) + (((kk * 2) + kkk))))) : ((TransposeA ? ((k * 4) + (((kk * 2) + kkk))) : ((i * 4) + (((ii * 2) + iii))))))) * A_stride_1)) - (((A_min_1 * A_stride_1) + A_min_0)));
                        auto _D7 = (((((i * 4) + (((ii * 2) + iii))) < C_extent_1) && (((((k * 2) + kk)) * 2) < serializer_k_extent_realized_s)) ? ((HermitianA ? (((((HermitianA || SymmetricA)) && (UpA != (((TransposeA ? ((k * 4) + (((kk * 2) + kkk))) : ((i * 4) + (((ii * 2) + iii))))) <= ((TransposeA ? ((i * 4) + (((ii * 2) + iii))) : ((k * 4) + (((kk * 2) + kkk))))))))) != ConjugateA) : ConjugateA) ? std::conj(((complexf *)A)[_D5]) : ((complexf *)A)[_D6]) /* conditional_conjugate_c64((HermitianA ? (((((HermitianA || SymmetricA)) && (UpA != (((TransposeA ? ((k * 4) + (((kk * 2) + kkk))) : ((i * 4) + (((ii * 2) + iii))))) <= ((TransposeA ? ((i * 4) + (((ii * 2) + iii))) : ((k * 4) + (((kk * 2) + kkk))))))))) != ConjugateA) : ConjugateA), ((complexf *)A)[_D4]) replaced */ : (complexf)(ADD_UINT64_T_SUFFIX(0)));
                        serializer_mem_channel[addr_temp] = _D7;
                        addr_temp = (addr_temp + 1);
                      }
                    }
                  }
                }
              }
            }
          }
          _halide_buffer_set_host_dirty(serializer_mem_channel_buffer, (bool)(ADD_UINT64_T_SUFFIX(1)));
        }
        halide_sycl_buffer_copy(serializer_mem_channel_buffer, 0, q_device);

        kernels_used_to_measure_time.push_back(oneapi_kernel_events.size());
        // kernel_DA
        log("kernel kernel_DA");
        serializer_mem_channel = (complexf*)(((device_handle*) serializer_mem_channel_buffer->device)->mem);
        oneapi_kernel_events.push_back(q_device.submit([&](sycl::handler &h){
          h.single_task<class kernel_DA_class>([=](){
            int addr_temp;
            addr_temp = 0;
            for (int i = 0; i < ((C_extent_1 + 3)) >> 2; i++) {
              for (int j = (HalfSpaceOut ? i : 0); j < (((C_extent_0 + 3)) / 4); j++) {
                for (int k = 0; k < ((serializer_k_extent_realized_s + 3)) >> 2; k++) {
                  for (int kk = 0; kk < 2; kk++) {
                    for (int ii = 0; ii < 2; ii++) {
                      for (int iii = 0; iii < 2; iii++) {
                        auto _D8 = ((((((addr_temp / ((((((serializer_k_extent_realized_s + 3)) >> 2) * (((((C_extent_0 + 3)) >> 2) - ((HalfSpaceOut ? i : 0))))) * 8))) * (((serializer_k_extent_realized_s + 3)) >> 2)) * 8) + (addr_temp % (((((serializer_k_extent_realized_s + 3)) >> 2) * 8))))) * 2);
                        DA_channel::write<>(complexf2{
                          serializer_mem_channel[_D8 + 0],
                          serializer_mem_channel[_D8 + 1]
                        });
                        addr_temp = (addr_temp + 1);
                      }
                    }
                  }
                }
              }
            }
          }); //  h.single_task kernel_DA_class
        })); // q_device.submit
        kernels_used_to_measure_time.push_back(oneapi_kernel_events.size());
        // kernel_SA
        log("kernel kernel_SA");
        oneapi_kernel_events.push_back(q_device.submit([&](sycl::handler &h){
          h.single_task<class kernel_SA_class>([=](){
            SA_channel_array_t SA_channel_array;
            complexf2 SA_value_shreg;
            uint32_t SA_time_stamp_shreg;
            complexf2 SA_in_v;
            uint SA_cycle;
            // OpenCL's __attribute__((memory, numbanks(2), singlepump, numwriteports(1), numreadports(1)))DB[2][2][2][2]
            [[intel::fpga_memory(), intel::numbanks(2), intel::singlepump, intel::simple_dual_port]]
            complexf2 DB[2][2][2][2];
            fpga_tools::UnrolledLoop<2>([&](auto jjj_init) {
              if ((jjj_init == 0)) {
                SA_cycle = (uint)(ADD_UINT64_T_SUFFIX(0));
              }
            });
            while(1) {
              if (((int)((SA_cycle / (uint)(ADD_UINT64_T_SUFFIX(8)))) < (((HalfSpaceOut ? (((((min(C_extent_0, C_extent_1) + 3)) / 4) * (((((((C_extent_0 + 3)) / 4) * 2) - (((min(C_extent_0, C_extent_1) + 3)) / 4)) + 1))) / 2) : ((((C_extent_0 + 3)) / 4) * ((((C_extent_1 + 3)) / 4))))) * ((((serializer_k_extent_realized_s + 3)) / 4))))) {
                SA_in_v = DA_channel::read<>();
              }
              fpga_tools::UnrolledLoop<2>([&](auto buf) {
                if ((buf == 0)) {
                  SA_value_shreg = SA_in_v;
                  SA_time_stamp_shreg = SA_cycle;
                }
                SA_value_shreg = complexf2{
                sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(SA_value_shreg[0])),
                sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(SA_value_shreg[1]))
                };
                SA_time_stamp_shreg = sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(SA_time_stamp_shreg));
                if ((buf == (int)(((SA_time_stamp_shreg % (uint)(ADD_UINT64_T_SUFFIX(8))) % (uint)(ADD_UINT64_T_SUFFIX(2)))))) {
                  DB[(bool)(((SA_time_stamp_shreg / (uint)(ADD_UINT64_T_SUFFIX(8))) % (uint)(ADD_UINT64_T_SUFFIX(2))))][((int)((SA_time_stamp_shreg % (uint)(ADD_UINT64_T_SUFFIX(8)))) / 4)][(((int)((SA_time_stamp_shreg % (uint)(ADD_UINT64_T_SUFFIX(8)))) / 2) % 2)][buf] = SA_value_shreg;
                }
                if (((uint)(ADD_UINT64_T_SUFFIX(0)) < (SA_time_stamp_shreg / (uint)(ADD_UINT64_T_SUFFIX(8))))) {
                  SA_channel_array.template get<buf>() = DB[!(bool)(((SA_time_stamp_shreg / (uint)(ADD_UINT64_T_SUFFIX(8))) % (uint)(ADD_UINT64_T_SUFFIX(2))))][((int)((SA_time_stamp_shreg % (uint)(ADD_UINT64_T_SUFFIX(8)))) / 4)][(((int)((SA_time_stamp_shreg % (uint)(ADD_UINT64_T_SUFFIX(8)))) / 2) % 2)][buf];
                }
              });
              if (((uint)(ADD_UINT64_T_SUFFIX(0)) < (SA_time_stamp_shreg / (uint)(ADD_UINT64_T_SUFFIX(8))))) {
                SA_channel::write<>(SA_channel_array);
              }
              SA_cycle = (SA_cycle + (uint)(ADD_UINT64_T_SUFFIX(1)));
            }
          }); //  h.single_task kernel_SA_class
        })); // q_device.submit
        halide_buffer_t b2;
        struct halide_dimension_t s8[9] = {
          {0, 2, 1, 0},
          {0, 2, 2, 0},
          {0, 1, 4, 0},
          {0, 2, 4, 0},
          {0, 1, 8, 0},
          {0, 2, 8, 0},
          {0, (((serializer_k_extent_realized_s + 3)) / 4), 16, 0},
          {0, serializer_2_j_extent_realized, ((((serializer_k_extent_realized_s + 3)) / 4) * 16), 0},
          {0, 1, (((((serializer_k_extent_realized_s + 3)) / 4) * serializer_2_j_extent_realized) * 16), 0},
        };
        struct halide_dimension_t s9[9] = {
          {0, 2, 1, 0},
          {0, 2, 2, 0},
          {0, 1, 4, 0},
          {0, 2, 4, 0},
          {0, 1, 8, 0},
          {0, 2, 8, 0},
          {0, (((serializer_k_extent_realized_s + 3)) / 4), 16, 0},
          {0, serializer_2_j_extent_realized, ((((serializer_k_extent_realized_s + 3)) / 4) * 16), 0},
          {0, 1, (((((serializer_k_extent_realized_s + 3)) / 4) * serializer_2_j_extent_realized) * 16), 0},
        };
        struct halide_buffer_t * serializer_1_mem_channel_buffer = _halide_buffer_init(&b2, s8, (void *)((uint64_t)(ADD_UINT64_T_SUFFIX(0))), (uint64_t)(ADD_UINT64_T_SUFFIX(0)), (struct halide_device_interface_t *)((uint64_t)(ADD_UINT64_T_SUFFIX(0))), 5, 64, 9, s9, (uint64_t)(ADD_UINT64_T_SUFFIX(0)));
        int32_t halide_device_and_host_malloc_result_1 = 0;
        halide_sycl_device_and_host_malloc(serializer_1_mem_channel_buffer, q_device);
;
        {
          complexf *serializer_1_mem_channel = (complexf *)(_halide_buffer_get_host(serializer_1_mem_channel_buffer));
          if (!serializer_1_mem_channel)
          {
            log("Condition 'serializer_1_mem_channel' failed with error id_msg: None");
            assert(false);
          }
          {
            int32_t addr_temp;
            addr_temp = 0;
            int32_t halide_copy_to_host_result_3 = 0;
            halide_sycl_buffer_copy(B_buffer, 1, q_device);
;
            // kernel_serializer_1
            log("kernel kernel_serializer_1");
            complexf *B = (complexf*)(B_buffer->host);
            serializer_1_mem_channel = (complexf*)(serializer_1_mem_channel_buffer->host);
            {
              for (int j = 0; j < (((C_extent_0 + 3)) / 4); j++) {
                for (int k = 0; k < (((serializer_k_extent_realized_s + 3)) / 4); k++) {
                  for (int kk = 0; kk < 2; kk++) {
                    for (int jj = 0; jj < 2; jj++) {
                      for (int jjj = 0; jjj < 2; jjj++) {
                        for (int kkk = 0; kkk < 2; kkk++) {
                          auto _D9 = (((((((HermitianB || SymmetricB)) && (UpB != (((TransposeB ? ((j * 4) + (((jj * 2) + jjj))) : ((k * 4) + (((kk * 2) + kkk))))) <= ((TransposeB ? ((k * 4) + (((kk * 2) + kkk))) : ((j * 4) + (((jj * 2) + jjj)))))))) ? ((TransposeB ? ((j * 4) + (((jj * 2) + jjj))) : ((k * 4) + (((kk * 2) + kkk))))) : ((TransposeB ? ((k * 4) + (((kk * 2) + kkk))) : ((j * 4) + (((jj * 2) + jjj))))))) + ((((((HermitianB || SymmetricB)) && (UpB != (((TransposeB ? ((j * 4) + (((jj * 2) + jjj))) : ((k * 4) + (((kk * 2) + kkk))))) <= ((TransposeB ? ((k * 4) + (((kk * 2) + kkk))) : ((j * 4) + (((jj * 2) + jjj)))))))) ? ((TransposeB ? ((k * 4) + (((kk * 2) + kkk))) : ((j * 4) + (((jj * 2) + jjj))))) : ((TransposeB ? ((j * 4) + (((jj * 2) + jjj))) : ((k * 4) + (((kk * 2) + kkk))))))) * B_stride_1)) - (((B_min_1 * B_stride_1) + B_min_0)));
                          auto _D10 = (((((((HermitianB || SymmetricB)) && (UpB != (((TransposeB ? ((j * 4) + (((jj * 2) + jjj))) : ((k * 4) + (((kk * 2) + kkk))))) <= ((TransposeB ? ((k * 4) + (((kk * 2) + kkk))) : ((j * 4) + (((jj * 2) + jjj)))))))) ? ((TransposeB ? ((j * 4) + (((jj * 2) + jjj))) : ((k * 4) + (((kk * 2) + kkk))))) : ((TransposeB ? ((k * 4) + (((kk * 2) + kkk))) : ((j * 4) + (((jj * 2) + jjj))))))) + ((((((HermitianB || SymmetricB)) && (UpB != (((TransposeB ? ((j * 4) + (((jj * 2) + jjj))) : ((k * 4) + (((kk * 2) + kkk))))) <= ((TransposeB ? ((k * 4) + (((kk * 2) + kkk))) : ((j * 4) + (((jj * 2) + jjj)))))))) ? ((TransposeB ? ((k * 4) + (((kk * 2) + kkk))) : ((j * 4) + (((jj * 2) + jjj))))) : ((TransposeB ? ((j * 4) + (((jj * 2) + jjj))) : ((k * 4) + (((kk * 2) + kkk))))))) * B_stride_1)) - (((B_min_1 * B_stride_1) + B_min_0)));
                          auto _D11 = (((((((HermitianB || SymmetricB)) && (UpB != (((TransposeB ? ((j * 4) + (((jj * 2) + jjj))) : ((k * 4) + (((kk * 2) + kkk))))) <= ((TransposeB ? ((k * 4) + (((kk * 2) + kkk))) : ((j * 4) + (((jj * 2) + jjj)))))))) ? ((TransposeB ? ((j * 4) + (((jj * 2) + jjj))) : ((k * 4) + (((kk * 2) + kkk))))) : ((TransposeB ? ((k * 4) + (((kk * 2) + kkk))) : ((j * 4) + (((jj * 2) + jjj))))))) + ((((((HermitianB || SymmetricB)) && (UpB != (((TransposeB ? ((j * 4) + (((jj * 2) + jjj))) : ((k * 4) + (((kk * 2) + kkk))))) <= ((TransposeB ? ((k * 4) + (((kk * 2) + kkk))) : ((j * 4) + (((jj * 2) + jjj)))))))) ? ((TransposeB ? ((k * 4) + (((kk * 2) + kkk))) : ((j * 4) + (((jj * 2) + jjj))))) : ((TransposeB ? ((j * 4) + (((jj * 2) + jjj))) : ((k * 4) + (((kk * 2) + kkk))))))) * B_stride_1)) - (((B_min_1 * B_stride_1) + B_min_0)));
                          auto _D12 = (((((((k * 2) + kk)) * 2) < serializer_k_extent_realized_s) && (((((j * 2) + jj)) * 2) < C_extent_0)) ? ((HermitianB ? (((((HermitianB || SymmetricB)) && (UpB != (((TransposeB ? ((j * 4) + (((jj * 2) + jjj))) : ((k * 4) + (((kk * 2) + kkk))))) <= ((TransposeB ? ((k * 4) + (((kk * 2) + kkk))) : ((j * 4) + (((jj * 2) + jjj))))))))) != ConjugateB) : ConjugateB) ? std::conj(((complexf *)B)[_D10]) : ((complexf *)B)[_D11]) /* conditional_conjugate_c64((HermitianB ? (((((HermitianB || SymmetricB)) && (UpB != (((TransposeB ? ((j * 4) + (((jj * 2) + jjj))) : ((k * 4) + (((kk * 2) + kkk))))) <= ((TransposeB ? ((k * 4) + (((kk * 2) + kkk))) : ((j * 4) + (((jj * 2) + jjj))))))))) != ConjugateB) : ConjugateB), ((complexf *)B)[_D9]) replaced */ : (complexf)(ADD_UINT64_T_SUFFIX(0)));
                          serializer_1_mem_channel[addr_temp] = _D12;
                          addr_temp = (addr_temp + 1);
                        }
                      }
                    }
                  }
                }
              }
            }
            _halide_buffer_set_host_dirty(serializer_1_mem_channel_buffer, (bool)(ADD_UINT64_T_SUFFIX(1)));
          }
          halide_sycl_buffer_copy(serializer_1_mem_channel_buffer, 0, q_device);

          kernels_used_to_measure_time.push_back(oneapi_kernel_events.size());
          // kernel_DB
          log("kernel kernel_DB");
          serializer_1_mem_channel = (complexf*)(((device_handle*) serializer_1_mem_channel_buffer->device)->mem);
          oneapi_kernel_events.push_back(q_device.submit([&](sycl::handler &h){
            h.single_task<class kernel_DB_class>([=](){
              int addr_temp;
              addr_temp = 0;
              for (int i = 0; i < ((C_extent_1 + 3)) >> 2; i++) {
                for (int j = (HalfSpaceOut ? i : 0); j < (((C_extent_0 + 3)) / 4); j++) {
                  for (int k = 0; k < ((serializer_k_extent_realized_s + 3)) >> 2; k++) {
                    for (int kk = 0; kk < 2; kk++) {
                      for (int jj = 0; jj < 2; jj++) {
                        for (int jjj = 0; jjj < 2; jjj++) {
                          auto _D13 = ((addr_temp % ((((((serializer_k_extent_realized_s + 3)) >> 2) * (((((C_extent_0 + 3)) >> 2) - ((HalfSpaceOut ? i : 0))))) * 8))) * 2);
                          DB_channel::write<>(complexf2{
                            serializer_1_mem_channel[_D13 + 0],
                            serializer_1_mem_channel[_D13 + 1]
                          });
                          addr_temp = (addr_temp + 1);
                        }
                      }
                    }
                  }
                }
              }
            }); //  h.single_task kernel_DB_class
          })); // q_device.submit
          kernels_used_to_measure_time.push_back(oneapi_kernel_events.size());
          // kernel_SB
          log("kernel kernel_SB");
          oneapi_kernel_events.push_back(q_device.submit([&](sycl::handler &h){
            h.single_task<class kernel_SB_class>([=](){
              SB_channel_array_t SB_channel_array;
              complexf2 SB_value_shreg;
              uint32_t SB_time_stamp_shreg;
              complexf2 SB_in_v;
              uint SB_cycle;
              // OpenCL's __attribute__((memory, numbanks(2), singlepump, numwriteports(1), numreadports(1)))DB[2][2][2][2]
              [[intel::fpga_memory(), intel::numbanks(2), intel::singlepump, intel::simple_dual_port]]
              complexf2 DB[2][2][2][2];
              fpga_tools::UnrolledLoop<2>([&](auto iii_init) {
                if ((iii_init == 0)) {
                  SB_cycle = (uint)(ADD_UINT64_T_SUFFIX(0));
                }
              });
              while(1) {
                if (((int)((SB_cycle / (uint)(ADD_UINT64_T_SUFFIX(8)))) < (((HalfSpaceOut ? (((((min(C_extent_0, C_extent_1) + 3)) / 4) * (((((((C_extent_0 + 3)) / 4) * 2) - (((min(C_extent_0, C_extent_1) + 3)) / 4)) + 1))) / 2) : ((((C_extent_0 + 3)) / 4) * ((((C_extent_1 + 3)) / 4))))) * ((((serializer_k_extent_realized_s + 3)) / 4))))) {
                  SB_in_v = DB_channel::read<>();
                }
                fpga_tools::UnrolledLoop<2>([&](auto buf) {
                  if ((buf == 0)) {
                    SB_value_shreg = SB_in_v;
                    SB_time_stamp_shreg = SB_cycle;
                  }
                  SB_value_shreg = complexf2{
                  sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(SB_value_shreg[0])),
                  sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(SB_value_shreg[1]))
                  };
                  SB_time_stamp_shreg = sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(SB_time_stamp_shreg));
                  if ((buf == (int)(((SB_time_stamp_shreg % (uint)(ADD_UINT64_T_SUFFIX(8))) % (uint)(ADD_UINT64_T_SUFFIX(2)))))) {
                    DB[(bool)(((SB_time_stamp_shreg / (uint)(ADD_UINT64_T_SUFFIX(8))) % (uint)(ADD_UINT64_T_SUFFIX(2))))][((int)((SB_time_stamp_shreg % (uint)(ADD_UINT64_T_SUFFIX(8)))) / 4)][(((int)((SB_time_stamp_shreg % (uint)(ADD_UINT64_T_SUFFIX(8)))) / 2) % 2)][buf] = SB_value_shreg;
                  }
                  if (((uint)(ADD_UINT64_T_SUFFIX(0)) < (SB_time_stamp_shreg / (uint)(ADD_UINT64_T_SUFFIX(8))))) {
                    SB_channel_array.template get<buf>() = DB[!(bool)(((SB_time_stamp_shreg / (uint)(ADD_UINT64_T_SUFFIX(8))) % (uint)(ADD_UINT64_T_SUFFIX(2))))][((int)((SB_time_stamp_shreg % (uint)(ADD_UINT64_T_SUFFIX(8)))) / 4)][((int)((SB_time_stamp_shreg % (uint)(ADD_UINT64_T_SUFFIX(8)))) % 2)][buf];
                  }
                });
                if (((uint)(ADD_UINT64_T_SUFFIX(0)) < (SB_time_stamp_shreg / (uint)(ADD_UINT64_T_SUFFIX(8))))) {
                  SB_channel::write<>(SB_channel_array);
                }
                SB_cycle = (SB_cycle + (uint)(ADD_UINT64_T_SUFFIX(1)));
              }
            }); //  h.single_task kernel_SB_class
          })); // q_device.submit
          kernels_used_to_measure_time.push_back(oneapi_kernel_events.size());
          // kernel_SignalGenerator
          log("kernel kernel_SignalGenerator");
          oneapi_kernel_events.push_back(q_device.submit([&](sycl::handler &h){
            h.single_task<class kernel_SignalGenerator_class>([=](){
              for (int i = 0; i < ((C_extent_1 + 7)) >> 2; i++) {
                for (int j = (HalfSpaceOut ? i : 0); j < (((C_extent_0 + 3)) / 4); j++) {
                  for (int k = 0; k < ((serializer_k_extent_realized_s + 3)) >> 2; k++) {
                    for (int kk = 0; kk < 2; kk++) {
                      for (int ii = 0; ii < 2; ii++) {
                        for (int jj = 0; jj < 2; jj++) {
                          bool signal0 = (i < ((C_extent_1 + 3)) >> 2);
                          bool signal1 = ((k == 0) && (kk == 0));
                          bool signal2 = ((k == ((serializer_k_extent_realized_s - 1)) >> 2) && (kk == 1));
                          bool signal3 = (((((jj == 0) && (ii == 0)) && (k == ((serializer_k_extent_realized_s - 1)) >> 2)) && (kk == 1)) && (i < ((C_extent_1 + 3)) >> 2));
                          cgs signals = (cgs){signal0, signal1, signal2, signal3};
                          SignalGenerator_channel::write<>(signals);
                        }
                      }
                    }
                  }
                }
              }
            }); //  h.single_task kernel_SignalGenerator_class
          })); // q_device.submit
          // kernel_Product
          log("kernel kernel_Product");
          oneapi_kernel_events.push_back(q_device.submit([&](sycl::handler &h){
            h.single_task<class kernel_Product_class>([=](){
              SB_channel_array_t SB_channel_array;
              SA_channel_array_t SA_channel_array;
              complexf Z_shreg[4][2][2];
              complexf Z_pipe_shreg[2][5];
              int Z_pipe_iter;
              int Z_pipe_base;
              Z_pipe_iter = 8;
              Z_pipe_base = 0;
              while(1) {
                cgs signals = SignalGenerator_channel::read<>();
                bool signal0 = signals.f0;
                bool signal1 = signals.f1;
                bool signal2 = signals.f2;
                bool signal3 = signals.f3;
                complexf2 Y_shreg[2];
                complexf2 X_shreg[2];
                complexf Z[2][2];
                fpga_tools::UnrolledLoop<2>([&](auto iii) {
                  fpga_tools::UnrolledLoop<2>([&](auto jjj) {
                    Z[jjj][iii] = Z_shreg[3][jjj][iii];
                    fpga_tools::UnrolledLoop<3>([&](auto l1) {
                      Z_shreg[(3 - l1)][jjj][iii] = Z_shreg[(2 - l1)][jjj][iii];
                    });
                    Z_shreg[0][jjj][iii] = Z[jjj][iii];
                  });
                });
                if (signal0) {
                  SB_channel_array = SB_channel::read<>();
                  SA_channel_array = SA_channel::read<>();
                }
                fpga_tools::UnrolledLoop<2>([&](auto iii) {
                  fpga_tools::UnrolledLoop<2>([&](auto jjj) {
                    X_shreg[iii] = ((jjj == 0) ? SA_channel_array.template get<iii>() : X_shreg[iii]);
                    X_shreg[iii] = complexf2{
                    sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(X_shreg[iii][0])),
                    sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(X_shreg[iii][1]))
                    };
                    Y_shreg[jjj] = ((iii == 0) ? SB_channel_array.template get<jjj>() : Y_shreg[jjj]);
                    Y_shreg[jjj] = complexf2{
                    sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(Y_shreg[jjj][0])),
                    sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(Y_shreg[jjj][1]))
                    };
                    complexf Z_shreg_;
                    Z_shreg_ = (signal1 ? (complexf)(ADD_UINT64_T_SUFFIX(0)) : sycl::ext::intel::fpga_reg(Z_shreg[0][jjj][iii]));
                    fpga_tools::UnrolledLoop<2>([&](auto kkk) {
                      Z_shreg_ = (Z_shreg_ + (X_shreg[iii][kkk] * Y_shreg[jjj][kkk]));
                    });
                    Z_shreg[0][jjj][iii] = Z_shreg_;
                    fpga_tools::UnrolledLoop<2>([&](auto kkk) {
                      if (((kkk == 1) && signal2)) {
                        Z_pipe_shreg[jjj][(iii * 4)] = Z_shreg[0][jjj][iii];
                      }
                    });
                  });
                });
                if (signal3) {
                  Z_pipe_base = Z_pipe_iter;
                }
                complexf2 Product_channel_;
                fpga_tools::UnrolledLoop<2>([&](auto b_62) {
                  Product_channel_[b_62] = Z_pipe_shreg[b_62][0];
                  fpga_tools::UnrolledLoop<2>([&](auto b_62_dummy) {
                    Product_channel_[b_62_dummy] = sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(Product_channel_[b_62_dummy]));
                  });
                });
                if ((Z_pipe_iter < (Z_pipe_base + 8))) {
                  Product_channel::write<>(Product_channel_);
                }
                fpga_tools::UnrolledLoop<2>([&](auto b_63) {
                  fpga_tools::UnrolledLoop<3>([&](auto l_31) {
                    Z_pipe_shreg[b_63][l_31] = Z_pipe_shreg[b_63][(l_31 + 1)];
                  });
                  Z_pipe_shreg[b_63][3] = sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(Z_pipe_shreg[b_63][4]));
                });
                Z_pipe_iter = (Z_pipe_iter + 1);
              }
            }); //  h.single_task kernel_Product_class
          })); // q_device.submit
          kernels_used_to_measure_time.push_back(oneapi_kernel_events.size());
          // kernel_Out
          log("kernel kernel_Out");
          oneapi_kernel_events.push_back(q_device.submit([&](sycl::handler &h){
            h.single_task<class kernel_Out_class>([=](){
              for (int i = 0; i < ((C_extent_1 + 3)) >> 2; i++) {
                for (int j = (HalfSpaceOut ? i : 0); j < (((C_extent_0 + 3)) / 4); j++) {
                  for (int iii = 0; iii < 2; iii++) {
                    for (int ii = 0; ii < 2; ii++) {
                      for (int jj = 0; jj < 2; jj++) {
                        complexf2 Add_shreg;
                        Add_shreg = (DC_channel::read<>() + (Product_channel::read<>() * alpha));
                        Out_channel::write<>(Add_shreg);
                      }
                    }
                  }
                }
              }
            }); //  h.single_task kernel_Out_class
          })); // q_device.submit
          halide_buffer_t b3;
          struct halide_dimension_t s10[6] = {
            {0, 2, 1, 0},
            {0, 2, 2, 0},
            {0, 2, 4, 0},
            {0, 2, 8, 0},
            {0, serializer_2_j_extent_realized, 16, 0},
            {0, (((C_extent_1 + 3)) / 4), (serializer_2_j_extent_realized * 16), 0},
          };
          struct halide_dimension_t s11[6] = {
            {0, 2, 1, 0},
            {0, 2, 2, 0},
            {0, 2, 4, 0},
            {0, 2, 8, 0},
            {0, serializer_2_j_extent_realized, 16, 0},
            {0, (((C_extent_1 + 3)) / 4), (serializer_2_j_extent_realized * 16), 0},
          };
          struct halide_buffer_t * DOut_mem_channel_buffer = _halide_buffer_init(&b3, s10, (void *)((uint64_t)(ADD_UINT64_T_SUFFIX(0))), (uint64_t)(ADD_UINT64_T_SUFFIX(0)), (struct halide_device_interface_t *)((uint64_t)(ADD_UINT64_T_SUFFIX(0))), 5, 64, 6, s11, (uint64_t)(ADD_UINT64_T_SUFFIX(0)));
          int32_t halide_device_and_host_malloc_result = 0;
          halide_sycl_device_and_host_malloc(DOut_mem_channel_buffer, q_device);
;
          {
            complexf *DOut_mem_channel = (complexf *)(_halide_buffer_get_host(DOut_mem_channel_buffer));
            if (!DOut_mem_channel)
            {
              log("Condition 'DOut_mem_channel' failed with error id_msg: None");
              assert(false);
            }
            halide_sycl_device_malloc(DOut_mem_channel_buffer, q_device);

            kernels_used_to_measure_time.push_back(oneapi_kernel_events.size());
            // kernel_DOut
            log("kernel kernel_DOut");
            DOut_mem_channel = (complexf*)(((device_handle*) DOut_mem_channel_buffer->device)->mem);
            oneapi_kernel_events.push_back(q_device.submit([&](sycl::handler &h){
              h.single_task<class kernel_DOut_class>([=](){
                int addr_temp;
                addr_temp = 0;
                for (int i = 0; i < ((C_extent_1 + 3)) >> 2; i++) {
                  for (int j = (HalfSpaceOut ? i : 0); j < (((C_extent_0 + 3)) / 4); j++) {
                    for (int iii = 0; iii < 2; iii++) {
                      for (int ii = 0; ii < 2; ii++) {
                        for (int jj = 0; jj < 2; jj++) {
                          auto _D14 = Out_channel::read<>();
                          DOut_mem_channel[(addr_temp * 2) + 0] = _D14[0];
                          DOut_mem_channel[(addr_temp * 2) + 1] = _D14[1];
                          addr_temp = (addr_temp + 1);
                        }
                      }
                    }
                  }
                }
              }); //  h.single_task kernel_DOut_class
            })); // q_device.submit
            oneapi_kernel_events.back().wait();
            halide_sycl_device_and_host_free(serializer_2_mem_channel_buffer, q_device);

            halide_sycl_device_and_host_free(serializer_mem_channel_buffer, q_device);

            halide_sycl_device_and_host_free(serializer_1_mem_channel_buffer, q_device);

            _halide_buffer_set_device_dirty(DOut_mem_channel_buffer, (bool)(ADD_UINT64_T_SUFFIX(1)));
            {
              int32_t addr_temp;
              addr_temp = 0;
              int32_t halide_copy_to_host_result_1 = 0;
              halide_sycl_buffer_copy(DOut_mem_channel_buffer, 1, q_device);
;
              int32_t halide_copy_to_host_result_5 = 0;
              halide_sycl_buffer_copy(Output_buffer, 1, q_device);
;
              // kernel_Output
              log("kernel kernel_Output");
              DOut_mem_channel = (complexf*)(DOut_mem_channel_buffer->host);
              complexf *Output = (complexf*)(Output_buffer->host);
              {
                for (int i = 0; i < (((C_extent_1 + 3)) / 4); i++) {
                  int Output_s0_j_loop_min = (HalfSpaceOut ? i : 0);
                  for (int j = Output_s0_j_loop_min; j < (((C_extent_0 + 3)) / 4); j++) {
                    for (int iii = 0; iii < 2; iii++) {
                      for (int ii = 0; ii < 2; ii++) {
                        for (int jj = 0; jj < 2; jj++) {
                          auto _D15 = complexf2{
                            DOut_mem_channel[(addr_temp * 2) + 0],
                            DOut_mem_channel[(addr_temp * 2) + 1]
                          };
                          auto _D16 = (((i * Output_stride_5) + (((j * Output_stride_4) + (((iii * Output_stride_3) + (((jj * Output_stride_1) + (ii * Output_stride_2)))))))) - (((Output_min_5 * Output_stride_5) + (((Output_min_4 * Output_stride_4) + (((Output_min_3 * Output_stride_3) + (((Output_min_2 * Output_stride_2) + (((Output_min_1 * Output_stride_1) + Output_min_0)))))))))));
                          Output[_D16 + 0] = _D15[0];
                          Output[_D16 + 1] = _D15[1];
                          addr_temp = (addr_temp + 1);
                        }
                      }
                    }
                  }
                }
              }
              _halide_buffer_set_host_dirty(Output_buffer, (bool)(ADD_UINT64_T_SUFFIX(1)));
              int32_t halide_device_and_host_free_result = 0;
              halide_sycl_device_and_host_free(DOut_mem_channel_buffer, q_device);
;
            }
            DOut_mem_channel = NULL;
          }
          serializer_1_mem_channel = NULL;
        }
        serializer_mem_channel = NULL;
      }
      serializer_2_mem_channel = NULL;
    }
  }
#ifndef T2SP_NDEBUG
  uint64_t k_earliest_start_time = std::numeric_limits<
    typename sycl::info::event_profiling::command_start::return_type>::max();
  uint64_t k_latest_end_time = std::numeric_limits<
    typename sycl::info::event_profiling::command_end::return_type>::min();
  for (auto i : kernels_used_to_measure_time) {
    uint64_t tmp_start = oneapi_kernel_events[i].get_profiling_info<sycl::info::event_profiling::command_start>();
    uint64_t tmp_end = oneapi_kernel_events[i].get_profiling_info<sycl::info::event_profiling::command_end>();
    if (tmp_start < k_earliest_start_time) {
      k_earliest_start_time = tmp_start;
    }
    if (tmp_end > k_latest_end_time) {
      k_latest_end_time = tmp_end;
    }
  }
  std::cout << "// Execution time of the device kernels (in nanoseconds) = " << (kernels_used_to_measure_time.empty() ? 0 : k_latest_end_time - k_earliest_start_time) << "\n";
#endif
  return oneapi_kernel_events.back();
}
} // namespace t2sp::blas::row_major::cccsmatmul

