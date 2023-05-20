#pragma once
#include "halide_runtime_etc.h"
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "pipe_wrapper.hpp"
#include "complex_helper.hpp"
#include "constexpr_math.hpp"
#include "tuple.hpp"
#include "unrolled_loop.hpp"

using namespace sycl;
namespace t2sp::sgemm {

typedef union {
bool __attribute__ ((aligned(16))) s[16];
struct {bool s0,  s1,  s2,  s3,  s4,  s5,  s6,  s7,  s8,  s9,  sa,  sb,  sc,  sd,  se,  sf;};
} bool16;
typedef union {
bool __attribute__ ((aligned(8))) s[8];
struct {bool s0,  s1,  s2,  s3,  s4,  s5,  s6,  s7;};
} bool8;
using aLoader_channel = pipe_wrapper<class aLoader_channel_pipe, float16, 256>;
using aFeeder_channel_array_t = fpga_tools::NTuple<float16, 10>;
using aFeeder_channel = pipe_wrapper<class aFeeder_channel_pipe, aFeeder_channel_array_t, 256>;
using bLoader_channel = pipe_wrapper<class bLoader_channel_pipe, float16, 256>;
using bFeeder_channel_array_t = fpga_tools::NTuple<float16, 8>;
using bFeeder_channel = pipe_wrapper<class bFeeder_channel_pipe, bFeeder_channel_array_t, 256>;
typedef struct {
    bool s[4];
} signals_t;
using SignalGenerator_channel = sycl::ext::intel::pipe<class _SignalGenerator_channel_pipe, signals_t, 256>;
using Product_channel = pipe_wrapper<class Product_channel_pipe, float8, 256>;
using cLoader_channel = pipe_wrapper<class cLoader_channel_pipe, float8, 256>;
using Out_channel = pipe_wrapper<class Out_channel_pipe, float8, 256>;
auto sgemm(device_selector_t device_selector_v, bool p0, bool p1, float p2, float p3, struct halide_buffer_t *A_buffer, struct halide_buffer_t *B_buffer, struct halide_buffer_t *C_buffer, struct halide_buffer_t *Output_buffer) {
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
  #ifndef T2SP_NDEBUG
  std::cout << "// creating device queues\n";
  #endif
  sycl::queue q_host(sycl::cpu_selector_v, exception_handler, sycl::property::queue::enable_profiling());
  sycl::queue q_device(device_selector_v, exception_handler, sycl::property::queue::enable_profiling());
  #ifndef T2SP_NDEBUG
  std::cout << "// Host: " << q_host.get_device().get_info<sycl::info::device::name>() << "\n";
  std::cout << "// Device: " << q_device.get_device().get_info<sycl::info::device::name>() << "\n";
  #endif
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
      {0, 8, 1, 0},
      {0, 32, 8, 0},
      {0, 32, 256, 0},
      {0, 10, 8192, 0},
      {0, (B_extent_0 + 255) / 256, 81920, 0},
      {0, (A_extent_1 + 319) / 320, (B_extent_0 + 255) / 256 * 81920, 0},
    };
  }
  if (!(_halide_buffer_is_bounds_query(Output_buffer) || (_halide_buffer_is_bounds_query(C_buffer) || (_halide_buffer_is_bounds_query(A_buffer) || _halide_buffer_is_bounds_query(B_buffer))))) {
    int64_t A_total_extent_1 = (int64_t)(A_extent_1) * (int64_t)(A_extent_0);
    int64_t B_total_extent_1 = (int64_t)(B_extent_1) * (int64_t)(B_extent_0);
    int64_t C_total_extent_1 = (int64_t)(C_extent_1) * (int64_t)(C_extent_0);
    int64_t Output_total_extent_1 = (int64_t)(Output_extent_1) * (int64_t)(Output_extent_0);
    int64_t Output_total_extent_2 = Output_total_extent_1 * (int64_t)(Output_extent_2);
    int64_t Output_total_extent_3 = Output_total_extent_2 * (int64_t)(Output_extent_3);
    int64_t Output_total_extent_4 = Output_total_extent_3 * (int64_t)(Output_extent_4);
    int64_t Output_total_extent_5 = Output_total_extent_4 * (int64_t)(Output_extent_5);
    halide_buffer_t b0;
    struct halide_dimension_t s4[9] = {
      {0, 16, 1, 0},
      {0, 1, 16, 0},
      {0, 10, 16, 0},
      {0, 1, 160, 0},
      {0, 32, 160, 0},
      {0, 32, 5120, 0},
      {0, (B_extent_1 + 511) / 512, 163840, 0},
      {0, 1, (B_extent_1 + 511) / 512 * 163840, 0},
      {0, (A_extent_1 + 319) / 320, (B_extent_1 + 511) / 512 * 163840, 0},
    };
    struct halide_dimension_t s5[9] = {
      {0, 16, 1, 0},
      {0, 1, 16, 0},
      {0, 10, 16, 0},
      {0, 1, 160, 0},
      {0, 32, 160, 0},
      {0, 32, 5120, 0},
      {0, (B_extent_1 + 511) / 512, 163840, 0},
      {0, 1, (B_extent_1 + 511) / 512 * 163840, 0},
      {0, (A_extent_1 + 319) / 320, (B_extent_1 + 511) / 512 * 163840, 0},
    };
    struct halide_buffer_t * A_serializer_mem_channel_buffer = _halide_buffer_init(&b0, s4, (void *)((uint64_t)(ADD_UINT64_T_SUFFIX(0))), (uint64_t)(ADD_UINT64_T_SUFFIX(0)), (struct halide_device_interface_t *)((uint64_t)(ADD_UINT64_T_SUFFIX(0))), 2, 32, 9, s5, (uint64_t)(ADD_UINT64_T_SUFFIX(0)));
    int32_t halide_device_and_host_malloc_result_3 = 0; // halide_device_and_host_malloc(A_serializer_mem_channel_buffer, NULL /* halide_oneapi_device_interface() replaced */) replaced with line(s) below 
    if (!A_serializer_mem_channel_buffer->device) { // device malloc
      #ifndef T2SP_NDEBUG
      std::cout << "//	 device malloc A_serializer_mem_channel_buffer\n";
      #endif
      assert(A_serializer_mem_channel_buffer->size_in_bytes() != 0);
      uint64_t lowest_index = 0;
      uint64_t highest_index = 0;
      for (int i = 0; i < A_serializer_mem_channel_buffer->dimensions; i++) {
        if (A_serializer_mem_channel_buffer->dim[i].stride < 0) {
          lowest_index += (uint64_t)(A_serializer_mem_channel_buffer->dim[i].stride) * (A_serializer_mem_channel_buffer->dim[i].extent - 1);
        }
        if (A_serializer_mem_channel_buffer->dim[i].stride > 0) {
          highest_index += (uint64_t)(A_serializer_mem_channel_buffer->dim[i].stride) * (A_serializer_mem_channel_buffer->dim[i].extent - 1);
        }
      }
      device_handle *dev_handle = (device_handle *)std::malloc(sizeof(device_handle));
      dev_handle->mem = (void*)sycl::malloc_device(A_serializer_mem_channel_buffer->size_in_bytes(), q_device);
      dev_handle->offset = 0;
      A_serializer_mem_channel_buffer->device = (uint64_t)dev_handle;
    };
    { // host malloc
      #ifndef T2SP_NDEBUG
      std::cout << "//\t host malloc A_serializer_mem_channel_buffer\n";
      #endif
      assert(A_serializer_mem_channel_buffer->size_in_bytes() != 0);
      A_serializer_mem_channel_buffer->host = (uint8_t*)std::malloc(A_serializer_mem_channel_buffer->size_in_bytes());
      assert(A_serializer_mem_channel_buffer->host != NULL);
    };
    struct s6 { void * const ucon; void * const arg; s6(void *ucon, void *a) : ucon(ucon), arg((void *)a) {} ~s6() { halide_device_and_host_free_as_destructor(ucon, arg); } } d0(_ucon, A_serializer_mem_channel_buffer);
    {
      float *A_serializer_mem_channel = (float *)(_halide_buffer_get_host(A_serializer_mem_channel_buffer));
      if (!A_serializer_mem_channel)
      {
        #ifndef T2SP_NDEBUG
        std::cout << "Condition 'A_serializer_mem_channel' failed with error id_msg: None\n";
        #endif
        assert(false);
      }
      {
        int32_t addr_temp;
        addr_temp = 0;
        int32_t halide_copy_to_host_result_1 = 0; // halide_copy_to_host(A_buffer) replaced with line(s) below 
        { // memcpy 
          bool from_host = (A_buffer->device == 0) || (A_buffer->host_dirty() && A_buffer->host != NULL);
          bool to_host = 1;
          if (!from_host && to_host) {
            #ifndef T2SP_NDEBUG
            std::cout << "//	 memcpy device->host A_buffer\n";
            #endif
            q_device.submit([&](handler& h){ h.memcpy((void *)A_buffer->host, (void *)(((device_handle*)A_buffer->device)->mem), A_buffer->size_in_bytes()); }).wait();
          } else if (from_host && !to_host) {
            #ifndef T2SP_NDEBUG
            std::cout << "//	 memcpy host->device A_buffer\n";
            #endif
            q_device.submit([&](handler& h){ h.memcpy((void *)(((device_handle*)A_buffer->device)->mem), (void *)A_buffer->host, A_buffer->size_in_bytes()); }).wait();
          } else if (!from_host && !to_host) {
            #ifndef T2SP_NDEBUG
            std::cout << "//	 memcpy device->device not implemented yet\n";
            #endif
            assert(false);
          } else {
            #ifndef T2SP_NDEBUG
            std::cout << "//	 memcpy A_buffer Do nothing.\n";
            #endif
          }
        };
        // kernel_A_serializer
        #ifndef T2SP_NDEBUG
        std::cout << "// kernel kernel_A_serializer\n";
        #endif
        float *A = (float*)(A_buffer->host);
        A_serializer_mem_channel = (float*)(A_serializer_mem_channel_buffer->host);
        {
          for (int i = 0; i < (A_extent_1 + 319) / 320; i++) {
            for (int k = 0; k < (B_extent_1 + 511) / 512; k++) {
              for (int kk_ii_iii_kkk = 0; kk_ii_iii_kkk < 163840; kk_ii_iii_kkk++) {
                if (i * 320 + (kk_ii_iii_kkk % 160 / 16 + kk_ii_iii_kkk % 5120 / 160 * 10) < A_extent_1 && (kk_ii_iii_kkk / 5120 + k * 32) * 16 < B_extent_1) {
                  auto _D0 = (!p0 ? k * 512 + (kk_ii_iii_kkk / 5120 * 16 + kk_ii_iii_kkk % 16) : i * 320 + (kk_ii_iii_kkk % 160 / 16 + kk_ii_iii_kkk % 5120 / 160 * 10)) + (!p0 ? i * 320 + (kk_ii_iii_kkk % 160 / 16 + kk_ii_iii_kkk % 5120 / 160 * 10) : k * 512 + (kk_ii_iii_kkk / 5120 * 16 + kk_ii_iii_kkk % 16)) * A_stride_1 - (A_min_1 * A_stride_1 + A_min_0);
                  A_serializer_mem_channel[addr_temp] = ((float *)A)[_D0];
                }
                addr_temp = addr_temp + 1;
              }
            }
          }
        }
        _halide_buffer_set_host_dirty(A_serializer_mem_channel_buffer, (bool)(ADD_UINT64_T_SUFFIX(1)));
      }
      { // memcpy 
        bool from_host = (A_serializer_mem_channel_buffer->device == 0) || (A_serializer_mem_channel_buffer->host_dirty() && A_serializer_mem_channel_buffer->host != NULL);
        bool to_host = 0;
        if (!from_host && to_host) {
          #ifndef T2SP_NDEBUG
          std::cout << "//	 memcpy device->host A_serializer_mem_channel_buffer\n";
          #endif
          q_device.submit([&](handler& h){ h.memcpy((void *)A_serializer_mem_channel_buffer->host, (void *)(((device_handle*)A_serializer_mem_channel_buffer->device)->mem), A_serializer_mem_channel_buffer->size_in_bytes()); }).wait();
        } else if (from_host && !to_host) {
          #ifndef T2SP_NDEBUG
          std::cout << "//	 memcpy host->device A_serializer_mem_channel_buffer\n";
          #endif
          q_device.submit([&](handler& h){ h.memcpy((void *)(((device_handle*)A_serializer_mem_channel_buffer->device)->mem), (void *)A_serializer_mem_channel_buffer->host, A_serializer_mem_channel_buffer->size_in_bytes()); }).wait();
        } else if (!from_host && !to_host) {
          #ifndef T2SP_NDEBUG
          std::cout << "//	 memcpy device->device not implemented yet\n";
          #endif
          assert(false);
        } else {
          #ifndef T2SP_NDEBUG
          std::cout << "//	 memcpy A_serializer_mem_channel_buffer Do nothing.\n";
          #endif
        }
      }
      kernels_used_to_measure_time.push_back(oneapi_kernel_events.size());
      // kernel_aLoader
      #ifndef T2SP_NDEBUG
      std::cout << "// kernel kernel_aLoader\n";
      #endif
      A_serializer_mem_channel = (float*)(((device_handle*) A_serializer_mem_channel_buffer->device)->mem);
      oneapi_kernel_events.push_back(q_device.submit([&](sycl::handler &h){
        h.single_task<class kernel_aLoader_class>([=](){
          int addr_temp;
          addr_temp = 0;
          for (int i = 0; i < (A_extent_1 + 639) / 320; i++) {
            for (int j = 0; j < (B_extent_0 + 255) / 256; j++) {
              for (int k = 0; k < (B_extent_1 + 511) / 512; k++) {
                for (int kk_ii_iii = 0; kk_ii_iii < 10240; kk_ii_iii++) {
                  if (j == 0 && k == 0 || i < (A_extent_1 + 319) / 320) {
                    auto _D1 = (addr_temp / ((B_extent_0 + 255) / 256 * ((B_extent_1 + 511) / 512) * 10240) * ((B_extent_1 + 511) / 512) * 10240 + addr_temp % ((B_extent_1 + 511) / 512 * 10240)) * 16;
                    aLoader_channel::write<>(i * 320 + (kk_ii_iii % 320 / 10 * 10 + kk_ii_iii % 10) < A_extent_1 && (kk_ii_iii / 320 + k * 32) * 16 < B_extent_1 && i < (A_extent_1 + 319) / 320 ? float16{
                      A_serializer_mem_channel[_D1 + 0],
                      A_serializer_mem_channel[_D1 + 1],
                      A_serializer_mem_channel[_D1 + 2],
                      A_serializer_mem_channel[_D1 + 3],
                      A_serializer_mem_channel[_D1 + 4],
                      A_serializer_mem_channel[_D1 + 5],
                      A_serializer_mem_channel[_D1 + 6],
                      A_serializer_mem_channel[_D1 + 7],
                      A_serializer_mem_channel[_D1 + 8],
                      A_serializer_mem_channel[_D1 + 9],
                      A_serializer_mem_channel[_D1 + 10],
                      A_serializer_mem_channel[_D1 + 11],
                      A_serializer_mem_channel[_D1 + 12],
                      A_serializer_mem_channel[_D1 + 13],
                      A_serializer_mem_channel[_D1 + 14],
                      A_serializer_mem_channel[_D1 + 15]
                    } : float16{float_from_bits(0)});
                  }
                  addr_temp = addr_temp + 1;
                }
              }
            }
          }
        }); //  h.single_task kernel_aLoader_class
      })); // q_device.submit
      // kernel_aFeeder
      #ifndef T2SP_NDEBUG
      std::cout << "// kernel kernel_aFeeder\n";
      #endif
      oneapi_kernel_events.push_back(q_device.submit([&](sycl::handler &h){
        h.single_task<class kernel_aFeeder_class>([=](){
          aFeeder_channel_array_t aFeeder_channel_array;
          float16 aFeeder_value_shreg;
          uint32_t aFeeder_time_stamp_shreg;
          float16 aFeeder_in_v;
          uint aFeeder_cycle;
          // OpenCL's __attribute__((memory, numbanks(16), singlepump, numwriteports(1), numreadports(1)))DB[2][32][32][16]
          [[intel::fpga_memory(), intel::numbanks(16), intel::singlepump, intel::simple_dual_port]]
          float16 DB[2][32][32][16];
          fpga_tools::UnrolledLoop<8>([&](auto jjj_init) {
            if (jjj_init == 0) {
              aFeeder_cycle = (uint)(ADD_UINT64_T_SUFFIX(22528));
            }
          });
          while(1) {
            if ((uint)(ADD_UINT64_T_SUFFIX(22528)) <= aFeeder_cycle % (uint)(ADD_UINT64_T_SUFFIX(32768))) {
              aFeeder_in_v = aLoader_channel::read<>();
            }
            fpga_tools::UnrolledLoop<10>([&](auto buf) {
              if (buf == 0) {
                aFeeder_value_shreg = aFeeder_in_v;
                aFeeder_time_stamp_shreg = aFeeder_cycle;
              } else {
                aFeeder_value_shreg = aFeeder_value_shreg;
                aFeeder_time_stamp_shreg = aFeeder_time_stamp_shreg;
              }
              aFeeder_value_shreg = float16{
              sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(aFeeder_value_shreg[0])),
              sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(aFeeder_value_shreg[1])),
              sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(aFeeder_value_shreg[2])),
              sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(aFeeder_value_shreg[3])),
              sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(aFeeder_value_shreg[4])),
              sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(aFeeder_value_shreg[5])),
              sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(aFeeder_value_shreg[6])),
              sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(aFeeder_value_shreg[7])),
              sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(aFeeder_value_shreg[8])),
              sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(aFeeder_value_shreg[9])),
              sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(aFeeder_value_shreg[10])),
              sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(aFeeder_value_shreg[11])),
              sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(aFeeder_value_shreg[12])),
              sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(aFeeder_value_shreg[13])),
              sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(aFeeder_value_shreg[14])),
              sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(aFeeder_value_shreg[15]))
              };
              aFeeder_time_stamp_shreg = sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(aFeeder_time_stamp_shreg));
              if ((uint)(ADD_UINT64_T_SUFFIX(22528)) <= aFeeder_time_stamp_shreg % (uint)(ADD_UINT64_T_SUFFIX(32768))) {
                if (buf == (int)((aFeeder_time_stamp_shreg % (uint)(ADD_UINT64_T_SUFFIX(32768)) - (uint)(ADD_UINT64_T_SUFFIX(22528))) % (uint)(ADD_UINT64_T_SUFFIX(10)))) {
                  DB[(bool)(aFeeder_time_stamp_shreg / (uint)(ADD_UINT64_T_SUFFIX(32768)) % (uint)(ADD_UINT64_T_SUFFIX(2)))][(int)(aFeeder_time_stamp_shreg % (uint)(ADD_UINT64_T_SUFFIX(32768)) - (uint)(ADD_UINT64_T_SUFFIX(22528))) / 320][(int)(aFeeder_time_stamp_shreg % (uint)(ADD_UINT64_T_SUFFIX(32768)) - (uint)(ADD_UINT64_T_SUFFIX(22528))) / 10 % 32][buf] = aFeeder_value_shreg;
                }
              }
              if ((uint)(ADD_UINT64_T_SUFFIX(0)) < aFeeder_time_stamp_shreg / (uint)(ADD_UINT64_T_SUFFIX(32768))) {
                aFeeder_channel_array.template get<buf>() = DB[!(bool)(aFeeder_time_stamp_shreg / (uint)(ADD_UINT64_T_SUFFIX(32768)) % (uint)(ADD_UINT64_T_SUFFIX(2)))][(int)(aFeeder_time_stamp_shreg % (uint)(ADD_UINT64_T_SUFFIX(32768))) / 1024][(int)(aFeeder_time_stamp_shreg % (uint)(ADD_UINT64_T_SUFFIX(32768))) / 32 % 32][buf];
              }
            });
            if ((uint)(ADD_UINT64_T_SUFFIX(0)) < aFeeder_time_stamp_shreg / (uint)(ADD_UINT64_T_SUFFIX(32768))) {
              aFeeder_channel::write<>(aFeeder_channel_array);
            }
            aFeeder_cycle = aFeeder_cycle + (uint)(ADD_UINT64_T_SUFFIX(1));
          }
        }); //  h.single_task kernel_aFeeder_class
      })); // q_device.submit
      int32_t B_serializer_mem_channel_stride_8_s = (B_extent_0 + 255) / 256 * ((B_extent_1 + 511) / 512);
      halide_buffer_t b1;
      struct halide_dimension_t s7[9] = {
        {0, 16, 1, 0},
        {0, 8, 16, 0},
        {0, 1, 128, 0},
        {0, 32, 128, 0},
        {0, 1, 4096, 0},
        {0, 32, 4096, 0},
        {0, (B_extent_1 + 511) / 512, 131072, 0},
        {0, (B_extent_0 + 255) / 256, (B_extent_1 + 511) / 512 * 131072, 0},
        {0, 1, B_serializer_mem_channel_stride_8_s * 131072, 0},
      };
      struct halide_dimension_t s8[9] = {
        {0, 16, 1, 0},
        {0, 8, 16, 0},
        {0, 1, 128, 0},
        {0, 32, 128, 0},
        {0, 1, 4096, 0},
        {0, 32, 4096, 0},
        {0, (B_extent_1 + 511) / 512, 131072, 0},
        {0, (B_extent_0 + 255) / 256, (B_extent_1 + 511) / 512 * 131072, 0},
        {0, 1, B_serializer_mem_channel_stride_8_s * 131072, 0},
      };
      struct halide_buffer_t * B_serializer_mem_channel_buffer = _halide_buffer_init(&b1, s7, (void *)((uint64_t)(ADD_UINT64_T_SUFFIX(0))), (uint64_t)(ADD_UINT64_T_SUFFIX(0)), (struct halide_device_interface_t *)((uint64_t)(ADD_UINT64_T_SUFFIX(0))), 2, 32, 9, s8, (uint64_t)(ADD_UINT64_T_SUFFIX(0)));
      int32_t halide_device_and_host_malloc_result_2 = 0; // halide_device_and_host_malloc(B_serializer_mem_channel_buffer, NULL /* halide_oneapi_device_interface() replaced */) replaced with line(s) below 
      if (!B_serializer_mem_channel_buffer->device) { // device malloc
        #ifndef T2SP_NDEBUG
        std::cout << "//	 device malloc B_serializer_mem_channel_buffer\n";
        #endif
        assert(B_serializer_mem_channel_buffer->size_in_bytes() != 0);
        uint64_t lowest_index = 0;
        uint64_t highest_index = 0;
        for (int i = 0; i < B_serializer_mem_channel_buffer->dimensions; i++) {
          if (B_serializer_mem_channel_buffer->dim[i].stride < 0) {
            lowest_index += (uint64_t)(B_serializer_mem_channel_buffer->dim[i].stride) * (B_serializer_mem_channel_buffer->dim[i].extent - 1);
          }
          if (B_serializer_mem_channel_buffer->dim[i].stride > 0) {
            highest_index += (uint64_t)(B_serializer_mem_channel_buffer->dim[i].stride) * (B_serializer_mem_channel_buffer->dim[i].extent - 1);
          }
        }
        device_handle *dev_handle = (device_handle *)std::malloc(sizeof(device_handle));
        dev_handle->mem = (void*)sycl::malloc_device(B_serializer_mem_channel_buffer->size_in_bytes(), q_device);
        dev_handle->offset = 0;
        B_serializer_mem_channel_buffer->device = (uint64_t)dev_handle;
      };
      { // host malloc
        #ifndef T2SP_NDEBUG
        std::cout << "//\t host malloc B_serializer_mem_channel_buffer\n";
        #endif
        assert(B_serializer_mem_channel_buffer->size_in_bytes() != 0);
        B_serializer_mem_channel_buffer->host = (uint8_t*)std::malloc(B_serializer_mem_channel_buffer->size_in_bytes());
        assert(B_serializer_mem_channel_buffer->host != NULL);
      };
      struct s9 { void * const ucon; void * const arg; s9(void *ucon, void *a) : ucon(ucon), arg((void *)a) {} ~s9() { halide_device_and_host_free_as_destructor(ucon, arg); } } d1(_ucon, B_serializer_mem_channel_buffer);
      {
        float *B_serializer_mem_channel = (float *)(_halide_buffer_get_host(B_serializer_mem_channel_buffer));
        if (!B_serializer_mem_channel)
        {
          #ifndef T2SP_NDEBUG
          std::cout << "Condition 'B_serializer_mem_channel' failed with error id_msg: None\n";
          #endif
          assert(false);
        }
        {
          int32_t addr_temp;
          addr_temp = 0;
          int32_t halide_copy_to_host_result_2 = 0; // halide_copy_to_host(B_buffer) replaced with line(s) below 
          { // memcpy 
            bool from_host = (B_buffer->device == 0) || (B_buffer->host_dirty() && B_buffer->host != NULL);
            bool to_host = 1;
            if (!from_host && to_host) {
              #ifndef T2SP_NDEBUG
              std::cout << "//	 memcpy device->host B_buffer\n";
              #endif
              q_device.submit([&](handler& h){ h.memcpy((void *)B_buffer->host, (void *)(((device_handle*)B_buffer->device)->mem), B_buffer->size_in_bytes()); }).wait();
            } else if (from_host && !to_host) {
              #ifndef T2SP_NDEBUG
              std::cout << "//	 memcpy host->device B_buffer\n";
              #endif
              q_device.submit([&](handler& h){ h.memcpy((void *)(((device_handle*)B_buffer->device)->mem), (void *)B_buffer->host, B_buffer->size_in_bytes()); }).wait();
            } else if (!from_host && !to_host) {
              #ifndef T2SP_NDEBUG
              std::cout << "//	 memcpy device->device not implemented yet\n";
              #endif
              assert(false);
            } else {
              #ifndef T2SP_NDEBUG
              std::cout << "//	 memcpy B_buffer Do nothing.\n";
              #endif
            }
          };
          // kernel_B_serializer
          #ifndef T2SP_NDEBUG
          std::cout << "// kernel kernel_B_serializer\n";
          #endif
          float *B = (float*)(B_buffer->host);
          B_serializer_mem_channel = (float*)(B_serializer_mem_channel_buffer->host);
          {
            for (int j = 0; j < (B_extent_0 + 255) / 256; j++) {
              for (int k = 0; k < (B_extent_1 + 511) / 512; k++) {
                for (int kk_jj_jjj = 0; kk_jj_jjj < 8192; kk_jj_jjj++) {
                  if ((kk_jj_jjj / 256 + k * 32) * 16 < B_extent_1 && j * 256 + (kk_jj_jjj % 256 / 8 * 8 + kk_jj_jjj % 8) < B_extent_0) {
                    auto _D2 = (!p1 ? int16{j * 256 + (kk_jj_jjj % 256 / 8 * 8 + kk_jj_jjj % 8)} : (kk_jj_jjj / 256 + k * 32) * 16 + 1 * int16{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}) + (!p1 ? (kk_jj_jjj / 256 + k * 32) * 16 + 1 * int16{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15} : int16{j * 256 + (kk_jj_jjj % 256 / 8 * 8 + kk_jj_jjj % 8)}) * int16{B_stride_1} - (int16{B_min_1 * B_stride_1 + B_min_0});
                    float16 _V0;
                    _V0[0] = ((float*)B)[_D2[0]];
                    _V0[1] = ((float*)B)[_D2[1]];
                    _V0[2] = ((float*)B)[_D2[2]];
                    _V0[3] = ((float*)B)[_D2[3]];
                    _V0[4] = ((float*)B)[_D2[4]];
                    _V0[5] = ((float*)B)[_D2[5]];
                    _V0[6] = ((float*)B)[_D2[6]];
                    _V0[7] = ((float*)B)[_D2[7]];
                    _V0[8] = ((float*)B)[_D2[8]];
                    _V0[9] = ((float*)B)[_D2[9]];
                    _V0[10] = ((float*)B)[_D2[10]];
                    _V0[11] = ((float*)B)[_D2[11]];
                    _V0[12] = ((float*)B)[_D2[12]];
                    _V0[13] = ((float*)B)[_D2[13]];
                    _V0[14] = ((float*)B)[_D2[14]];
                    _V0[15] = ((float*)B)[_D2[15]];
                    B_serializer_mem_channel[addr_temp * 16 + 0] = _V0[0];
                    B_serializer_mem_channel[addr_temp * 16 + 1] = _V0[1];
                    B_serializer_mem_channel[addr_temp * 16 + 2] = _V0[2];
                    B_serializer_mem_channel[addr_temp * 16 + 3] = _V0[3];
                    B_serializer_mem_channel[addr_temp * 16 + 4] = _V0[4];
                    B_serializer_mem_channel[addr_temp * 16 + 5] = _V0[5];
                    B_serializer_mem_channel[addr_temp * 16 + 6] = _V0[6];
                    B_serializer_mem_channel[addr_temp * 16 + 7] = _V0[7];
                    B_serializer_mem_channel[addr_temp * 16 + 8] = _V0[8];
                    B_serializer_mem_channel[addr_temp * 16 + 9] = _V0[9];
                    B_serializer_mem_channel[addr_temp * 16 + 10] = _V0[10];
                    B_serializer_mem_channel[addr_temp * 16 + 11] = _V0[11];
                    B_serializer_mem_channel[addr_temp * 16 + 12] = _V0[12];
                    B_serializer_mem_channel[addr_temp * 16 + 13] = _V0[13];
                    B_serializer_mem_channel[addr_temp * 16 + 14] = _V0[14];
                    B_serializer_mem_channel[addr_temp * 16 + 15] = _V0[15];
                  }
                  addr_temp = addr_temp + 1;
                }
              }
            }
          }
          _halide_buffer_set_host_dirty(B_serializer_mem_channel_buffer, (bool)(ADD_UINT64_T_SUFFIX(1)));
        }
        { // memcpy 
          bool from_host = (B_serializer_mem_channel_buffer->device == 0) || (B_serializer_mem_channel_buffer->host_dirty() && B_serializer_mem_channel_buffer->host != NULL);
          bool to_host = 0;
          if (!from_host && to_host) {
            #ifndef T2SP_NDEBUG
            std::cout << "//	 memcpy device->host B_serializer_mem_channel_buffer\n";
            #endif
            q_device.submit([&](handler& h){ h.memcpy((void *)B_serializer_mem_channel_buffer->host, (void *)(((device_handle*)B_serializer_mem_channel_buffer->device)->mem), B_serializer_mem_channel_buffer->size_in_bytes()); }).wait();
          } else if (from_host && !to_host) {
            #ifndef T2SP_NDEBUG
            std::cout << "//	 memcpy host->device B_serializer_mem_channel_buffer\n";
            #endif
            q_device.submit([&](handler& h){ h.memcpy((void *)(((device_handle*)B_serializer_mem_channel_buffer->device)->mem), (void *)B_serializer_mem_channel_buffer->host, B_serializer_mem_channel_buffer->size_in_bytes()); }).wait();
          } else if (!from_host && !to_host) {
            #ifndef T2SP_NDEBUG
            std::cout << "//	 memcpy device->device not implemented yet\n";
            #endif
            assert(false);
          } else {
            #ifndef T2SP_NDEBUG
            std::cout << "//	 memcpy B_serializer_mem_channel_buffer Do nothing.\n";
            #endif
          }
        }
        kernels_used_to_measure_time.push_back(oneapi_kernel_events.size());
        // kernel_bLoader
        #ifndef T2SP_NDEBUG
        std::cout << "// kernel kernel_bLoader\n";
        #endif
        B_serializer_mem_channel = (float*)(((device_handle*) B_serializer_mem_channel_buffer->device)->mem);
        oneapi_kernel_events.push_back(q_device.submit([&](sycl::handler &h){
          h.single_task<class kernel_bLoader_class>([=](){
            int addr_temp;
            addr_temp = 0;
            for (int i = 0; i < (A_extent_1 + 639) / 320; i++) {
              for (int j = 0; j < (B_extent_0 + 255) / 256; j++) {
                for (int k = 0; k < (B_extent_1 + 511) / 512; k++) {
                  for (int kk_jj_jjj = 0; kk_jj_jjj < 8192; kk_jj_jjj++) {
                    if (j == 0 && k == 0 || i < (A_extent_1 + 319) / 320) {
                      auto _D3 = addr_temp % ((B_extent_0 + 255) / 256 * ((B_extent_1 + 511) / 512) * 8192) * 16;
                      bLoader_channel::write<>((kk_jj_jjj / 256 + k * 32) * 16 < B_extent_1 && j * 256 + (kk_jj_jjj % 256 / 8 * 8 + kk_jj_jjj % 8) < B_extent_0 && i < (A_extent_1 + 319) / 320 ? float16{
                        B_serializer_mem_channel[_D3 + 0],
                        B_serializer_mem_channel[_D3 + 1],
                        B_serializer_mem_channel[_D3 + 2],
                        B_serializer_mem_channel[_D3 + 3],
                        B_serializer_mem_channel[_D3 + 4],
                        B_serializer_mem_channel[_D3 + 5],
                        B_serializer_mem_channel[_D3 + 6],
                        B_serializer_mem_channel[_D3 + 7],
                        B_serializer_mem_channel[_D3 + 8],
                        B_serializer_mem_channel[_D3 + 9],
                        B_serializer_mem_channel[_D3 + 10],
                        B_serializer_mem_channel[_D3 + 11],
                        B_serializer_mem_channel[_D3 + 12],
                        B_serializer_mem_channel[_D3 + 13],
                        B_serializer_mem_channel[_D3 + 14],
                        B_serializer_mem_channel[_D3 + 15]
                      } : float16{float_from_bits(0)});
                    }
                    addr_temp = addr_temp + 1;
                  }
                }
              }
            }
          }); //  h.single_task kernel_bLoader_class
        })); // q_device.submit
        // kernel_bFeeder
        #ifndef T2SP_NDEBUG
        std::cout << "// kernel kernel_bFeeder\n";
        #endif
        oneapi_kernel_events.push_back(q_device.submit([&](sycl::handler &h){
          h.single_task<class kernel_bFeeder_class>([=](){
            bFeeder_channel_array_t bFeeder_channel_array;
            float16 bFeeder_value_shreg;
            uint32_t bFeeder_time_stamp_shreg;
            float16 bFeeder_in_v;
            uint bFeeder_cycle;
            // OpenCL's __attribute__((memory, numbanks(8), singlepump, numwriteports(1), numreadports(1)))DB[2][32][32][8]
            [[intel::fpga_memory(), intel::numbanks(8), intel::singlepump, intel::simple_dual_port]]
            float16 DB[2][32][32][8];
            fpga_tools::UnrolledLoop<10>([&](auto iii_init) {
              if (iii_init == 0) {
                bFeeder_cycle = (uint)(ADD_UINT64_T_SUFFIX(24576));
              }
            });
            while(1) {
              if ((uint)(ADD_UINT64_T_SUFFIX(24576)) <= bFeeder_cycle % (uint)(ADD_UINT64_T_SUFFIX(32768))) {
                bFeeder_in_v = bLoader_channel::read<>();
              }
              fpga_tools::UnrolledLoop<8>([&](auto buf) {
                if (buf == 0) {
                  bFeeder_value_shreg = bFeeder_in_v;
                  bFeeder_time_stamp_shreg = bFeeder_cycle;
                } else {
                  bFeeder_value_shreg = bFeeder_value_shreg;
                  bFeeder_time_stamp_shreg = bFeeder_time_stamp_shreg;
                }
                bFeeder_value_shreg = float16{
                sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(bFeeder_value_shreg[0])),
                sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(bFeeder_value_shreg[1])),
                sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(bFeeder_value_shreg[2])),
                sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(bFeeder_value_shreg[3])),
                sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(bFeeder_value_shreg[4])),
                sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(bFeeder_value_shreg[5])),
                sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(bFeeder_value_shreg[6])),
                sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(bFeeder_value_shreg[7])),
                sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(bFeeder_value_shreg[8])),
                sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(bFeeder_value_shreg[9])),
                sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(bFeeder_value_shreg[10])),
                sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(bFeeder_value_shreg[11])),
                sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(bFeeder_value_shreg[12])),
                sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(bFeeder_value_shreg[13])),
                sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(bFeeder_value_shreg[14])),
                sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(bFeeder_value_shreg[15]))
                };
                bFeeder_time_stamp_shreg = sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(bFeeder_time_stamp_shreg));
                if ((uint)(ADD_UINT64_T_SUFFIX(24576)) <= bFeeder_time_stamp_shreg % (uint)(ADD_UINT64_T_SUFFIX(32768))) {
                  if (buf == (int)((bFeeder_time_stamp_shreg % (uint)(ADD_UINT64_T_SUFFIX(32768)) - (uint)(ADD_UINT64_T_SUFFIX(24576))) % (uint)(ADD_UINT64_T_SUFFIX(8)))) {
                    DB[(bool)(bFeeder_time_stamp_shreg / (uint)(ADD_UINT64_T_SUFFIX(32768)) % (uint)(ADD_UINT64_T_SUFFIX(2)))][(int)(bFeeder_time_stamp_shreg % (uint)(ADD_UINT64_T_SUFFIX(32768)) - (uint)(ADD_UINT64_T_SUFFIX(24576))) / 256][(int)(bFeeder_time_stamp_shreg % (uint)(ADD_UINT64_T_SUFFIX(32768)) - (uint)(ADD_UINT64_T_SUFFIX(24576))) / 8 % 32][buf] = bFeeder_value_shreg;
                  }
                }
                if ((uint)(ADD_UINT64_T_SUFFIX(0)) < bFeeder_time_stamp_shreg / (uint)(ADD_UINT64_T_SUFFIX(32768))) {
                  bFeeder_channel_array.template get<buf>() = DB[!(bool)(bFeeder_time_stamp_shreg / (uint)(ADD_UINT64_T_SUFFIX(32768)) % (uint)(ADD_UINT64_T_SUFFIX(2)))][(int)(bFeeder_time_stamp_shreg % (uint)(ADD_UINT64_T_SUFFIX(32768))) / 1024][(int)(bFeeder_time_stamp_shreg % (uint)(ADD_UINT64_T_SUFFIX(32768))) % 32][buf];
                }
              });
              if ((uint)(ADD_UINT64_T_SUFFIX(0)) < bFeeder_time_stamp_shreg / (uint)(ADD_UINT64_T_SUFFIX(32768))) {
                bFeeder_channel::write<>(bFeeder_channel_array);
              }
              bFeeder_cycle = bFeeder_cycle + (uint)(ADD_UINT64_T_SUFFIX(1));
            }
          }); //  h.single_task kernel_bFeeder_class
        })); // q_device.submit

        // kernel_SignalGenerator
        kernels_used_to_measure_time.push_back(oneapi_kernel_events.size());
        std::cout << "kernel_SignalGenerator" << std::endl;
        oneapi_kernel_events.push_back(q_device.submit([&](sycl::handler &h) {
            h.single_task<class kernel_SignalGenerator>([=]() {
                for (int i = 0; i < (A_extent_1 + 639) / 320; i++) {
                  for (int j = 0; j < (B_extent_0 + 255) / 256; j++) {
                    for (int k = 0; k < (B_extent_1 + 511) / 512; k++) {
                      for (int kk_ii_jj = 0; kk_ii_jj < 32768; kk_ii_jj++) {
                        bool compute_not_done = (i < (A_extent_1 + 319) / 320);
                        bool reduction_start = (k == 0 && kk_ii_jj / 1024 == 0);
                        bool reduction_end = (kk_ii_jj / 1024 == 31 && k == (B_extent_1 + -1) / 512);
                        bool draining_start = (kk_ii_jj % 32 == 0 && kk_ii_jj % 1024 / 32 == 0 && k == (B_extent_1 + -1) / 512 && kk_ii_jj / 1024 == 31 && i < (A_extent_1 + 319) / 320);
                        signals_t signals = {compute_not_done, reduction_start, reduction_end, draining_start};
                        SignalGenerator_channel::write(signals);
                      }
                    }
                  }
                }
            });
        }));

        //kernels_used_to_measure_time.push_back(oneapi_kernel_events.size());
        // kernel_Product
        #ifndef T2SP_NDEBUG
        std::cout << "// kernel kernel_Product\n";
        #endif
        /*oneapi_kernel_events.push_back(*/q_device.submit([&](sycl::handler &h){
          h.single_task<class kernel_Product_class>([=](){
            bFeeder_channel_array_t bFeeder_channel_array;
            aFeeder_channel_array_t aFeeder_channel_array;
            float Z_shreg[1024][8][10];
            float Z_pipe_shreg[8][9217];
            int Z_pipe_iter;
            int Z_pipe_base;
            Z_pipe_iter = 10240;
            Z_pipe_base = 0;
            while (1) {
                signals_t signals = SignalGenerator_channel::read();
                bool compute_not_done = signals.s[0];
                bool reduction_start = signals.s[1];
                bool reduction_end = signals.s[2];
                bool draining_start = signals.s[3];

                    float16 Y_shreg[8];
                    float16 X_shreg[10];
                    float Z[8][10];
                    fpga_tools::UnrolledLoop<10>([&](auto iii) {
                      fpga_tools::UnrolledLoop<8>([&](auto jjj) {
                        Z[jjj][iii] = Z_shreg[1023][jjj][iii];
                        //fpga_tools::UnrolledLoop<1023, int>([&](auto l1) { // instantiating fold expression with 1024 arguments exceeded expression nesting limit of 256
                        #pragma unroll
                        for (auto l1 = 0; l1 < 1023; l1++) {
                          Z_shreg[1023 - l1][jjj][iii] = Z_shreg[1022 - l1][jjj][iii];
                        };
                        Z_shreg[0][jjj][iii] = Z[jjj][iii];
                      });
                    });
                    if (compute_not_done) {
                      bFeeder_channel_array = bFeeder_channel::read<>();
                      aFeeder_channel_array = aFeeder_channel::read<>();
                    }
                    fpga_tools::UnrolledLoop<10>([&](auto iii) {
                      fpga_tools::UnrolledLoop<8>([&](auto jjj) {
                        X_shreg[iii] = jjj == 0 ? aFeeder_channel_array.template get<iii>() : X_shreg[iii];
                        X_shreg[iii] = float16{
                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(X_shreg[iii][0])),
                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(X_shreg[iii][1])),
                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(X_shreg[iii][2])),
                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(X_shreg[iii][3])),
                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(X_shreg[iii][4])),
                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(X_shreg[iii][5])),
                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(X_shreg[iii][6])),
                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(X_shreg[iii][7])),
                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(X_shreg[iii][8])),
                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(X_shreg[iii][9])),
                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(X_shreg[iii][10])),
                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(X_shreg[iii][11])),
                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(X_shreg[iii][12])),
                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(X_shreg[iii][13])),
                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(X_shreg[iii][14])),
                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(X_shreg[iii][15]))
                        };
                        Y_shreg[jjj] = iii == 0 ? bFeeder_channel_array.template get<jjj>() : Y_shreg[jjj];
                        Y_shreg[jjj] = float16{
                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(Y_shreg[jjj][0])),
                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(Y_shreg[jjj][1])),
                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(Y_shreg[jjj][2])),
                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(Y_shreg[jjj][3])),
                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(Y_shreg[jjj][4])),
                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(Y_shreg[jjj][5])),
                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(Y_shreg[jjj][6])),
                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(Y_shreg[jjj][7])),
                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(Y_shreg[jjj][8])),
                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(Y_shreg[jjj][9])),
                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(Y_shreg[jjj][10])),
                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(Y_shreg[jjj][11])),
                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(Y_shreg[jjj][12])),
                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(Y_shreg[jjj][13])),
                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(Y_shreg[jjj][14])),
                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(Y_shreg[jjj][15]))
                        };
                        float Z_shreg_;
                        Z_shreg_ = reduction_start ? float_from_bits(0) : sycl::ext::intel::fpga_reg(Z_shreg[0][jjj][iii]);
                        fpga_tools::UnrolledLoop<16>([&](auto kkk) {
                          Z_shreg_ = Z_shreg_ + X_shreg[iii][kkk] * Y_shreg[jjj][kkk];
                          if (kkk % 4 == 3) {
                            Z_shreg_ = sycl::ext::intel::fpga_reg(Z_shreg_);
                          }
                        });
                        Z_shreg[0][jjj][iii] = Z_shreg_;
                        fpga_tools::UnrolledLoop<16>([&](auto kkk) {
                          if (kkk == 15 && reduction_end) {
                            Z_pipe_shreg[jjj][iii * 1024] = Z_shreg[0][jjj][iii];
                          }
                        });
                      });
                    });
                    if (draining_start) {
                      Z_pipe_base = Z_pipe_iter;
                    }
                    float8 Product_channel_;
                    fpga_tools::UnrolledLoop<8>([&](auto b_62) {
                      Product_channel_[b_62] = Z_pipe_shreg[b_62][0];
                      fpga_tools::UnrolledLoop<8>([&](auto b_62_dummy) {
                        Product_channel_[b_62_dummy] = sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(Product_channel_[b_62_dummy]));
                      });
                    });
                    if (Z_pipe_iter < Z_pipe_base + 10240) {
                      Product_channel::write<>(Product_channel_);
                    }
                    fpga_tools::UnrolledLoop<8>([&](auto b_63) {
                      fpga_tools::UnrolledLoop<9>([&](auto p_31) {
                        //fpga_tools::UnrolledLoop<1023>([&](auto l_31) {
                        #pragma unroll
                        for (auto l_31 = 0; l_31 < 1023; l_31++) {
                          Z_pipe_shreg[b_63][p_31 * 1024 + l_31] = Z_pipe_shreg[b_63][p_31 * 1024 + l_31 + 1];
                        };
                        Z_pipe_shreg[b_63][p_31 * 1024 + 1023] = sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(Z_pipe_shreg[b_63][p_31 * 1024 + 1024]));
                      });
                    });
                    Z_pipe_iter = Z_pipe_iter + 1;
                  }
          }); //  h.single_task kernel_Product_class
        }); // q_device.submit
        halide_buffer_t b2;
        struct halide_dimension_t s10[6] = {
          {0, 8, 1, 0},
          {0, 32, 8, 0},
          {0, 32, 256, 0},
          {0, 10, 8192, 0},
          {0, (B_extent_0 + 255) / 256, 81920, 0},
          {0, (A_extent_1 + 319) / 320, (B_extent_0 + 255) / 256 * 81920, 0},
        };
        struct halide_dimension_t s11[6] = {
          {0, 8, 1, 0},
          {0, 32, 8, 0},
          {0, 32, 256, 0},
          {0, 10, 8192, 0},
          {0, (B_extent_0 + 255) / 256, 81920, 0},
          {0, (A_extent_1 + 319) / 320, (B_extent_0 + 255) / 256 * 81920, 0},
        };
        struct halide_buffer_t * C_serializer_mem_channel_buffer = _halide_buffer_init(&b2, s10, (void *)((uint64_t)(ADD_UINT64_T_SUFFIX(0))), (uint64_t)(ADD_UINT64_T_SUFFIX(0)), (struct halide_device_interface_t *)((uint64_t)(ADD_UINT64_T_SUFFIX(0))), 2, 32, 6, s11, (uint64_t)(ADD_UINT64_T_SUFFIX(0)));
        int32_t halide_device_and_host_malloc_result_1 = 0; // halide_device_and_host_malloc(C_serializer_mem_channel_buffer, NULL /* halide_oneapi_device_interface() replaced */) replaced with line(s) below 
        if (!C_serializer_mem_channel_buffer->device) { // device malloc
          #ifndef T2SP_NDEBUG
          std::cout << "//	 device malloc C_serializer_mem_channel_buffer\n";
          #endif
          assert(C_serializer_mem_channel_buffer->size_in_bytes() != 0);
          uint64_t lowest_index = 0;
          uint64_t highest_index = 0;
          for (int i = 0; i < C_serializer_mem_channel_buffer->dimensions; i++) {
            if (C_serializer_mem_channel_buffer->dim[i].stride < 0) {
              lowest_index += (uint64_t)(C_serializer_mem_channel_buffer->dim[i].stride) * (C_serializer_mem_channel_buffer->dim[i].extent - 1);
            }
            if (C_serializer_mem_channel_buffer->dim[i].stride > 0) {
              highest_index += (uint64_t)(C_serializer_mem_channel_buffer->dim[i].stride) * (C_serializer_mem_channel_buffer->dim[i].extent - 1);
            }
          }
          device_handle *dev_handle = (device_handle *)std::malloc(sizeof(device_handle));
          dev_handle->mem = (void*)sycl::malloc_device(C_serializer_mem_channel_buffer->size_in_bytes(), q_device);
          dev_handle->offset = 0;
          C_serializer_mem_channel_buffer->device = (uint64_t)dev_handle;
        };
        { // host malloc
          #ifndef T2SP_NDEBUG
          std::cout << "//\t host malloc C_serializer_mem_channel_buffer\n";
          #endif
          assert(C_serializer_mem_channel_buffer->size_in_bytes() != 0);
          C_serializer_mem_channel_buffer->host = (uint8_t*)std::malloc(C_serializer_mem_channel_buffer->size_in_bytes());
          assert(C_serializer_mem_channel_buffer->host != NULL);
        };
        struct s12 { void * const ucon; void * const arg; s12(void *ucon, void *a) : ucon(ucon), arg((void *)a) {} ~s12() { halide_device_and_host_free_as_destructor(ucon, arg); } } d2(_ucon, C_serializer_mem_channel_buffer);
        {
          float *C_serializer_mem_channel = (float *)(_halide_buffer_get_host(C_serializer_mem_channel_buffer));
          if (!C_serializer_mem_channel)
          {
            #ifndef T2SP_NDEBUG
            std::cout << "Condition 'C_serializer_mem_channel' failed with error id_msg: None\n";
            #endif
            assert(false);
          }
          {
            int32_t addr_temp;
            addr_temp = 0;
            int32_t halide_copy_to_host_result_3 = 0; // halide_copy_to_host(C_buffer) replaced with line(s) below 
            { // memcpy 
              bool from_host = (C_buffer->device == 0) || (C_buffer->host_dirty() && C_buffer->host != NULL);
              bool to_host = 1;
              if (!from_host && to_host) {
                #ifndef T2SP_NDEBUG
                std::cout << "//	 memcpy device->host C_buffer\n";
                #endif
                q_device.submit([&](handler& h){ h.memcpy((void *)C_buffer->host, (void *)(((device_handle*)C_buffer->device)->mem), C_buffer->size_in_bytes()); }).wait();
              } else if (from_host && !to_host) {
                #ifndef T2SP_NDEBUG
                std::cout << "//	 memcpy host->device C_buffer\n";
                #endif
                q_device.submit([&](handler& h){ h.memcpy((void *)(((device_handle*)C_buffer->device)->mem), (void *)C_buffer->host, C_buffer->size_in_bytes()); }).wait();
              } else if (!from_host && !to_host) {
                #ifndef T2SP_NDEBUG
                std::cout << "//	 memcpy device->device not implemented yet\n";
                #endif
                assert(false);
              } else {
                #ifndef T2SP_NDEBUG
                std::cout << "//	 memcpy C_buffer Do nothing.\n";
                #endif
              }
            };
            // kernel_C_serializer
            #ifndef T2SP_NDEBUG
            std::cout << "// kernel kernel_C_serializer\n";
            #endif
            float *C = (float*)(C_buffer->host);
            C_serializer_mem_channel = (float*)(C_serializer_mem_channel_buffer->host);
            {
              for (int i = 0; i < (A_extent_1 + 319) / 320; i++) {
                for (int j = 0; j < (B_extent_0 + 255) / 256; j++) {
                  for (int iii_ii_jj = 0; iii_ii_jj < 10240; iii_ii_jj++) {
                    if (p3 != float_from_bits(0)) {
                      auto _D4 = (j * 32 + iii_ii_jj % 32) * 8 + (i * 320 + (iii_ii_jj / 1024 + iii_ii_jj % 1024 / 32 * 10)) * C_stride_1 - (C_min_1 * C_stride_1 + C_min_0);
                      auto _D5 = float8{
                        C[_D4 + 0],
                        C[_D4 + 1],
                        C[_D4 + 2],
                        C[_D4 + 3],
                        C[_D4 + 4],
                        C[_D4 + 5],
                        C[_D4 + 6],
                        C[_D4 + 7]
                      };
                      C_serializer_mem_channel[addr_temp * 8 + 0] = _D5[0];
                      C_serializer_mem_channel[addr_temp * 8 + 1] = _D5[1];
                      C_serializer_mem_channel[addr_temp * 8 + 2] = _D5[2];
                      C_serializer_mem_channel[addr_temp * 8 + 3] = _D5[3];
                      C_serializer_mem_channel[addr_temp * 8 + 4] = _D5[4];
                      C_serializer_mem_channel[addr_temp * 8 + 5] = _D5[5];
                      C_serializer_mem_channel[addr_temp * 8 + 6] = _D5[6];
                      C_serializer_mem_channel[addr_temp * 8 + 7] = _D5[7];
                    }
                    addr_temp = addr_temp + 1;
                  }
                }
              }
            }
            _halide_buffer_set_host_dirty(C_serializer_mem_channel_buffer, (bool)(ADD_UINT64_T_SUFFIX(1)));
          }
          { // memcpy 
            bool from_host = (C_serializer_mem_channel_buffer->device == 0) || (C_serializer_mem_channel_buffer->host_dirty() && C_serializer_mem_channel_buffer->host != NULL);
            bool to_host = 0;
            if (!from_host && to_host) {
              #ifndef T2SP_NDEBUG
              std::cout << "//	 memcpy device->host C_serializer_mem_channel_buffer\n";
              #endif
              q_device.submit([&](handler& h){ h.memcpy((void *)C_serializer_mem_channel_buffer->host, (void *)(((device_handle*)C_serializer_mem_channel_buffer->device)->mem), C_serializer_mem_channel_buffer->size_in_bytes()); }).wait();
            } else if (from_host && !to_host) {
              #ifndef T2SP_NDEBUG
              std::cout << "//	 memcpy host->device C_serializer_mem_channel_buffer\n";
              #endif
              q_device.submit([&](handler& h){ h.memcpy((void *)(((device_handle*)C_serializer_mem_channel_buffer->device)->mem), (void *)C_serializer_mem_channel_buffer->host, C_serializer_mem_channel_buffer->size_in_bytes()); }).wait();
            } else if (!from_host && !to_host) {
              #ifndef T2SP_NDEBUG
              std::cout << "//	 memcpy device->device not implemented yet\n";
              #endif
              assert(false);
            } else {
              #ifndef T2SP_NDEBUG
              std::cout << "//	 memcpy C_serializer_mem_channel_buffer Do nothing.\n";
              #endif
            }
          }
          kernels_used_to_measure_time.push_back(oneapi_kernel_events.size());
          // kernel_cLoader
          #ifndef T2SP_NDEBUG
          std::cout << "// kernel kernel_cLoader\n";
          #endif
          C_serializer_mem_channel = (float*)(((device_handle*) C_serializer_mem_channel_buffer->device)->mem);
          oneapi_kernel_events.push_back(q_device.submit([&](sycl::handler &h){
            h.single_task<class kernel_cLoader_class>([=](){
              int addr_temp;
              addr_temp = 0;
              for (int i = 0; i < (A_extent_1 + 319) / 320; i++) {
                for (int j = 0; j < (B_extent_0 + 255) / 256; j++) {
                  for (int iii_ii_jj = 0; iii_ii_jj < 10240; iii_ii_jj++) {
                    if (p3 != float_from_bits(0)) {
                      cLoader_channel::write<>(float8{
                        C_serializer_mem_channel[addr_temp * 8 + 0],
                        C_serializer_mem_channel[addr_temp * 8 + 1],
                        C_serializer_mem_channel[addr_temp * 8 + 2],
                        C_serializer_mem_channel[addr_temp * 8 + 3],
                        C_serializer_mem_channel[addr_temp * 8 + 4],
                        C_serializer_mem_channel[addr_temp * 8 + 5],
                        C_serializer_mem_channel[addr_temp * 8 + 6],
                        C_serializer_mem_channel[addr_temp * 8 + 7]
                      });
                    }
                    addr_temp = addr_temp + 1;
                  }
                }
              }
            }); //  h.single_task kernel_cLoader_class
          })); // q_device.submit
          kernels_used_to_measure_time.push_back(oneapi_kernel_events.size());
          // kernel_Out
          #ifndef T2SP_NDEBUG
          std::cout << "// kernel kernel_Out\n";
          #endif
          oneapi_kernel_events.push_back(q_device.submit([&](sycl::handler &h){
            h.single_task<class kernel_Out_class>([=](){
              for (int i = 0; i < (A_extent_1 + 319) / 320; i++) {
                for (int j = 0; j < (B_extent_0 + 255) / 256; j++) {
                  for (int iii_ii_jj = 0; iii_ii_jj < 10240; iii_ii_jj++) {
                    float8 Add_shreg;
                    Add_shreg = (p3 == float_from_bits(0) ? float8{float_from_bits(0)} : cLoader_channel::read<>() * float8{p3}) + Product_channel::read<>() * float8{p2};
                    Out_channel::write<>(Add_shreg);
                  }
                }
              }
            }); //  h.single_task kernel_Out_class
          })); // q_device.submit
          halide_buffer_t b3;
          struct halide_dimension_t s13[6] = {
            {0, 8, 1, 0},
            {0, 32, 8, 0},
            {0, 32, 256, 0},
            {0, 10, 8192, 0},
            {0, (B_extent_0 + 255) / 256, 81920, 0},
            {0, (A_extent_1 + 319) / 320, (B_extent_0 + 255) / 256 * 81920, 0},
          };
          struct halide_dimension_t s14[6] = {
            {0, 8, 1, 0},
            {0, 32, 8, 0},
            {0, 32, 256, 0},
            {0, 10, 8192, 0},
            {0, (B_extent_0 + 255) / 256, 81920, 0},
            {0, (A_extent_1 + 319) / 320, (B_extent_0 + 255) / 256 * 81920, 0},
          };
          struct halide_buffer_t * unloader_mem_channel_buffer = _halide_buffer_init(&b3, s13, (void *)((uint64_t)(ADD_UINT64_T_SUFFIX(0))), (uint64_t)(ADD_UINT64_T_SUFFIX(0)), (struct halide_device_interface_t *)((uint64_t)(ADD_UINT64_T_SUFFIX(0))), 2, 32, 6, s14, (uint64_t)(ADD_UINT64_T_SUFFIX(0)));
          int32_t halide_device_and_host_malloc_result = 0; // halide_device_and_host_malloc(unloader_mem_channel_buffer, NULL /* halide_oneapi_device_interface() replaced */) replaced with line(s) below 
          if (!unloader_mem_channel_buffer->device) { // device malloc
            #ifndef T2SP_NDEBUG
            std::cout << "//	 device malloc unloader_mem_channel_buffer\n";
            #endif
            assert(unloader_mem_channel_buffer->size_in_bytes() != 0);
            uint64_t lowest_index = 0;
            uint64_t highest_index = 0;
            for (int i = 0; i < unloader_mem_channel_buffer->dimensions; i++) {
              if (unloader_mem_channel_buffer->dim[i].stride < 0) {
                lowest_index += (uint64_t)(unloader_mem_channel_buffer->dim[i].stride) * (unloader_mem_channel_buffer->dim[i].extent - 1);
              }
              if (unloader_mem_channel_buffer->dim[i].stride > 0) {
                highest_index += (uint64_t)(unloader_mem_channel_buffer->dim[i].stride) * (unloader_mem_channel_buffer->dim[i].extent - 1);
              }
            }
            device_handle *dev_handle = (device_handle *)std::malloc(sizeof(device_handle));
            dev_handle->mem = (void*)sycl::malloc_device(unloader_mem_channel_buffer->size_in_bytes(), q_device);
            dev_handle->offset = 0;
            unloader_mem_channel_buffer->device = (uint64_t)dev_handle;
          };
          { // host malloc
            #ifndef T2SP_NDEBUG
            std::cout << "//\t host malloc unloader_mem_channel_buffer\n";
            #endif
            assert(unloader_mem_channel_buffer->size_in_bytes() != 0);
            unloader_mem_channel_buffer->host = (uint8_t*)std::malloc(unloader_mem_channel_buffer->size_in_bytes());
            assert(unloader_mem_channel_buffer->host != NULL);
          };
          struct s15 { void * const ucon; void * const arg; s15(void *ucon, void *a) : ucon(ucon), arg((void *)a) {} ~s15() { halide_device_and_host_free_as_destructor(ucon, arg); } } d3(_ucon, unloader_mem_channel_buffer);
          {
            float *unloader_mem_channel = (float *)(_halide_buffer_get_host(unloader_mem_channel_buffer));
            if (!unloader_mem_channel)
            {
              #ifndef T2SP_NDEBUG
              std::cout << "Condition 'unloader_mem_channel' failed with error id_msg: None\n";
              #endif
              assert(false);
            }
            if (!unloader_mem_channel_buffer->device) { // device malloc
              #ifndef T2SP_NDEBUG
              std::cout << "//	 device malloc unloader_mem_channel_buffer\n";
              #endif
              assert(unloader_mem_channel_buffer->size_in_bytes() != 0);
              uint64_t lowest_index = 0;
              uint64_t highest_index = 0;
              for (int i = 0; i < unloader_mem_channel_buffer->dimensions; i++) {
                if (unloader_mem_channel_buffer->dim[i].stride < 0) {
                  lowest_index += (uint64_t)(unloader_mem_channel_buffer->dim[i].stride) * (unloader_mem_channel_buffer->dim[i].extent - 1);
                }
                if (unloader_mem_channel_buffer->dim[i].stride > 0) {
                  highest_index += (uint64_t)(unloader_mem_channel_buffer->dim[i].stride) * (unloader_mem_channel_buffer->dim[i].extent - 1);
                }
              }
              device_handle *dev_handle = (device_handle *)std::malloc(sizeof(device_handle));
              dev_handle->mem = (void*)sycl::malloc_device(unloader_mem_channel_buffer->size_in_bytes(), q_device);
              dev_handle->offset = 0;
              unloader_mem_channel_buffer->device = (uint64_t)dev_handle;
            }
            kernels_used_to_measure_time.push_back(oneapi_kernel_events.size());
            // kernel_unloader
            #ifndef T2SP_NDEBUG
            std::cout << "// kernel kernel_unloader\n";
            #endif
            unloader_mem_channel = (float*)(((device_handle*) unloader_mem_channel_buffer->device)->mem);
            oneapi_kernel_events.push_back(q_device.submit([&](sycl::handler &h){
              h.single_task<class kernel_unloader_class>([=](){
                int addr_temp;
                addr_temp = 0;
                for (int i = 0; i < (A_extent_1 + 319) / 320; i++) {
                  for (int j = 0; j < (B_extent_0 + 255) / 256; j++) {
                    for (int iii_ii_jj = 0; iii_ii_jj < 10240; iii_ii_jj++) {
                      auto _D6 = Out_channel::read<>();
                      unloader_mem_channel[addr_temp * 8 + 0] = _D6[0];
                      unloader_mem_channel[addr_temp * 8 + 1] = _D6[1];
                      unloader_mem_channel[addr_temp * 8 + 2] = _D6[2];
                      unloader_mem_channel[addr_temp * 8 + 3] = _D6[3];
                      unloader_mem_channel[addr_temp * 8 + 4] = _D6[4];
                      unloader_mem_channel[addr_temp * 8 + 5] = _D6[5];
                      unloader_mem_channel[addr_temp * 8 + 6] = _D6[6];
                      unloader_mem_channel[addr_temp * 8 + 7] = _D6[7];
                      addr_temp = addr_temp + 1;
                    }
                  }
                }
              }); //  h.single_task kernel_unloader_class
            })); // q_device.submit
            oneapi_kernel_events.back().wait();
            _halide_buffer_set_device_dirty(unloader_mem_channel_buffer, (bool)(ADD_UINT64_T_SUFFIX(1)));
            {
              int32_t addr_temp;
              addr_temp = 0;
              int32_t halide_copy_to_host_result = 0; // halide_copy_to_host(unloader_mem_channel_buffer) replaced with line(s) below 
              { // memcpy 
                bool from_host = (unloader_mem_channel_buffer->device == 0) || (unloader_mem_channel_buffer->host_dirty() && unloader_mem_channel_buffer->host != NULL);
                bool to_host = 1;
                if (!from_host && to_host) {
                  #ifndef T2SP_NDEBUG
                  std::cout << "//	 memcpy device->host unloader_mem_channel_buffer\n";
                  #endif
                  q_device.submit([&](handler& h){ h.memcpy((void *)unloader_mem_channel_buffer->host, (void *)(((device_handle*)unloader_mem_channel_buffer->device)->mem), unloader_mem_channel_buffer->size_in_bytes()); }).wait();
                } else if (from_host && !to_host) {
                  #ifndef T2SP_NDEBUG
                  std::cout << "//	 memcpy host->device unloader_mem_channel_buffer\n";
                  #endif
                  q_device.submit([&](handler& h){ h.memcpy((void *)(((device_handle*)unloader_mem_channel_buffer->device)->mem), (void *)unloader_mem_channel_buffer->host, unloader_mem_channel_buffer->size_in_bytes()); }).wait();
                } else if (!from_host && !to_host) {
                  #ifndef T2SP_NDEBUG
                  std::cout << "//	 memcpy device->device not implemented yet\n";
                  #endif
                  assert(false);
                } else {
                  #ifndef T2SP_NDEBUG
                  std::cout << "//	 memcpy unloader_mem_channel_buffer Do nothing.\n";
                  #endif
                }
              };
              int32_t halide_copy_to_host_result_4 = 0; // halide_copy_to_host(Output_buffer) replaced with line(s) below 
              { // memcpy 
                bool from_host = (Output_buffer->device == 0) || (Output_buffer->host_dirty() && Output_buffer->host != NULL);
                bool to_host = 1;
                if (!from_host && to_host) {
                  #ifndef T2SP_NDEBUG
                  std::cout << "//	 memcpy device->host Output_buffer\n";
                  #endif
                  q_device.submit([&](handler& h){ h.memcpy((void *)Output_buffer->host, (void *)(((device_handle*)Output_buffer->device)->mem), Output_buffer->size_in_bytes()); }).wait();
                } else if (from_host && !to_host) {
                  #ifndef T2SP_NDEBUG
                  std::cout << "//	 memcpy host->device Output_buffer\n";
                  #endif
                  q_device.submit([&](handler& h){ h.memcpy((void *)(((device_handle*)Output_buffer->device)->mem), (void *)Output_buffer->host, Output_buffer->size_in_bytes()); }).wait();
                } else if (!from_host && !to_host) {
                  #ifndef T2SP_NDEBUG
                  std::cout << "//	 memcpy device->device not implemented yet\n";
                  #endif
                  assert(false);
                } else {
                  #ifndef T2SP_NDEBUG
                  std::cout << "//	 memcpy Output_buffer Do nothing.\n";
                  #endif
                }
              };
              // kernel_Output
              #ifndef T2SP_NDEBUG
              std::cout << "// kernel kernel_Output\n";
              #endif
              unloader_mem_channel = (float*)(unloader_mem_channel_buffer->host);
              float *Output = (float*)(Output_buffer->host);
              {
                for (int i = 0; i < (A_extent_1 + 319) / 320; i++) {
                  for (int j = 0; j < (B_extent_0 + 255) / 256; j++) {
                    for (int iii_ii_jj = 0; iii_ii_jj < 10240; iii_ii_jj++) {
                      auto _D7 = float8{
                        unloader_mem_channel[addr_temp * 8 + 0],
                        unloader_mem_channel[addr_temp * 8 + 1],
                        unloader_mem_channel[addr_temp * 8 + 2],
                        unloader_mem_channel[addr_temp * 8 + 3],
                        unloader_mem_channel[addr_temp * 8 + 4],
                        unloader_mem_channel[addr_temp * 8 + 5],
                        unloader_mem_channel[addr_temp * 8 + 6],
                        unloader_mem_channel[addr_temp * 8 + 7]
                      };
                      auto _D8 = i * Output_stride_5 + (j * Output_stride_4 + (iii_ii_jj / 1024 * Output_stride_3 + (iii_ii_jj % 32 * Output_stride_1 + iii_ii_jj % 1024 / 32 * Output_stride_2))) - (Output_min_5 * Output_stride_5 + (Output_min_4 * Output_stride_4 + (Output_min_3 * Output_stride_3 + (Output_min_2 * Output_stride_2 + (Output_min_1 * Output_stride_1 + Output_min_0)))));
                      Output[_D8 + 0] = _D7[0];
                      Output[_D8 + 1] = _D7[1];
                      Output[_D8 + 2] = _D7[2];
                      Output[_D8 + 3] = _D7[3];
                      Output[_D8 + 4] = _D7[4];
                      Output[_D8 + 5] = _D7[5];
                      Output[_D8 + 6] = _D7[6];
                      Output[_D8 + 7] = _D7[7];
                      addr_temp = addr_temp + 1;
                    }
                  }
                }
              }
              _halide_buffer_set_host_dirty(Output_buffer, (bool)(ADD_UINT64_T_SUFFIX(1)));
              int32_t halide_device_and_host_free_result = 0; // halide_device_and_host_free(unloader_mem_channel_buffer) replaced with line(s) below 
              if (unloader_mem_channel_buffer->device) { // device free
               sycl::free(((device_handle*)unloader_mem_channel_buffer->device)->mem , q_device);
               assert(((device_handle *)unloader_mem_channel_buffer->device)->offset == 0);
               std::free((device_handle *)unloader_mem_channel_buffer->device);
               unloader_mem_channel_buffer->set_device_dirty(false);
              }
              if (unloader_mem_channel_buffer->host) { // host free
                std::free((void*)unloader_mem_channel_buffer->host);
                unloader_mem_channel_buffer->host = NULL;
                unloader_mem_channel_buffer->set_host_dirty(false);
              };
            }
            unloader_mem_channel = NULL;
          }
          C_serializer_mem_channel = NULL;
        }
        B_serializer_mem_channel = NULL;
      }
      A_serializer_mem_channel = NULL;
    }
  }
  oneapi_kernel_events.back().wait();
  #ifndef T2SP_NDEBUG
  std::cout << "// return the kernel execution time in nanoseconds\n";
  #endif
  auto k_earliest_start_time = std::numeric_limits<
    typename sycl::info::event_profiling::command_start::return_type>::max();
  auto k_latest_end_time = std::numeric_limits<
    typename sycl::info::event_profiling::command_end::return_type>::min();
  for (auto i : kernels_used_to_measure_time) {
    auto tmp_start = oneapi_kernel_events[i].get_profiling_info<sycl::info::event_profiling::command_start>();
    auto tmp_end = oneapi_kernel_events[i].get_profiling_info<sycl::info::event_profiling::command_end>();
    if (tmp_start < k_earliest_start_time) {
      k_earliest_start_time = tmp_start;
    }
    if (tmp_end > k_latest_end_time) {
      k_latest_end_time = tmp_end;
    }
  }
  // Get time in ns
  return kernels_used_to_measure_time.empty() ? decltype(k_latest_end_time){} : k_latest_end_time - k_earliest_start_time;
}
}

