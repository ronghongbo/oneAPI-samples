#pragma once

#include <iostream>
#include <list>
#include <stdlib.h>
#include <vector>

#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#ifdef FPGA_EMULATOR
#define KKK 4
#define JJJ 4
#define III 4
#define JJ 4
#define II 4
#define KK 4
#define K 4
#define J 4
#define I 4
#else
#define KKK 4
#define JJJ 4
#define III 4
#define JJ 4
#define II 4
#define KK 4
#define K 4
#define J 4
#define I 4
#endif

#define ADD_INT64_T_SUFFIX(x) x##l
#define ADD_UINT64_T_SUFFIX(x) x##ul

#ifdef __SYCL_DEVICE_ONLY__
#define CL_CONSTANT __attribute__((opencl_constant))
#else
#define CL_CONSTANT
#endif

using namespace sycl;

#define PRINTF(format, ...)                                        \
    {                                                              \
        static const CL_CONSTANT char _format[] = format;          \
        ext::oneapi::experimental::printf(_format, ##__VA_ARGS__); \
    }

constexpr size_t TOTAL_I = III * II * I;
constexpr size_t TOTAL_J = JJJ * JJ * J;
constexpr size_t TOTAL_K = KKK * KK * K;
constexpr size_t num_elem_A = TOTAL_K * TOTAL_I;
constexpr size_t num_elem_B = TOTAL_J * TOTAL_K;
constexpr size_t num_elem_C = TOTAL_J * TOTAL_I;

using aLoader_channel =
    sycl::ext::intel::pipe<class _aLoader_channel_pipe, float4, 256>;
typedef struct {
    float4 s[4];
} aFeeder_channel_array_t;
using aFeeder_channel = sycl::ext::intel::pipe<class _aFeeder_channel_pipe,
                                               aFeeder_channel_array_t, 256>;
using bLoader_channel =
    sycl::ext::intel::pipe<class _bLoader_channel_pipe, float4, 256>;
typedef struct {
    float4 s[4];
} bFeeder_channel_array_t;
using bFeeder_channel = sycl::ext::intel::pipe<class _bFeeder_channel_pipe,
                                               bFeeder_channel_array_t, 256>;
using Product_channel = sycl::ext::intel::pipe<class Product_channel_pipe, float4, 256>;
using cLoader_channel = sycl::ext::intel::pipe<class cLoader_channel_pipe, float4, 256>;
using Out_channel = sycl::ext::intel::pipe<class Out_channel_pipe, float4, 256>;

void gemm(float *A, float *B, float *C, float *result, float p2, float p3, sycl::queue &q) {
    std::vector<sycl::event> oneapi_kernel_events;
    float *serialized_A = (float *)malloc(num_elem_A * sizeof(float));
    float *serialized_B = (float *)malloc(num_elem_B * sizeof(float));
    float *serialized_C = (float *)malloc(num_elem_C * sizeof(float));
    float *serialized_reuslt = (float *)malloc(num_elem_C * sizeof(float));
    float *serialized_A_device = sycl::malloc_device<float>(num_elem_A, q);
    float *serialized_B_device = sycl::malloc_device<float>(num_elem_B, q);
    float *serialized_C_device = sycl::malloc_device<float>(num_elem_C, q);
    float *serialized_reuslt_device = sycl::malloc_device<float>(num_elem_C, q);

    // kernel_A_serializer
    std::cout << "kernel_A_serializer" << std::endl;
    size_t addr = 0;
    for (int i = 0; i < I; i++)
        for (int k = 0; k < K; k++)
            for (int kk = 0; kk < KK; kk++)
                for (int ii = 0; ii < II; ii++)
                    for (int iii = 0; iii < III; iii++)
                        for (int kkk = 0; kkk < KKK; kkk++) {
                            int total_k = kkk + KKK * kk + KKK * KK * k;
                            int total_i = iii + III * ii + III * II * i;
                            serialized_A[addr++] =
                                A[total_k + total_i * TOTAL_K];
                        }

    // kernel_B_serializer
    std::cout << "kernel_B_serializer" << std::endl;
    addr = 0;
    for (int j = 0; j < J; j++)
        for (int k = 0; k < K; k++)
            for (int kk = 0; kk < KK; kk++)
                for (int jj = 0; jj < JJ; jj++)
                    for (int jjj = 0; jjj < JJJ; jjj++)
                        for (int kkk = 0; kkk < KKK; kkk++) {
                            int total_k = kkk + KKK * kk + KKK * KK * k;
                            int total_j = jjj + JJJ * jj + JJJ * JJ * j;
                            serialized_B[addr++] =
                                B[total_j + total_k * TOTAL_J];
                        }

    // kernel_C_serializer
    std::cout << "kernel_C_serializer" << std::endl;
    addr = 0;
    for (int i = 0; i < I; i++)
        for (int j = 0; j < J; j++)
            for (int iii = 0; iii < III; iii++)
                for (int ii = 0; ii < II; ii++)
                    for (int jj = 0; jj < JJ; jj++)
                        for (int jjj = 0; jjj < JJJ; jjj++) {
                            int total_i = iii + III * ii + III * II * i;
                            int total_j = jjj + JJJ * jj + JJJ * JJ * j;
                            serialized_C[addr++] =
                                C[total_j + total_i * TOTAL_J];
                        }

    std::cout << "memcpy matrix A host->device" << std::endl;
    q.memcpy(serialized_A_device, serialized_A, num_elem_A * sizeof(float))
        .wait();

    // kernel_aLoader
    std::cout << "kernel_aLoader" << std::endl;
    oneapi_kernel_events.push_back(q.submit([&](sycl::handler &h) {
        h.single_task<class kernel_aLoader>(
            [=]() [[intel::kernel_args_restrict]] {
                device_ptr<float> serialized_A_d(serialized_A_device);
                int addr_temp;
                addr_temp = 0;
                for (int i = 0; i < (TOTAL_I + 31) / 16; i++) {
                    for (int j = 0; j < (TOTAL_J + 15) / 16; j++) {
                        for (int k = 0; k < (TOTAL_K + 15) / 16; k++) {
                            for (int kk_ii_iii = 0; kk_ii_iii < 64; kk_ii_iii++) {
                                if (j == 0 && k == 0 || i < (TOTAL_I + 15) / 16) {
                                    auto _D1 = (addr_temp / ((TOTAL_J + 15) / 16 * ((TOTAL_K + 15) / 16) * 64) * ((TOTAL_K + 15) / 16) * 64 + addr_temp % ((TOTAL_K + 15) / 16 * 64)) * 4;
                                    aLoader_channel::write(i * 16 + (kk_ii_iii % 16 / 4 * 4 + kk_ii_iii % 4) < TOTAL_I && (kk_ii_iii / 16 + k * 4) * 4 < TOTAL_K && i < (TOTAL_I + 15) / 16 ? float4{
                                                                                                                                                                                                  serialized_A_d[_D1 + 0],
                                                                                                                                                                                                  serialized_A_d[_D1 + 1],
                                                                                                                                                                                                  serialized_A_d[_D1 + 2],
                                                                                                                                                                                                  serialized_A_d[_D1 + 3]}
                                                                                                                                                                                            : float4{0.0f});
                                }
                                addr_temp = addr_temp + 1;
                            }
                        }
                    }
                }
            });
    }));

    // kernel_aFeeder
    std::cout << "kernel_aFeeder" << std::endl;
    q.submit([&](sycl::handler &h) {
        h.single_task<class kernel_aFeeder>([=]() {
            aFeeder_channel_array_t aFeeder_channel_array;
            float4 aFeeder_value_shreg;
            uint32_t aFeeder_time_stamp_shreg;
            float4 aFeeder_in_v;
            uint aFeeder_cycle;
            // OpenCL's __attribute__((memory, numbanks(4), singlepump, numwriteports(1), numreadports(1)))DB[2][4][4][4]
            [[intel::fpga_memory(), intel::numbanks(4), intel::singlepump, intel::simple_dual_port]] float4 DB[2][4][4][4];
#pragma unroll
            for (int jjj_init = 0; jjj_init < 4; jjj_init++) {
                if (jjj_init == 0) {
                    aFeeder_cycle = (uint)(ADD_UINT64_T_SUFFIX(0));
                }
            }
            while (1) {
                aFeeder_in_v = aLoader_channel::read();
#pragma unroll
                for (int buf = 0; buf < 4; buf++) {
                    if (buf == 0) {
                        aFeeder_value_shreg = aFeeder_in_v;
                        aFeeder_time_stamp_shreg = aFeeder_cycle;
                    } else {
                        aFeeder_value_shreg = aFeeder_value_shreg;
                        aFeeder_time_stamp_shreg = aFeeder_time_stamp_shreg;
                    }
                    aFeeder_value_shreg = float4{
                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(aFeeder_value_shreg[0])),
                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(aFeeder_value_shreg[1])),
                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(aFeeder_value_shreg[2])),
                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(aFeeder_value_shreg[3]))};
                    aFeeder_time_stamp_shreg = sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(aFeeder_time_stamp_shreg));
                    if (buf == (int)(aFeeder_time_stamp_shreg % (uint)(ADD_UINT64_T_SUFFIX(64)) % (uint)(ADD_UINT64_T_SUFFIX(4)))) {
                        DB[(bool)(aFeeder_time_stamp_shreg / (uint)(ADD_UINT64_T_SUFFIX(64)) % (uint)(ADD_UINT64_T_SUFFIX(2)))][(int)(aFeeder_time_stamp_shreg % (uint)(ADD_UINT64_T_SUFFIX(64))) / 16][(int)(aFeeder_time_stamp_shreg % (uint)(ADD_UINT64_T_SUFFIX(64))) / 4 % 4][buf] = aFeeder_value_shreg;
                    }
                    if ((uint)(ADD_UINT64_T_SUFFIX(0)) < aFeeder_time_stamp_shreg / (uint)(ADD_UINT64_T_SUFFIX(64))) {
                        aFeeder_channel_array.s[buf] = DB[!(bool)(aFeeder_time_stamp_shreg / (uint)(ADD_UINT64_T_SUFFIX(64)) % (uint)(ADD_UINT64_T_SUFFIX(2)))][(int)(aFeeder_time_stamp_shreg % (uint)(ADD_UINT64_T_SUFFIX(64))) / 16][(int)(aFeeder_time_stamp_shreg % (uint)(ADD_UINT64_T_SUFFIX(64))) / 4 % 4][buf];
                    }
                }
                if ((uint)(ADD_UINT64_T_SUFFIX(0)) < aFeeder_time_stamp_shreg / (uint)(ADD_UINT64_T_SUFFIX(64))) {
                    aFeeder_channel::write(aFeeder_channel_array);
                }
                aFeeder_cycle = aFeeder_cycle + (uint)(ADD_UINT64_T_SUFFIX(1));
            }
        });
    });

    std::cout << "memcpy matrix B host->device" << std::endl;
    q.memcpy(serialized_B_device, serialized_B, num_elem_B * sizeof(float))
        .wait();
    // kernel_bLoader
    std::cout << "kernel_bLoader" << std::endl;
    oneapi_kernel_events.push_back(q.submit([&](sycl::handler &h) {
        h.single_task<class kernel_bLoader>(
            [=]() [[intel::kernel_args_restrict]] {
                device_ptr<float> serialized_B_d(serialized_B_device);
                int addr_temp;
                addr_temp = 0;
                for (int i = 0; i < (TOTAL_I + 31) / 16; i++) {
                    for (int j = 0; j < (TOTAL_J + 15) / 16; j++) {
                        for (int k = 0; k < (TOTAL_K + 15) / 16; k++) {
                            for (int kk_jj_jjj = 0; kk_jj_jjj < 64; kk_jj_jjj++) {
                                if (j == 0 && k == 0 || i < (TOTAL_I + 15) / 16) {
                                    auto _D3 = addr_temp % ((TOTAL_J + 15) / 16 * ((TOTAL_K + 15) / 16) * 64) * 4;
                                    bLoader_channel::write((kk_jj_jjj / 16 + k * 4) * 4 < TOTAL_K && j * 16 + (kk_jj_jjj % 16 / 4 * 4 + kk_jj_jjj % 4) < TOTAL_J && i < (TOTAL_I + 15) / 16 ? float4{
                                                                                                                                                                                                  serialized_B_d[_D3 + 0],
                                                                                                                                                                                                  serialized_B_d[_D3 + 1],
                                                                                                                                                                                                  serialized_B_d[_D3 + 2],
                                                                                                                                                                                                  serialized_B_d[_D3 + 3]}
                                                                                                                                                                                            : float4{0.0f});
                                }
                                addr_temp = addr_temp + 1;
                            }
                        }
                    }
                }
            });
    }));

    // kernel_bFeeder
    std::cout << "kernel_bFeeder" << std::endl;
    q.submit([&](sycl::handler &h) {
        h.single_task<class kernel_bFeeder>([=]() {
            bFeeder_channel_array_t bFeeder_channel_array;
            float4 bFeeder_value_shreg;
            uint32_t bFeeder_time_stamp_shreg;
            float4 bFeeder_in_v;
            uint bFeeder_cycle;
            // OpenCL's __attribute__((memory, numbanks(4), singlepump, numwriteports(1), numreadports(1)))DB[2][4][4][4]
            [[intel::fpga_memory(), intel::numbanks(4), intel::singlepump, intel::simple_dual_port]] float4 DB[2][4][4][4];
#pragma unroll
            for (int iii_init = 0; iii_init < 4; iii_init++) {
                if (iii_init == 0) {
                    bFeeder_cycle = (uint)(ADD_UINT64_T_SUFFIX(0));
                }
            }
            while (1) {
                bFeeder_in_v = bLoader_channel::read();
#pragma unroll
                for (int buf = 0; buf < 4; buf++) {
                    if (buf == 0) {
                        bFeeder_value_shreg = bFeeder_in_v;
                        bFeeder_time_stamp_shreg = bFeeder_cycle;
                    } else {
                        bFeeder_value_shreg = bFeeder_value_shreg;
                        bFeeder_time_stamp_shreg = bFeeder_time_stamp_shreg;
                    }
                    bFeeder_value_shreg = float4{
                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(bFeeder_value_shreg[0])),
                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(bFeeder_value_shreg[1])),
                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(bFeeder_value_shreg[2])),
                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(bFeeder_value_shreg[3]))};
                    bFeeder_time_stamp_shreg = sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(bFeeder_time_stamp_shreg));
                    if (buf == (int)(bFeeder_time_stamp_shreg % (uint)(ADD_UINT64_T_SUFFIX(64)) % (uint)(ADD_UINT64_T_SUFFIX(4)))) {
                        DB[(bool)(bFeeder_time_stamp_shreg / (uint)(ADD_UINT64_T_SUFFIX(64)) % (uint)(ADD_UINT64_T_SUFFIX(2)))][(int)(bFeeder_time_stamp_shreg % (uint)(ADD_UINT64_T_SUFFIX(64))) / 16][(int)(bFeeder_time_stamp_shreg % (uint)(ADD_UINT64_T_SUFFIX(64))) / 4 % 4][buf] = bFeeder_value_shreg;
                    }
                    if ((uint)(ADD_UINT64_T_SUFFIX(0)) < bFeeder_time_stamp_shreg / (uint)(ADD_UINT64_T_SUFFIX(64))) {
                        bFeeder_channel_array.s[buf] = DB[!(bool)(bFeeder_time_stamp_shreg / (uint)(ADD_UINT64_T_SUFFIX(64)) % (uint)(ADD_UINT64_T_SUFFIX(2)))][(int)(bFeeder_time_stamp_shreg % (uint)(ADD_UINT64_T_SUFFIX(64))) / 16][(int)(bFeeder_time_stamp_shreg % (uint)(ADD_UINT64_T_SUFFIX(64))) % 4][buf];
                    }
                }
                if ((uint)(ADD_UINT64_T_SUFFIX(0)) < bFeeder_time_stamp_shreg / (uint)(ADD_UINT64_T_SUFFIX(64))) {
                    bFeeder_channel::write(bFeeder_channel_array);
                }
                bFeeder_cycle = bFeeder_cycle + (uint)(ADD_UINT64_T_SUFFIX(1));
            }
        });
    });

    // kernel_Product
    std::cout << "kernel_Product" << std::endl;
    oneapi_kernel_events.push_back(q.submit([&](sycl::handler &h) {
        h.single_task<class kernel_Product>([=]() {
            bFeeder_channel_array_t bFeeder_channel_array;
            aFeeder_channel_array_t aFeeder_channel_array;
            float Z_shreg[16][4][4];
            float Z_pipe_shreg[4][49];
            float4 Y_shreg[4];
            float Z[4][4];
            float4 X_shreg[4];
            float Z_shreg_;
            int Z_pipe_iter;
            int Z_pipe_base;
            Z_pipe_iter = 64;
            Z_pipe_base = 0;
            for (int i = 0; i < (TOTAL_I + 31) / 16; i++) {
                for (int j = 0; j < (TOTAL_J + 15) / 16; j++) {
                    for (int k = 0; k < (TOTAL_K + 15) / 16; k++) {
                        for (int kk_ii_jj = 0; kk_ii_jj < 64; kk_ii_jj++) {
#pragma unroll
                            for (int iii = 0; iii < 4; iii++) {
#pragma unroll
                                for (int jjj = 0; jjj < 4; jjj++) {
                                    Z[jjj][iii] = Z_shreg[15][jjj][iii];
#pragma unroll
                                    for (int l1 = 0; l1 < 15; l1++) {
                                        Z_shreg[15 - l1][jjj][iii] = Z_shreg[14 - l1][jjj][iii];
                                    }
                                    Z_shreg[0][jjj][iii] = Z[jjj][iii];
                                }
                            }
                            if (i < (TOTAL_I + 15) / 16) {
                                bFeeder_channel_array = bFeeder_channel::read();
                                aFeeder_channel_array = aFeeder_channel::read();
                            }
#pragma unroll
                            for (int iii = 0; iii < 4; iii++) {
#pragma unroll
                                for (int jjj = 0; jjj < 4; jjj++) {
                                    X_shreg[iii] = jjj == 0 ? aFeeder_channel_array.s[iii] : X_shreg[iii];
                                    X_shreg[iii] = float4{
                                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(X_shreg[iii][0])),
                                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(X_shreg[iii][1])),
                                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(X_shreg[iii][2])),
                                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(X_shreg[iii][3]))};
                                    Y_shreg[jjj] = iii == 0 ? bFeeder_channel_array.s[jjj] : Y_shreg[jjj];
                                    Y_shreg[jjj] = float4{
                                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(Y_shreg[jjj][0])),
                                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(Y_shreg[jjj][1])),
                                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(Y_shreg[jjj][2])),
                                        sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(Y_shreg[jjj][3]))};
                                    Z_shreg_ = k == 0 && kk_ii_jj / 16 == 0 ? 0.0f : sycl::ext::intel::fpga_reg(Z_shreg[0][jjj][iii]);
#pragma unroll
                                    for (int kkk = 0; kkk < 4; kkk++) {
                                        Z_shreg_ = Z_shreg_ + X_shreg[iii][kkk] * Y_shreg[jjj][kkk];
                                        if (kkk == 3) {
                                            Z_shreg_ = sycl::ext::intel::fpga_reg(Z_shreg_);
                                        }
                                    }
                                    Z_shreg[0][jjj][iii] = Z_shreg_;
#pragma unroll
                                    for (int kkk = 0; kkk < 4; kkk++) {
                                        if (kkk == 3 && kk_ii_jj / 16 == 3 && k == (TOTAL_K + -1) / 16) {
                                            Z_pipe_shreg[jjj][iii * 16] = Z_shreg[0][jjj][iii];
                                        }
                                    }
                                }
                            }
                            if (kk_ii_jj % 4 == 0 && kk_ii_jj % 16 / 4 == 0 && k == (TOTAL_K + -1) / 16 && kk_ii_jj / 16 == 3 && i < (TOTAL_I + 15) / 16) {
                                Z_pipe_base = Z_pipe_iter;
                            }
                            float4 Product_channel_;
#pragma unroll
                            for (int b_62 = 0; b_62 < 4; b_62++) {
                                Product_channel_[b_62] = Z_pipe_shreg[b_62][0];
#pragma unroll
                                for (int b_62_dummy = 0; b_62_dummy < 4; b_62_dummy++) {
                                    Product_channel_[b_62_dummy] = sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(Product_channel_[b_62_dummy]));
                                }
                            }
                            if (Z_pipe_iter < Z_pipe_base + 64) {
                                Product_channel::write(Product_channel_);
                            }
#pragma unroll
                            for (int b_63 = 0; b_63 < 4; b_63++) {
#pragma unroll
                                for (int p_31 = 0; p_31 < 3; p_31++) {
#pragma unroll
                                    for (int l_31 = 0; l_31 < 15; l_31++) {
                                        Z_pipe_shreg[b_63][p_31 * 16 + l_31] = Z_pipe_shreg[b_63][p_31 * 16 + l_31 + 1];
                                    }
                                    Z_pipe_shreg[b_63][p_31 * 16 + 15] = sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(Z_pipe_shreg[b_63][p_31 * 16 + 16]));
                                }
                            }
                            Z_pipe_iter = Z_pipe_iter + 1;
                        }
                    }
                }
            }
        });
    }));

    std::cout << "memcpy matrix C host->device" << std::endl;
    q.memcpy(serialized_C_device, serialized_C, num_elem_C * sizeof(float))
        .wait();
    // kernel_cLoader
    std::cout << "kernel_cLoader" << std::endl;
    oneapi_kernel_events.push_back(q.submit([&](sycl::handler &h) {
        h.single_task<class kernel_cLoader>([=]() [[intel::kernel_args_restrict]] {
            device_ptr<float> serialized_C_d(serialized_C_device);
            int addr_temp;
            addr_temp = 0;
            for (int i = 0; i < (TOTAL_I + 15) / 16; i++) {
                for (int j = 0; j < (TOTAL_J + 15) / 16; j++) {
                    for (int iii_ii_jj = 0; iii_ii_jj < 64; iii_ii_jj++) {
                        if (p3 != 0.0f) {
                            cLoader_channel::write(float4{
                                serialized_C_d[addr_temp * 4 + 0],
                                serialized_C_d[addr_temp * 4 + 1],
                                serialized_C_d[addr_temp * 4 + 2],
                                serialized_C_d[addr_temp * 4 + 3]});
                        }
                        addr_temp = addr_temp + 1;
                    }
                }
            }
        });
    }));

    // kernel_Out
    std::cout << "kernel_Out" << std::endl;
    oneapi_kernel_events.push_back(q.submit([&](sycl::handler &h) {
        h.single_task<class kernel_Out>([=]() {
            float Z[4][4];
            float4 Add_shreg;
            for (int i = 0; i < (TOTAL_I + 15) / 16; i++) {
                for (int j = 0; j < (TOTAL_J + 15) / 16; j++) {
                    for (int iii_ii_jj = 0; iii_ii_jj < 64; iii_ii_jj++) {
                        Add_shreg = (p3 == 0.0f ? float4{0.0f} : cLoader_channel::read() * float4{p3}) + Product_channel::read() * float4{p2};
                        Out_channel::write(Add_shreg);
                    }
                }
            }
        }); //  h.single_task kernel_Out_class
    }));

    // kernel_unloader
    std::cout << "kernel_unloader" << std::endl;
    oneapi_kernel_events.push_back(q.submit([&](sycl::handler &h) {
        h.single_task<class kernel_unloader>(
            [=]() [[intel::kernel_args_restrict]] {
                device_ptr<float> serialized_reuslt_d(serialized_reuslt_device);
                int addr_temp;
                addr_temp = 0;
                for (int i = 0; i < (TOTAL_I + 15) / 16; i++) {
                    for (int j = 0; j < (TOTAL_J + 15) / 16; j++) {
                        for (int iii_ii_jj = 0; iii_ii_jj < 64; iii_ii_jj++) {
                            auto _D6 = Out_channel::read();
                            serialized_reuslt_d[addr_temp * 4 + 0] = _D6[0];
                            serialized_reuslt_d[addr_temp * 4 + 1] = _D6[1];
                            serialized_reuslt_d[addr_temp * 4 + 2] = _D6[2];
                            serialized_reuslt_d[addr_temp * 4 + 3] = _D6[3];
                            addr_temp = addr_temp + 1;
                        }
                    }
                }
            });
    }));

    for (unsigned int i = 0; i < oneapi_kernel_events.size(); i++) {
        oneapi_kernel_events.at(i).wait();
        std::cout << i << "finished" << std::endl;
    };

    if (oneapi_kernel_events.size() > 0) {
        double k_earliest_start_time =
            oneapi_kernel_events.at(0)
                .get_profiling_info<
                    sycl::info::event_profiling::command_start>();
        double k_latest_end_time =
            oneapi_kernel_events.at(0)
                .get_profiling_info<sycl::info::event_profiling::command_end>();
        for (unsigned i = 1; i < oneapi_kernel_events.size(); i++) {
            double tmp_start =
                oneapi_kernel_events.at(i)
                    .get_profiling_info<
                        sycl::info::event_profiling::command_start>();
            double tmp_end =
                oneapi_kernel_events.at(i)
                    .get_profiling_info<
                        sycl::info::event_profiling::command_end>();
            if (tmp_start < k_earliest_start_time) {
                k_earliest_start_time = tmp_start;
            }
            if (tmp_end > k_latest_end_time) {
                k_latest_end_time = tmp_end;
            }
        }
        // Get time in ns
        double events_time = (k_latest_end_time - k_earliest_start_time);
        printf("  Time: %.5f ns\n", events_time);
        printf("  Throughput: %.2f GFLOPS\n",
               (double)2.0 * (TOTAL_K) * (double)(TOTAL_I) * (double)(TOTAL_J) /
                   events_time);
    }

    std::cout << "memcpy result device->host" << std::endl;
    q.memcpy(serialized_reuslt, serialized_reuslt_device, num_elem_C * sizeof(float))
        .wait();

    // kernel_C_deserializer
    std::cout << "kernel_C_deserializer" << std::endl;
    addr = 0;
    for (size_t i = 0; i < I; i++)
        for (size_t j = 0; j < J; j++)
            for (size_t iii = 0; iii < III; iii++)
                for (size_t ii = 0; ii < II; ii++)
                    for (size_t jj = 0; jj < JJ; jj++)
                        for (size_t jjj = 0; jjj < JJJ; jjj++) {
                            size_t total_i = iii + III * ii + III * II * i;
                            size_t total_j = jjj + JJJ * jj + JJJ * JJ * j;
                            result[total_j + total_i * TOTAL_J] =
                                serialized_reuslt[addr++];
                        }

    free(serialized_A);
    free(serialized_B);
    free(serialized_C);
    free(serialized_reuslt);
    sycl::free(serialized_A_device, q);
    sycl::free(serialized_B_device, q);
    sycl::free(serialized_C_device, q);
    sycl::free(serialized_reuslt_device, q);
}
