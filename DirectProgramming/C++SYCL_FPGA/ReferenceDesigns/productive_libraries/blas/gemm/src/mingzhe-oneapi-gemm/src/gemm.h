#pragma once

#include <iostream>
#include <list>
#include <stdlib.h>
#include <vector>

#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#ifdef FPGA_EMULATOR
#define KKK 16
#define JJJ 8
#define III 10
#define JJ 32
#define II 32
#define KK 32
#define K 2
#define J 2
#define I 2
#else
#define KKK 16
#define JJJ 8
#define III 10
#define JJ 32
#define II 32
#define KK 32
#define K 32
#define J 32
#define I 32
#endif

#define ADD_INT64_T_SUFFIX(x) x##l
#define ADD_UINT64_T_SUFFIX(x) x##ul

#ifdef __SYCL_DEVICE_ONLY__
#define CL_CONSTANT __attribute__((opencl_constant))
#else
#define CL_CONSTANT
#endif

using namespace sycl;

#define PRINTF(format, ...)                                                    \
    {                                                                          \
        static const CL_CONSTANT char _format[] = format;                      \
        ext::oneapi::experimental::printf(_format, ##__VA_ARGS__);             \
    }

constexpr size_t TOTAL_I = III * II * I;
constexpr size_t TOTAL_J = JJJ * JJ * J;
constexpr size_t TOTAL_K = KKK * KK * K;
constexpr size_t num_elem_A = TOTAL_K * TOTAL_I;
constexpr size_t num_elem_B = TOTAL_J * TOTAL_K;
constexpr size_t num_elem_C = TOTAL_J * TOTAL_I;

using _aLoader_channel =
    sycl::ext::intel::pipe<class _aLoader_channel_pipe, float16, 256>;
typedef struct {
    float16 s[10];
} _aFeeder_channel_array_t;
using _aFeeder_channel = sycl::ext::intel::pipe<class _aFeeder_channel_pipe,
                                                _aFeeder_channel_array_t, 256>;
using _bLoader_channel =
    sycl::ext::intel::pipe<class _bLoader_channel_pipe, float16, 256>;
typedef struct {
    float16 s[8];
} _bFeeder_channel_array_t;
using _bFeeder_channel = sycl::ext::intel::pipe<class _bFeeder_channel_pipe,
                                                _bFeeder_channel_array_t, 256>;
using _Out_channel =
    sycl::ext::intel::pipe<class _Out_channel_pipe, float8, 256>;

void gemm(float *A, float *B, float *C, sycl::queue &q) {
    std::vector<sycl::event> oneapi_kernel_events;
    float *serialized_A = (float *)malloc(num_elem_A * sizeof(float));
    float *serialized_B = (float *)malloc(num_elem_B * sizeof(float));
    float *serialized_C = (float *)malloc(num_elem_C * sizeof(float));
    float *serialized_A_device = sycl::malloc_device<float>(num_elem_A, q);
    float *serialized_B_device = sycl::malloc_device<float>(num_elem_B, q);
    float *serialized_C_device = sycl::malloc_device<float>(num_elem_C, q);

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

    std::cout << "memcpy matrix A host->device" << std::endl;
    q.memcpy(serialized_A_device, serialized_A, num_elem_A * sizeof(float))
        .wait();
    std::cout << "memcpy matrix B host->device" << std::endl;
    q.memcpy(serialized_B_device, serialized_B, num_elem_B * sizeof(float))
        .wait();

    // kernel_aLoader
    std::cout << "kernel_aLoader" << std::endl;
    oneapi_kernel_events.push_back(q.submit([&](sycl::handler &h) {
        h.single_task<class kernel_aLoader>(
            [=]() [[intel::kernel_args_restrict]] {
                device_ptr<float> serialized_A_d(serialized_A_device);
                int _addr_temp;
                _addr_temp = 0;
                int _0 = TOTAL_I / 320;
                for (int _aLoader_s0_i = 0; _aLoader_s0_i < 0 + _0;
                     _aLoader_s0_i++) {
                    int _1 = TOTAL_J >> 8;
                    for (int _aLoader_s0_j = 0; _aLoader_s0_j < 0 + _1;
                         _aLoader_s0_j++) {
                        int _2 = TOTAL_K >> 9;
                        for (int _aLoader_s0_k = 0; _aLoader_s0_k < 0 + _2;
                             _aLoader_s0_k++) {
                            for (int _aLoader_s0_kk_ii_iii = 0;
                                 _aLoader_s0_kk_ii_iii < 0 + 10240;
                                 _aLoader_s0_kk_ii_iii++) {
                                int _3 = _addr_temp;
                                int _4 = TOTAL_J >> 8;
                                int _5 = TOTAL_K >> 9;
                                int _6 = _4 * _5;
                                int _7 = _6 * 10240;
                                int _8 = _3 / _7;
                                int _9 = _8 * _5;
                                int _10 = _9 * 10240;
                                int _11 = _5 * 10240;
                                int _12 = _3 % _11;
                                int _13 = _10 + _12;
                                int _14 = _13 * 16;
                                float16 _15;
#pragma unroll
                                for (int vec_idx = 0; vec_idx < 16; vec_idx++) {
                                    _15[vec_idx] =
                                        serialized_A_d[_14 + vec_idx];
                                }
                                _aLoader_channel::write(_15);
                                (void)_15;
                                int _16 = _3 + 1;
                                _addr_temp = _16;
                            } // for _aLoader_s0_kk_ii_iii
                        }     // for _aLoader_s0_k
                    }         // for _aLoader_s0_j
                }             // for _aLoader_s0_i
            });
    }));

    // kernel_aFeeder
    std::cout << "kernel_aFeeder" << std::endl;
    oneapi_kernel_events.push_back(q.submit([&](sycl::handler &h) {
        h.single_task<class kernel_aFeeder>([=]() {
            _aFeeder_channel_array_t _aFeeder_channel_array;
            float16 _aFeeder_value_shreg;
            uint _aFeeder_time_stamp_shreg;
            float16 _aFeeder_in_v_temp;
            uint _aFeeder_cycle_temp;
            [[intel::fpga_memory(), intel::numbanks(16), intel::singlepump,
              intel::simple_dual_port]] float16 _aFeeder_DB_0_ibuffer[2][32][32]
                                                                     [16];
#pragma unroll
            for (int _aFeeder_s0_jjj_init = 0; _aFeeder_s0_jjj_init < 0 + 8;
                 _aFeeder_s0_jjj_init++) {
                bool _17 = _aFeeder_s0_jjj_init == 0;
                if (_17) {
                    uint _18 = (uint)(ADD_UINT64_T_SUFFIX(22528));
                    _aFeeder_cycle_temp = _18;
                } // if _17
            }     // for _aFeeder_s0_jjj_init
            int _19 = TOTAL_K >> 9;
            int _20 = TOTAL_I / 320;
            int _21 = TOTAL_J >> 8;
            int _22 = _20 * _21;
            int _23 = _19 * _22;
            int _24 = _23 * 32768;
            int _25 = _24 + 32768;
            for (int _aFeeder_s0_outermost_loop = 0;
                 _aFeeder_s0_outermost_loop < 0 + _25;
                 _aFeeder_s0_outermost_loop++) {
                uint _26 = (uint)(ADD_UINT64_T_SUFFIX(22528));
                uint _27 = _aFeeder_cycle_temp;
                uint _28 = (uint)(ADD_UINT64_T_SUFFIX(32767));
                uint _29 = _27 & _28;
                bool _30 = _26 <= _29;
                uint _31 = (uint)(ADD_UINT64_T_SUFFIX(15));
                uint _32 = _27 >> _31;
                int _33 = (int)(_32);
                int _34 = TOTAL_K >> 9;
                int _35 = TOTAL_I / 320;
                int _36 = TOTAL_J >> 8;
                int _37 = _35 * _36;
                int _38 = _34 * _37;
                bool _39 = _33 < _38;
                bool _40 = _30 && _39;
                if (_40) {
                    float16 __41 = _aLoader_channel::read();
                    _aFeeder_in_v_temp = __41;
                } // if _40
#pragma unroll
                for (int _aFeeder_s0_buf = 0; _aFeeder_s0_buf < 0 + 10;
                     _aFeeder_s0_buf++) {
                    bool _42 = _aFeeder_s0_buf == 0;
                    if (_42) {
                        float16 _43 = _aFeeder_in_v_temp;
                        _aFeeder_value_shreg = _43;
                        (void)_43;
                        uint _44 = _aFeeder_cycle_temp;
                        _aFeeder_time_stamp_shreg = _44;
                        (void)_44;
                    } // if _42
                    else {
                        float16 _46 = _aFeeder_value_shreg;
                        _aFeeder_value_shreg = _46;
                        (void)_46;
                        uint _48 = _aFeeder_time_stamp_shreg;
                        _aFeeder_time_stamp_shreg = _48;
                        (void)_48;
                    } // if _42 else
                    float16 _50 = _aFeeder_value_shreg;
                    float16 _51;
#pragma unroll
                    for (int reg_idx = 0; reg_idx < 16; reg_idx++) {
                        _51[reg_idx] = sycl::ext::intel::fpga_reg(
                            sycl::ext::intel::fpga_reg(_50[reg_idx]));
                    }
                    _aFeeder_value_shreg = _51;
                    (void)_51;
                    uint _53 = _aFeeder_time_stamp_shreg;
                    uint _54 = sycl::ext::intel::fpga_reg(
                        sycl::ext::intel::fpga_reg(_53));
                    _aFeeder_time_stamp_shreg = _54;
                    (void)_54;
                    uint _55 = (uint)(ADD_UINT64_T_SUFFIX(22528));
                    uint _57 = _aFeeder_time_stamp_shreg;
                    uint _58 = (uint)(ADD_UINT64_T_SUFFIX(32767));
                    uint _59 = _57 & _58;
                    bool _60 = _55 <= _59;
                    if (_60) {
                        uint _62 = _aFeeder_time_stamp_shreg;
                        uint _63 = (uint)(ADD_UINT64_T_SUFFIX(32767));
                        uint _64 = _62 & _63;
                        uint _65 = (uint)(ADD_UINT64_T_SUFFIX(22528));
                        uint _66 = _64 - _65;
                        uint _67 = (uint)(ADD_UINT64_T_SUFFIX(10));
                        uint _68 = _66 % _67;
                        int _69 = (int)(_68);
                        bool _70 = _aFeeder_s0_buf == _69;
                        if (_70) {
                            float16 _72 = _aFeeder_value_shreg;
                            uint _74 = _aFeeder_time_stamp_shreg;
                            uint _75 = (uint)(ADD_UINT64_T_SUFFIX(15));
                            uint _76 = _74 >> _75;
                            uint _77 = (uint)(ADD_UINT64_T_SUFFIX(1));
                            uint _78 = _76 & _77;
                            bool _79 = (bool)(_78);
                            uint _81 = (uint)(ADD_UINT64_T_SUFFIX(32767));
                            uint _82 = _74 & _81;
                            uint _83 = (uint)(ADD_UINT64_T_SUFFIX(22528));
                            uint _84 = _82 - _83;
                            int _85 = (int)(_84);
                            int _86 = _85 / 320;
                            int _88 = _85 / 10;
                            int _89 = _88 & 31;
                            _aFeeder_DB_0_ibuffer[_79][_86][_89]
                                                 [_aFeeder_s0_buf] = _72;
                        } // if _70
                    }     // if _60
                    uint _91 = _aFeeder_time_stamp_shreg;
                    uint _92 = (uint)(ADD_UINT64_T_SUFFIX(15));
                    uint _93 = _91 >> _92;
                    int _94 = (int)(_93);
                    int _95 = TOTAL_K >> 9;
                    int _96 = TOTAL_I / 320;
                    int _97 = TOTAL_J >> 8;
                    int _98 = _96 * _97;
                    int _99 = _95 * _98;
                    bool _100 = _94 <= _99;
                    uint _101 = (uint)(ADD_UINT64_T_SUFFIX(0));
                    bool _103 = _101 < _93;
                    bool _104 = _100 && _103;
                    if (_104) {
                        uint _106 = _aFeeder_time_stamp_shreg;
                        uint _107 = (uint)(ADD_UINT64_T_SUFFIX(15));
                        uint _108 = _106 >> _107;
                        uint _109 = (uint)(ADD_UINT64_T_SUFFIX(1));
                        uint _110 = _108 & _109;
                        bool _111 = (bool)(_110);
                        bool _112 = !(_111);
                        uint _114 = (uint)(ADD_UINT64_T_SUFFIX(32767));
                        uint _115 = _106 & _114;
                        int _116 = (int)(_115);
                        int _117 = _116 >> 10;
                        int _119 = _116 >> 5;
                        int _120 = _119 & 31;
                        float16 _121 = _aFeeder_DB_0_ibuffer[_112][_117][_120]
                                                            [_aFeeder_s0_buf];
                        _aFeeder_channel_array.s[_aFeeder_s0_buf] = _121;
                        (void)_aFeeder_s0_buf;
                    } // if _104
                }     // for _aFeeder_s0_buf
                uint _123 = _aFeeder_time_stamp_shreg;
                uint _124 = (uint)(ADD_UINT64_T_SUFFIX(15));
                uint _125 = _123 >> _124;
                int _126 = (int)(_125);
                int _127 = TOTAL_K >> 9;
                int _128 = TOTAL_I / 320;
                int _129 = TOTAL_J >> 8;
                int _130 = _128 * _129;
                int _131 = _127 * _130;
                bool _132 = _126 <= _131;
                uint _133 = (uint)(ADD_UINT64_T_SUFFIX(0));
                bool _135 = _133 < _125;
                bool _136 = _132 && _135;
                if (_136) {
                    _aFeeder_channel::write(_aFeeder_channel_array);
                    (void)_aFeeder_channel_array;
                } // if _136
                uint _137 = _aFeeder_cycle_temp;
                uint _138 = (uint)(ADD_UINT64_T_SUFFIX(1));
                uint _139 = _137 + _138;
                _aFeeder_cycle_temp = _139;
            } // for _aFeeder_s0_outermost_loop
        });
    }));

    // kernel_bLoader
    std::cout << "kernel_bLoader" << std::endl;
    oneapi_kernel_events.push_back(q.submit([&](sycl::handler &h) {
        h.single_task<class kernel_bLoader>(
            [=]() [[intel::kernel_args_restrict]] {
                device_ptr<float> serialized_B_d(serialized_B_device);
                int _addr_temp;
                _addr_temp = 0;
                int _140 = TOTAL_I / 320;
                for (int _bLoader_s0_i = 0; _bLoader_s0_i < 0 + _140;
                     _bLoader_s0_i++) {
                    int _141 = TOTAL_J >> 8;
                    for (int _bLoader_s0_j = 0; _bLoader_s0_j < 0 + _141;
                         _bLoader_s0_j++) {
                        int _142 = TOTAL_K >> 9;
                        for (int _bLoader_s0_k = 0; _bLoader_s0_k < 0 + _142;
                             _bLoader_s0_k++) {
                            for (int _bLoader_s0_kk_jj_jjj = 0;
                                 _bLoader_s0_kk_jj_jjj < 0 + 8192;
                                 _bLoader_s0_kk_jj_jjj++) {
                                int _143 = _addr_temp;
                                int _144 = TOTAL_J >> 8;
                                int _145 = TOTAL_K >> 9;
                                int _146 = _144 * _145;
                                int _147 = _146 * 8192;
                                int _148 = _143 % _147;
                                int _149 = _148 * 16;
                                float16 _150;
#pragma unroll
                                for (int vec_idx = 0; vec_idx < 16; vec_idx++) {
                                    _150[vec_idx] =
                                        serialized_B_d[_149 + vec_idx];
                                }
                                _bLoader_channel::write(_150);
                                (void)_150;
                                int _151 = _143 + 1;
                                _addr_temp = _151;
                            } // for _bLoader_s0_kk_jj_jjj
                        }     // for _bLoader_s0_k
                    }         // for _bLoader_s0_j
                }             // for _bLoader_s0_i
            });
    }));

    // kernel_bFeeder
    std::cout << "kernel_bFeeder" << std::endl;
    oneapi_kernel_events.push_back(q.submit([&](sycl::handler &h) {
        h.single_task<class kernel_bFeeder>([=]() {
            _bFeeder_channel_array_t _bFeeder_channel_array;
            float16 _bFeeder_value_shreg;
            uint _bFeeder_time_stamp_shreg;
            float16 _bFeeder_in_v_temp;
            uint _bFeeder_cycle_temp;
            [[intel::fpga_memory(), intel::numbanks(8), intel::singlepump,
              intel::simple_dual_port]] float16 _bFeeder_DB_0_ibuffer[2][32][32]
                                                                     [8];
#pragma unroll
            for (int _bFeeder_s0_iii_init = 0; _bFeeder_s0_iii_init < 0 + 10;
                 _bFeeder_s0_iii_init++) {
                bool _152 = _bFeeder_s0_iii_init == 0;
                if (_152) {
                    uint _153 = (uint)(ADD_UINT64_T_SUFFIX(24576));
                    _bFeeder_cycle_temp = _153;
                } // if _152
            }     // for _bFeeder_s0_iii_init
            int _154 = TOTAL_K >> 9;
            int _155 = TOTAL_I / 320;
            int _156 = TOTAL_J >> 8;
            int _157 = _155 * _156;
            int _158 = _154 * _157;
            int _159 = _158 * 32768;
            int _160 = _159 + 32768;
            for (int _bFeeder_s0_outermost_loop = 0;
                 _bFeeder_s0_outermost_loop < 0 + _160;
                 _bFeeder_s0_outermost_loop++) {
                uint _161 = (uint)(ADD_UINT64_T_SUFFIX(24576));
                uint _162 = _bFeeder_cycle_temp;
                uint _163 = (uint)(ADD_UINT64_T_SUFFIX(32767));
                uint _164 = _162 & _163;
                bool _165 = _161 <= _164;
                uint _166 = (uint)(ADD_UINT64_T_SUFFIX(15));
                uint _167 = _162 >> _166;
                int _168 = (int)(_167);
                int _169 = TOTAL_K >> 9;
                int _170 = TOTAL_I / 320;
                int _171 = TOTAL_J >> 8;
                int _172 = _170 * _171;
                int _173 = _169 * _172;
                bool _174 = _168 < _173;
                bool _175 = _165 && _174;
                if (_175) {
                    float16 __176 = _bLoader_channel::read();
                    _bFeeder_in_v_temp = __176;
                } // if _175
#pragma unroll
                for (int _bFeeder_s0_buf = 0; _bFeeder_s0_buf < 0 + 8;
                     _bFeeder_s0_buf++) {
                    bool _177 = _bFeeder_s0_buf == 0;
                    if (_177) {
                        float16 _178 = _bFeeder_in_v_temp;
                        _bFeeder_value_shreg = _178;
                        (void)_178;
                        uint _179 = _bFeeder_cycle_temp;
                        _bFeeder_time_stamp_shreg = _179;
                        (void)_179;
                    } // if _177
                    else {
                        float16 _181 = _bFeeder_value_shreg;
                        _bFeeder_value_shreg = _181;
                        (void)_181;
                        uint _183 = _bFeeder_time_stamp_shreg;
                        _bFeeder_time_stamp_shreg = _183;
                        (void)_183;
                    } // if _177 else
                    float16 _185 = _bFeeder_value_shreg;
                    float16 _186;
#pragma unroll
                    for (int reg_idx = 0; reg_idx < 16; reg_idx++) {
                        _186[reg_idx] = sycl::ext::intel::fpga_reg(
                            sycl::ext::intel::fpga_reg(_185[reg_idx]));
                    }
                    _bFeeder_value_shreg = _186;
                    (void)_186;
                    uint _188 = _bFeeder_time_stamp_shreg;
                    uint _189 = sycl::ext::intel::fpga_reg(
                        sycl::ext::intel::fpga_reg(_188));
                    _bFeeder_time_stamp_shreg = _189;
                    (void)_189;
                    uint _190 = (uint)(ADD_UINT64_T_SUFFIX(24576));
                    uint _192 = _bFeeder_time_stamp_shreg;
                    uint _193 = (uint)(ADD_UINT64_T_SUFFIX(32767));
                    uint _194 = _192 & _193;
                    bool _195 = _190 <= _194;
                    if (_195) {
                        uint _197 = _bFeeder_time_stamp_shreg;
                        uint _198 = (uint)(ADD_UINT64_T_SUFFIX(32767));
                        uint _199 = _197 & _198;
                        uint _200 = (uint)(ADD_UINT64_T_SUFFIX(24576));
                        uint _201 = _199 - _200;
                        uint _202 = (uint)(ADD_UINT64_T_SUFFIX(7));
                        uint _203 = _201 & _202;
                        int _204 = (int)(_203);
                        bool _205 = _bFeeder_s0_buf == _204;
                        if (_205) {
                            float16 _207 = _bFeeder_value_shreg;
                            uint _209 = _bFeeder_time_stamp_shreg;
                            uint _210 = (uint)(ADD_UINT64_T_SUFFIX(15));
                            uint _211 = _209 >> _210;
                            uint _212 = (uint)(ADD_UINT64_T_SUFFIX(1));
                            uint _213 = _211 & _212;
                            bool _214 = (bool)(_213);
                            uint _216 = (uint)(ADD_UINT64_T_SUFFIX(32767));
                            uint _217 = _209 & _216;
                            uint _218 = (uint)(ADD_UINT64_T_SUFFIX(24576));
                            uint _219 = _217 - _218;
                            int _220 = (int)(_219);
                            int _221 = _220 >> 8;
                            int _223 = _220 >> 3;
                            int _224 = _223 & 31;
                            _bFeeder_DB_0_ibuffer[_214][_221][_224]
                                                 [_bFeeder_s0_buf] = _207;
                        } // if _205
                    }     // if _195
                    uint _226 = _bFeeder_time_stamp_shreg;
                    uint _227 = (uint)(ADD_UINT64_T_SUFFIX(15));
                    uint _228 = _226 >> _227;
                    int _229 = (int)(_228);
                    int _230 = TOTAL_K >> 9;
                    int _231 = TOTAL_I / 320;
                    int _232 = TOTAL_J >> 8;
                    int _233 = _231 * _232;
                    int _234 = _230 * _233;
                    bool _235 = _229 <= _234;
                    uint _236 = (uint)(ADD_UINT64_T_SUFFIX(0));
                    bool _238 = _236 < _228;
                    bool _239 = _235 && _238;
                    if (_239) {
                        uint _241 = _bFeeder_time_stamp_shreg;
                        uint _242 = (uint)(ADD_UINT64_T_SUFFIX(15));
                        uint _243 = _241 >> _242;
                        uint _244 = (uint)(ADD_UINT64_T_SUFFIX(1));
                        uint _245 = _243 & _244;
                        bool _246 = (bool)(_245);
                        bool _247 = !(_246);
                        uint _249 = (uint)(ADD_UINT64_T_SUFFIX(32767));
                        uint _250 = _241 & _249;
                        int _251 = (int)(_250);
                        int _252 = _251 >> 10;
                        int _254 = _251 & 31;
                        float16 _255 = _bFeeder_DB_0_ibuffer[_247][_252][_254]
                                                            [_bFeeder_s0_buf];
                        _bFeeder_channel_array.s[_bFeeder_s0_buf] = _255;
                        (void)_bFeeder_s0_buf;
                    } // if _239
                }     // for _bFeeder_s0_buf
                uint _257 = _bFeeder_time_stamp_shreg;
                uint _258 = (uint)(ADD_UINT64_T_SUFFIX(15));
                uint _259 = _257 >> _258;
                int _260 = (int)(_259);
                int _261 = TOTAL_K >> 9;
                int _262 = TOTAL_I / 320;
                int _263 = TOTAL_J >> 8;
                int _264 = _262 * _263;
                int _265 = _261 * _264;
                bool _266 = _260 <= _265;
                uint _267 = (uint)(ADD_UINT64_T_SUFFIX(0));
                bool _269 = _267 < _259;
                bool _270 = _266 && _269;
                if (_270) {
                    _bFeeder_channel::write(_bFeeder_channel_array);
                    (void)_bFeeder_channel_array;
                } // if _270
                uint _271 = _bFeeder_cycle_temp;
                uint _272 = (uint)(ADD_UINT64_T_SUFFIX(1));
                uint _273 = _271 + _272;
                _bFeeder_cycle_temp = _273;
            } // for _bFeeder_s0_outermost_loop
        });
    }));

    // kernel_Out
    std::cout << "kernel_Out" << std::endl;
    oneapi_kernel_events.push_back(q.submit([&](sycl::handler &h) {
        h.single_task<class kernel_Out>([=]() {
            _bFeeder_channel_array_t _bFeeder_channel_array;
            _aFeeder_channel_array_t _aFeeder_channel_array;
            // produce Z
            float _Z_shreg[1024][8][10];
            float _Z_pipe_shreg[8][9217];
            // produce Y
            float16 _Y_shreg[8];
            float _Z_temp[8][10];
            // produce X
            float16 _X_shreg[10];
            float _Z_shreg_temp;
            int _Z_pipe_iter_temp;
            int _Z_pipe_base_temp;
            _Z_pipe_iter_temp = 10240;
            _Z_pipe_base_temp = 0;
            int _274 = TOTAL_K >> 9;
            int _275 = TOTAL_I / 320;
            int _276 = TOTAL_J >> 8;
            int _277 = _275 * _276;
            int _278 = _274 * _277;
            int _279 = _278 + 1;
            for (int _X_s0_i_j_k = 0; _X_s0_i_j_k < 0 + _279; _X_s0_i_j_k++) {
                for (int _X_s0_kk_ii_jj = 0; _X_s0_kk_ii_jj < 0 + 32768;
                     _X_s0_kk_ii_jj++) {
#pragma unroll
                    for (int _dummy__1_s0_iii = 0; _dummy__1_s0_iii < 0 + 10;
                         _dummy__1_s0_iii++) {
#pragma unroll
                        for (int _dummy_s0_jjj = 0; _dummy_s0_jjj < 0 + 8;
                             _dummy_s0_jjj++) {
                            float _281 =
                                _Z_shreg[1023][_dummy_s0_jjj][_dummy__1_s0_iii];
                            _Z_temp[_dummy_s0_jjj][_dummy__1_s0_iii] = _281;
#pragma unroll
                            for (int _dummy__2_s0_l1 = 0;
                                 _dummy__2_s0_l1 < 0 + 1023;
                                 _dummy__2_s0_l1++) {
                                int _282 = 1023 - _dummy__2_s0_l1;
                                int _283 = 1022 - _dummy__2_s0_l1;
                                float _285 = _Z_shreg[_283][_dummy_s0_jjj]
                                                     [_dummy__1_s0_iii];
                                _Z_shreg[_282][_dummy_s0_jjj]
                                        [_dummy__1_s0_iii] = _285;
                                (void)_285;
                            } // for _dummy__2_s0_l1
                            float _286 =
                                _Z_temp[_dummy_s0_jjj][_dummy__1_s0_iii];
                            _Z_shreg[0][_dummy_s0_jjj][_dummy__1_s0_iii] = _286;
                            (void)_286;
                        } // for _dummy_s0_jjj
                    }     // for _dummy__1_s0_iii
                    int _287 = TOTAL_K >> 9;
                    int _288 = TOTAL_I / 320;
                    int _289 = TOTAL_J >> 8;
                    int _290 = _288 * _289;
                    int _291 = _287 * _290;
                    bool _292 = _X_s0_i_j_k < _291;
                    if (_292) {
                        _bFeeder_channel_array_t __293 =
                            _bFeeder_channel::read();
                        _bFeeder_channel_array = __293;
                        (void)__293;
                        _aFeeder_channel_array_t __294 =
                            _aFeeder_channel::read();
                        _aFeeder_channel_array = __294;
                        (void)__294;
                    } // if _292
#pragma unroll
                    for (int _X_s0_iii = 0; _X_s0_iii < 0 + 10; _X_s0_iii++) {
#pragma unroll
                        for (int _X_s0_jjj = 0; _X_s0_jjj < 0 + 8;
                             _X_s0_jjj++) {
                            float16 _295;
                            bool _296 = _X_s0_jjj == 0;
                            if (_296) {
                                float16 __297 =
                                    _aFeeder_channel_array.s[_X_s0_iii];
                                _295 = __297;
                            } // if _296
                            else {
                                float16 _299 = _X_shreg[_X_s0_iii];
                                _295 = _299;
                            } // if _296 else
                            float16 _300 = _295;
                            _X_shreg[_X_s0_iii] = _300;
                            (void)_300;
                            float16 _302 = _X_shreg[_X_s0_iii];
                            float16 _303;
#pragma unroll
                            for (int reg_idx = 0; reg_idx < 16; reg_idx++) {
                                _303[reg_idx] = sycl::ext::intel::fpga_reg(
                                    sycl::ext::intel::fpga_reg(_302[reg_idx]));
                            }

                            _X_shreg[_X_s0_iii] = _303;
                            (void)_303;
                            float16 _304;
                            bool _305 = _X_s0_iii == 0;
                            if (_305) {
                                float16 __306 =
                                    _bFeeder_channel_array.s[_X_s0_jjj];
                                _304 = __306;
                            } // if _305
                            else {
                                float16 _308 = _Y_shreg[_X_s0_jjj];
                                _304 = _308;
                            } // if _305 else
                            float16 _309 = _304;
                            _Y_shreg[_X_s0_jjj] = _309;
                            (void)_309;
                            float16 _311 = _Y_shreg[_X_s0_jjj];
                            float16 _312;
#pragma unroll
                            for (int reg_idx = 0; reg_idx < 16; reg_idx++) {
                                _312[reg_idx] = sycl::ext::intel::fpga_reg(
                                    sycl::ext::intel::fpga_reg(_311[reg_idx]));
                            }

                            _Y_shreg[_X_s0_jjj] = _312;
                            (void)_312;
                            float _313;
                            int _314 = TOTAL_K >> 9;
                            int _315 = _X_s0_i_j_k % _314;
                            bool _316 = _315 == 0;
                            int _317 = _X_s0_kk_ii_jj >> 10;
                            bool _318 = _317 == 0;
                            bool _319 = _316 && _318;
                            if (_319) {
                                float _320 = 0.0f;
                                _313 = _320;
                            } // if _319
                            else {
                                float _322 = _Z_shreg[0][_X_s0_jjj][_X_s0_iii];
                                float _323 = sycl::ext::intel::fpga_reg(_322);
                                _313 = _323;
                            } // if _319 else
                            float _324 = _313;
                            _Z_shreg_temp = _324;
#pragma unroll
                            for (int _X_s0_kkk = 0; _X_s0_kkk < 0 + 16;
                                 _X_s0_kkk++) {
                                float _325 = _Z_shreg_temp;
                                float _327 = _X_shreg[_X_s0_iii][_X_s0_kkk];
                                float _329 = _Y_shreg[_X_s0_jjj][_X_s0_kkk];
                                float _330 = _327 * _329;
                                float _331 = _325 + _330;
                                _Z_shreg_temp = _331;
                                int _332 = _X_s0_kkk & 3;
                                bool _333 = _332 == 3;
                                if (_333) {
                                    float _334 = _Z_shreg_temp;
                                    float _335 =
                                        sycl::ext::intel::fpga_reg(_334);
                                    _Z_shreg_temp = _335;
                                } // if _333
                            }     // for _X_s0_kkk
                            float _336 = _Z_shreg_temp;
                            _Z_shreg[0][_X_s0_jjj][_X_s0_iii] = _336;
                            (void)_336;
#pragma unroll
                            for (int _X_s0_kkk = 0; _X_s0_kkk < 0 + 16;
                                 _X_s0_kkk++) {
                                bool _337 = _X_s0_kkk == 15;
                                int _338 = _X_s0_kk_ii_jj >> 10;
                                bool _339 = _338 == 31;
                                bool _340 = _337 && _339;
                                int _341 = TOTAL_K >> 9;
                                int _342 = _X_s0_i_j_k % _341;
                                int _343 = _341 + -1;
                                bool _344 = _342 == _343;
                                bool _345 = _340 && _344;
                                if (_345) {
                                    int _346 = _X_s0_iii * 1024;
                                    float _348 =
                                        _Z_shreg[0][_X_s0_jjj][_X_s0_iii];
                                    _Z_pipe_shreg[_X_s0_jjj][_346] = _348;
                                    (void)_348;
                                } // if _345
                            }     // for _X_s0_kkk
                        }         // for _X_s0_jjj
                    }             // for _X_s0_iii
                    int _349 = TOTAL_K >> 9;
                    int _350 = _X_s0_i_j_k % _349;
                    int _351 = _349 + -1;
                    bool _352 = _350 == _351;
                    int _353 = _X_s0_kk_ii_jj >> 10;
                    bool _354 = _353 == 31;
                    bool _355 = _352 && _354;
                    int _356 = _X_s0_kk_ii_jj & 31;
                    bool _357 = _356 == 0;
                    int _358 = _X_s0_kk_ii_jj & 1023;
                    int _359 = _358 >> 5;
                    bool _360 = _359 == 0;
                    bool _361 = _357 && _360;
                    bool _362 = _355 && _361;
                    if (_362) {
                        int _363 = _Z_pipe_iter_temp;
                        _Z_pipe_base_temp = _363;
                    } // if _362
                    float8 _Out_channel_temp;
#pragma unroll
                    for (int _Z_pipe_b__14 = 0; _Z_pipe_b__14 < 0 + 8;
                         _Z_pipe_b__14++) {
                        float _365 = _Z_pipe_shreg[_Z_pipe_b__14][0];
                        _Out_channel_temp[_Z_pipe_b__14] = _365;
#pragma unroll
                        for (int _Z_pipe_b__14_dummy = 0;
                             _Z_pipe_b__14_dummy < 0 + 8;
                             _Z_pipe_b__14_dummy++) {
                            float _366 = _Out_channel_temp[_Z_pipe_b__14_dummy];
                            float _367 = sycl::ext::intel::fpga_reg(
                                sycl::ext::intel::fpga_reg(_366));
                            _Out_channel_temp[_Z_pipe_b__14_dummy] = _367;
                        } // for _Z_pipe_b__14_dummy
                    }     // for _Z_pipe_b__14
                    int _368 = _Z_pipe_iter_temp;
                    int _369 = _Z_pipe_base_temp;
                    int _370 = _369 + 10240;
                    bool _371 = _368 < _370;
                    if (_371) {
                        float8 _372 = _Out_channel_temp;
                        _Out_channel::write(_372);
                        (void)_372;
                    } // if _371
#pragma unroll
                    for (int _Z_pipe_b__15 = 0; _Z_pipe_b__15 < 0 + 8;
                         _Z_pipe_b__15++) {
#pragma unroll
                        for (int _Z_pipe_p__7 = 0; _Z_pipe_p__7 < 0 + 9;
                             _Z_pipe_p__7++) {
#pragma unroll
                            for (int _Z_pipe_l__7 = 0; _Z_pipe_l__7 < 0 + 1023;
                                 _Z_pipe_l__7++) {
                                int _373 = _Z_pipe_p__7 * 1024;
                                int _374 = _373 + _Z_pipe_l__7;
                                int _375 = _374 + 1;
                                float _377 = _Z_pipe_shreg[_Z_pipe_b__15][_375];
                                _Z_pipe_shreg[_Z_pipe_b__15][_374] = _377;
                                (void)_377;
                            } // for _Z_pipe_l__7
                            int _378 = _Z_pipe_p__7 * 1024;
                            int _379 = _378 + 1023;
                            int _380 = _378 + 1024;
                            float _382 = _Z_pipe_shreg[_Z_pipe_b__15][_380];
                            float _383 = sycl::ext::intel::fpga_reg(
                                sycl::ext::intel::fpga_reg(_382));
                            _Z_pipe_shreg[_Z_pipe_b__15][_379] = _383;
                            (void)_383;
                        } // for _Z_pipe_p__7
                    }     // for _Z_pipe_b__15
                    int _384 = _Z_pipe_iter_temp;
                    int _385 = _384 + 1;
                    _Z_pipe_iter_temp = _385;
                } // for _X_s0_kk_ii_jj
            }     // for _X_s0_i_j_k
        });
    }));

    // kernel_unloader
    std::cout << "kernel_unloader" << std::endl;
    oneapi_kernel_events.push_back(q.submit([&](sycl::handler &h) {
        h.single_task<class kernel_unloader>(
            [=]() [[intel::kernel_args_restrict]] {
                device_ptr<float> serialized_C_d(serialized_C_device);
                int _addr_temp;
                _addr_temp = 0;
                int _386 = TOTAL_I / 320;
                for (int _unloader_s0_i = 0; _unloader_s0_i < 0 + _386;
                     _unloader_s0_i++) {
                    int _387 = TOTAL_J >> 8;
                    for (int _unloader_s0_j = 0; _unloader_s0_j < 0 + _387;
                         _unloader_s0_j++) {
                        for (int _unloader_s0_iii_ii_jj = 0;
                             _unloader_s0_iii_ii_jj < 0 + 10240;
                             _unloader_s0_iii_ii_jj++) {
                            float8 __388 = _Out_channel::read();
                            int _389 = _addr_temp;
                            int _390 = _389 * 8;
#pragma unroll
                            for (int vec_idx = 0; vec_idx < 8; vec_idx++) {
                                serialized_C_d[_390 + vec_idx] = __388[vec_idx];
                            }
                            int _391 = _addr_temp;
                            int _392 = _391 + 1;
                            _addr_temp = _392;
                        } // for _unloader_s0_iii_ii_jj
                    }     // for _unloader_s0_j
                }         // for _unloader_s0_i
            });
    }));

    for (unsigned int i = 0; i < oneapi_kernel_events.size(); i++) {
        oneapi_kernel_events.at(i).wait();
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

    std::cout << "memcpy matrix C device->host" << std::endl;
    q.memcpy(serialized_C, serialized_C_device, num_elem_C * sizeof(float))
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
                            C[total_j + total_i * TOTAL_J] =
                                serialized_C[addr++];
                        }

    free(serialized_A);
    free(serialized_B);
    free(serialized_C);
    sycl::free(serialized_A_device, q);
    sycl::free(serialized_B_device, q);
    sycl::free(serialized_C_device, q);
}
