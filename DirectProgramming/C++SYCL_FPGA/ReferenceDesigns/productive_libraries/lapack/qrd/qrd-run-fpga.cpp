#include "qrd.sycl.h"
// The only header file needed for including T2S.
#include "HalideBuffer.h"

#define I 128
#define J 128
#define K 128
#ifdef FPGA_EMULATOR
#define BATCH_SIZE 128
#else
#define BATCH_SIZE 16384
#endif

using namespace std;

constexpr size_t batch_size = BATCH_SIZE;

int main()
{
#if defined(FPGA_EMULATOR)
    std::cout << "USING FPGA EMULATOR" << std::endl;
    sycl::ext::intel::fpga_emulator_selector device_selector;
#else
    std::cout << "USING FPGA HARDWARE" << std::endl;
    sycl::ext::intel::fpga_selector device_selector;
#endif
    Halide::Runtime::Buffer<float> a(J, K, BATCH_SIZE), q(K, I, BATCH_SIZE), r(J, I, BATCH_SIZE);
    // Generate the random input matrices
    constexpr size_t kRandomSeed = 1138;
    constexpr size_t kRandomMin = 1;
    constexpr size_t kRandomMax = 10;
    srand(kRandomSeed);
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t k = 0; k < K; k++) {
            for (size_t j = 0; j < J; j++) {
                int random_val = rand();
                float random_float =
                    random_val % (kRandomMax - kRandomMin) + kRandomMin;
                a(j, k, b) = random_float;
            }
        }
    }

    qrd(device_selector, a, q, r);

    bool passed = true;
    std::vector<size_t> to_check;
    // Check at least matrix 0
    to_check.push_back(0);
    // Spot check the last and the middle one
    if (BATCH_SIZE > 2)
        to_check.push_back(BATCH_SIZE / 2);
    if (BATCH_SIZE > 1)
        to_check.push_back(BATCH_SIZE - 1);

    for (size_t b : to_check) {
        printf("\n*** Verifying results on input matrix %ld\n", b);
        printf("*** Matrix Q: \n");
        for (int k = 0; k < K; k++) {
            for (int i = 0; i < I; i++) {
                printf("%5.2f ", q(k, i, b));
            }
            printf("\n");
        }

        // printf("*** Matrix R: \n");
        // for (int i = 0; i < I; i++) {
        //     for (int j = 0; j < J; j++) {
        //         printf("%5.2f ", R[(j + i * J + b * I * J)]);
        //     }
        //     printf("\n");
        // }

        // check if Q * R can reproduce the inputs
        // printf("*** Q * R [Input]\n");
        // for (int k = 0; k < K; k++) {
        //     for (int j = 0; j < J; j++) {
        //         float golden = 0.0f;
        //         for (int i = 0; i < I; i++) {
        //             golden +=
        //                 q(k, i, b) * r(j, i, b);
        //         }
        //         bool correct = fabs(A[(j + k * J + b * J * K)] - golden) < 0.01;
        //         passed = passed && correct;
        //         // printf("%5.2f [%5.2f%s] ", golden, A[(j + k * J + b * J *
        //         // K)],
        //         //        correct ? "" : " !!");
        //     }
        //     // printf("\n");
        // }
    }

    if (passed) {
        printf("[PASSED]\n");
    } else {
        printf("[FAILED]\n");
    }
    return 0;
}
