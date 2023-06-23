#include "Halide.h"

// Constant parameters and data types of the kernel (dimensions of the systolic array)
#include "parameters.h"
using namespace Halide;

int main()
{
    // Loop indices
    #define P               kkk,      jjj,  iii,  jj, ii, kk,     k,  j,i
    #define P_kkk_minus_1   kkk-1,    jjj,  iii,  jj, ii, kk,     k,  j,i
    #define P_kk_minus_1    kkk+KKK-1,jjj,  iii,  jj, ii, kk-1,   k,  j,i
    #define P_k_minus_1     kkk+KKK-1,jjj,  iii,  jj, ii, kk+KK-1,k-1,j,i
    #define P_jjj_minus_1   kkk,      jjj-1,iii,  jj, ii, kk,     k,  j,i
    #define P_iii_minus_1   kkk,      jjj,  iii-1,jj, ii, kk,     k,  j,i
    #define P_reduced                 jjj,  iii,  jj, ii,             j,i // This specifies the order of the resulting data after reduction is done
    #define P_reorder                 jjj,        jj, ii, iii,        j,i // This specifies the order of the resulting data when writing to the device DRAM

    // Loop indices before tiling. They might be bigger than the matrices' actual dimensions, due to the fix dimensions of the systolic array.
    #define total_i         (iii + III * ii + III * II * i)
    #define total_j         (jjj + JJJ * jj + JJJ * JJ * j)
    #define total_k         (kkk + KKK * kk + KKK * KK * k)

    // Matrices' dimensions.
    #define MATRICES_I      (A.dim(1).extent())
    #define MATRICES_K      (B.dim(1).extent())
    #define MATRICES_J      (B.dim(0).extent())

    // Are the loop indices outside of the valid range of the matrices' dimensions? In this implementation, we load the input matrix A and B in KKK-wide vectors,
    // load the input matrix C and store the output matrix in JJJ-wide vectors. Thus we require the matrices' dimensions must be multiples of KKK and JJJ, respectively.
    #define addr_A_out_of_range total_i >= MATRICES_I || KKK * (kk + KK * k) >= MATRICES_K
    #define addr_B_out_of_range JJJ * (jj + JJ * j) >= MATRICES_J || KKK * (kk + KK * k) >= MATRICES_K

    // Outer loop bounds, which are determined by the matrices' dimensions
    #define I ((MATRICES_I + (III * II - 1)) / (III * II))
    #define J ((MATRICES_J + (JJJ * JJ - 1)) / (JJJ * JJ))
    #define K ((MATRICES_K + (KKK * KK - 1)) / (KKK * KK))

    // Inputs. Depending if an input matrix is symmetric or is general but transposed, we may read, e.g., the upper triangle from the upper triangle directly or from the corresponding lower triangle.
    Param<bool> Upper_From_Upper_A("Upper_From_Upper_A"), Upper_From_Upper_B("Upper_From_Upper_B"), Upper_From_Upper_C("Upper_From_Upper_C");
    Param<bool> Lower_From_Lower_A("Lower_From_Lower_A"), Lower_From_Lower_B("Lower_From_Lower_B"), Lower_From_Lower_C("Lower_From_Lower_C");
    Param<bool> ConjugateA("ConjugateA"), ConjugateB("ConjugateB"), ConjugateC("ConjugateC"); // Conjugate the read values of A/B/C?
    Param<bool> HalfSpaceOut("HalfSpaceOut"); // Compute only half output? This is true when the output is symmetric. In this case, we compute only the upper triangle of the output.
    Param<CONST_TYPE> alpha("alpha"), beta("beta");
    ImageParam A("A", TTYPE, 2), B("B", TTYPE, 2), C("C", TTYPE, 2);

    // UREs
    Var kkk("kkk"), jjj("jjj"), iii("iii"), jj("jj"), ii("ii"), kk("kk"), k("k"), j("j"), i("i");
    URE X("X", TTYPE, {P}), Y("Y", TTYPE, {P}), Z("Z", TTYPE, {P}), Product("Product");
    URE Add("Add", TTYPE, {P_reorder}), Out("Out", TTYPE, {P_reorder});

    Expr Is_Upper_A = total_i <= total_k;
    Expr Is_Lower_A = !Is_Upper_A;
    Expr Effective_dim0_A = select(((Is_Upper_A && Upper_From_Upper_A) || (Is_Lower_A && Lower_From_Lower_A)), total_k, total_i);
    Expr Effective_dim1_A = select(((Is_Upper_A && Upper_From_Upper_A) || (Is_Lower_A && Lower_From_Lower_A)), total_i, total_k);
    Expr Check_Load_A = select(addr_A_out_of_range, ZERO, conditional_conjugate(ConjugateA, A(Effective_dim0_A, Effective_dim1_A)));

    Expr Is_Upper_B = total_k <= total_j;
    Expr Is_Lower_B = !Is_Upper_B;
    Expr Effective_dim0_B = select(((Is_Upper_B && Upper_From_Upper_B) || (Is_Lower_B && Lower_From_Lower_B)), total_j, total_k);
    Expr Effective_dim1_B = select(((Is_Upper_B && Upper_From_Upper_B) || (Is_Lower_B && Lower_From_Lower_B)), total_k, total_j);
    Expr Check_Load_B = select(addr_B_out_of_range, ZERO, conditional_conjugate(ConjugateB, B(Effective_dim0_B, Effective_dim1_B)));

    X(P) = select(jjj == 0, Check_Load_A, X(P_jjj_minus_1));
    Y(P) = select(iii == 0, Check_Load_B, Y(P_iii_minus_1));
    Z(P) = select(k == 0 && kk == 0 && kkk == 0, ZERO,
                select(kkk == 0, select(kk == 0, Z(P_k_minus_1), Z(P_kk_minus_1)), Z(P_kkk_minus_1)))
                + X(P) * Y(P);
    Product(P_reduced) = select(k == K-1 && kk == KK-1 && kkk == KKK-1, Z(P));

    // For C, we do not need check its range: the loading of C happens only when adding C with the product, and the product is ensured to be in the valid range.
    Expr Is_Upper_C = total_i <= total_j;
    Expr Is_Lower_C = !Is_Upper_C;
    Expr Effective_dim0_C = select(((Is_Upper_C && Upper_From_Upper_C) || (Is_Lower_C && Lower_From_Lower_C)), total_j, total_i);
    Expr Effective_dim1_C = select(((Is_Upper_C && Upper_From_Upper_C) || (Is_Lower_C && Lower_From_Lower_C)), total_i, total_j);
    Expr Check_Load_C = conditional_conjugate(ConjugateC, C(Effective_dim0_C, Effective_dim1_C));

    Add(P_reorder) = alpha * Product(P_reorder) + select(beta == ZERO, ZERO, beta * Check_Load_C);
    Out(P_reorder) = select(true, Add(P_reorder));

    // Put the UREs that compute A*B (i.e. X, Y, Z and Product) inside the same loop nest.
    X.merge_ures(Y, Z, Product);
    Add.merge_ures(Out);

    // Explicitly set the loop bounds: every loop has a min, and an extent (number of iterations).
    // If HalfSpaceOut is true, we generate the upper half output. However, since every invocation of the systolic array produces a rectangular tile,
    // the output is "upper half" in terms of tiles. For the tiles at the diagonal, post-processing is needed to remove values that are within the tiles but
    // are below the diagonal, so that the output is "upper half" in terms of elements.
    X.set_bounds(jjj, 0, JJJ, iii, 0, III, kkk, 0, KKK)
     .set_bounds(jj,  0, JJ,  ii,  0, II,  kk,  0, KK)
     .set_bounds(j,   select(HalfSpaceOut, i, 0), select(HalfSpaceOut, J-i, J))
     .set_bounds(i,   0, I,   k,   0, K);

    Add.set_bounds(jjj, 0, JJJ, iii, 0, III)
       .set_bounds(jj,  0, JJ,  ii,  0, II)
       .set_bounds(j,   select(HalfSpaceOut, i, 0), select(HalfSpaceOut, J-i, J))
       .set_bounds(i,   0, I);

    // Create a systolic array
    X.space_time_transform(jjj, iii).run_forever();
    Add.vectorize(jjj);

    // I/O network
    Stensor DA("ALoader", DRAM), SA("AFeeder", SRAM), DB("BLoader", DRAM), SB("BFeeder", SRAM), DC("CLoader", DRAM);
    Stensor RCollector("RCollector", REG), SCollector("SCollector", SRAM), DOut("Unloader", DRAM), Output("Output");
    A   >> DA.out(kkk).apply_transform(Check_Load_A) >> FIFO(256) >> SA.scope(k).out(kkk, iii) >> FIFO(256);
    B   >> DB.out(kkk).apply_transform(Check_Load_B) >> FIFO(256) >> SB.scope(k).out(kkk, jjj) >> FIFO(256);
    C   >> DC.out(jjj).apply_transform(Check_Load_C) >> FIFO(256);
    Product >> RCollector.scope(iii).out(jjj) >> FIFO(256) >> SCollector >> FIFO(256);
    Out >> FIFO(256) >> DOut >> Output(total_j, total_i);

    // For performance, we require that MATRICES_K and MATRIX_J must be multiples of KKK and JJJ, respectively (i.e. the vector lengths of the input and output data)
    Output.require(MATRICES_K % KKK == 0)
          .require(MATRICES_J % JJJ == 0);

    // Compile the kernel to an oneAPI impl, and expose a C interface for the host to invoke
    Output.compile_to_oneapi(OUTPUT_FILE, {Upper_From_Upper_A, Upper_From_Upper_B, Upper_From_Upper_C, Lower_From_Lower_A, Lower_From_Lower_B, Lower_From_Lower_C,
                                           ConjugateA, ConjugateB, ConjugateC, HalfSpaceOut, alpha, beta, A, B, C}, KERNEL, IntelFPGA);

    return 0;
}
