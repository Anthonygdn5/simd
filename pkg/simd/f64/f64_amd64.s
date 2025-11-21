//go:build amd64

#include "textflag.h"

// Constants for abs (sign bit mask)
DATA absf64mask<>+0x00(SB)/8, $0x7fffffffffffffff
DATA absf64mask<>+0x08(SB)/8, $0x7fffffffffffffff
DATA absf64mask<>+0x10(SB)/8, $0x7fffffffffffffff
DATA absf64mask<>+0x18(SB)/8, $0x7fffffffffffffff
GLOBL absf64mask<>(SB), RODATA|NOPTR, $32

// func dotProductAVX(a, b []float64) float64
// Uses AVX+FMA: processes 4 float64s per iteration
TEXT ·dotProductAVX(SB), NOSPLIT, $0-56
    MOVQ a_base+0(FP), SI      // SI = &a[0]
    MOVQ a_len+8(FP), CX       // CX = len(a)
    MOVQ b_base+24(FP), DI     // DI = &b[0]

    VXORPD Y0, Y0, Y0          // Y0 = accumulator (4 x float64)

    // Main loop: process 4 elements at a time
    MOVQ CX, AX
    SHRQ $2, AX                // AX = len / 4
    JZ   dot_remainder

dot_loop4:
    VMOVUPD (SI), Y1           // Load 4 floats from a
    VMOVUPD (DI), Y2           // Load 4 floats from b
    VFMADD231PD Y1, Y2, Y0     // Y0 += Y1 * Y2 (FMA)
    ADDQ $32, SI
    ADDQ $32, DI
    DECQ AX
    JNZ  dot_loop4

dot_remainder:
    // Reduce Y0 to X0 BEFORE scalar ops (VEX scalar ops zero upper YMM bits)
    VEXTRACTF128 $1, Y0, X1    // X1 = [c, d]
    VADDPD X1, X0, X0          // X0 = [a+c, b+d]
    VHADDPD X0, X0, X0         // X0[0] = a+b+c+d

    // Handle remaining 0-3 elements (scalar)
    ANDQ $3, CX
    JZ   dot_done

dot_scalar:
    VMOVSD (SI), X1
    VMOVSD (DI), X2
    VFMADD231SD X1, X2, X0     // X0[0] += X1 * X2
    ADDQ $8, SI
    ADDQ $8, DI
    DECQ CX
    JNZ  dot_scalar

dot_done:
    VMOVSD X0, ret+48(FP)
    VZEROUPPER
    RET

// func addAVX(dst, a, b []float64)
TEXT ·addAVX(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX    // DX = &dst[0]
    MOVQ dst_len+8(FP), CX     // CX = len(dst)
    MOVQ a_base+24(FP), SI     // SI = &a[0]
    MOVQ b_base+48(FP), DI     // DI = &b[0]

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   add_remainder

add_loop4:
    VMOVUPD (SI), Y0
    VMOVUPD (DI), Y1
    VADDPD Y0, Y1, Y2
    VMOVUPD Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, DX
    DECQ AX
    JNZ  add_loop4

add_remainder:
    ANDQ $3, CX
    JZ   add_done

add_scalar:
    VMOVSD (SI), X0
    VADDSD (DI), X0, X0
    VMOVSD X0, (DX)
    ADDQ $8, SI
    ADDQ $8, DI
    ADDQ $8, DX
    DECQ CX
    JNZ  add_scalar

add_done:
    VZEROUPPER
    RET

// func subAVX(dst, a, b []float64)
TEXT ·subAVX(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   sub_remainder

sub_loop4:
    VMOVUPD (SI), Y0
    VMOVUPD (DI), Y1
    VSUBPD Y1, Y0, Y2
    VMOVUPD Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, DX
    DECQ AX
    JNZ  sub_loop4

sub_remainder:
    ANDQ $3, CX
    JZ   sub_done

sub_scalar:
    VMOVSD (SI), X0
    VSUBSD (DI), X0, X0
    VMOVSD X0, (DX)
    ADDQ $8, SI
    ADDQ $8, DI
    ADDQ $8, DX
    DECQ CX
    JNZ  sub_scalar

sub_done:
    VZEROUPPER
    RET

// func mulAVX(dst, a, b []float64)
TEXT ·mulAVX(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   mul_remainder

mul_loop4:
    VMOVUPD (SI), Y0
    VMOVUPD (DI), Y1
    VMULPD Y0, Y1, Y2
    VMOVUPD Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, DX
    DECQ AX
    JNZ  mul_loop4

mul_remainder:
    ANDQ $3, CX
    JZ   mul_done

mul_scalar:
    VMOVSD (SI), X0
    VMULSD (DI), X0, X0
    VMOVSD X0, (DX)
    ADDQ $8, SI
    ADDQ $8, DI
    ADDQ $8, DX
    DECQ CX
    JNZ  mul_scalar

mul_done:
    VZEROUPPER
    RET

// func divAVX(dst, a, b []float64)
TEXT ·divAVX(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   div_remainder

div_loop4:
    VMOVUPD (SI), Y0
    VMOVUPD (DI), Y1
    VDIVPD Y1, Y0, Y2
    VMOVUPD Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, DX
    DECQ AX
    JNZ  div_loop4

div_remainder:
    ANDQ $3, CX
    JZ   div_done

div_scalar:
    VMOVSD (SI), X0
    VDIVSD (DI), X0, X0
    VMOVSD X0, (DX)
    ADDQ $8, SI
    ADDQ $8, DI
    ADDQ $8, DX
    DECQ CX
    JNZ  div_scalar

div_done:
    VZEROUPPER
    RET

// func scaleAVX(dst, a []float64, s float64)
TEXT ·scaleAVX(SB), NOSPLIT, $0-56
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    VBROADCASTSD s+48(FP), Y1  // Broadcast scalar to all lanes

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   scale_remainder

scale_loop4:
    VMOVUPD (SI), Y0
    VMULPD Y0, Y1, Y2
    VMOVUPD Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  scale_loop4

scale_remainder:
    ANDQ $3, CX
    JZ   scale_done
    VMOVSD s+48(FP), X1

scale_scalar:
    VMOVSD (SI), X0
    VMULSD X0, X1, X0
    VMOVSD X0, (DX)
    ADDQ $8, SI
    ADDQ $8, DX
    DECQ CX
    JNZ  scale_scalar

scale_done:
    VZEROUPPER
    RET

// func addScalarAVX(dst, a []float64, s float64)
TEXT ·addScalarAVX(SB), NOSPLIT, $0-56
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    VBROADCASTSD s+48(FP), Y1

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   addsc_remainder

addsc_loop4:
    VMOVUPD (SI), Y0
    VADDPD Y0, Y1, Y2
    VMOVUPD Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  addsc_loop4

addsc_remainder:
    ANDQ $3, CX
    JZ   addsc_done
    VMOVSD s+48(FP), X1

addsc_scalar:
    VMOVSD (SI), X0
    VADDSD X0, X1, X0
    VMOVSD X0, (DX)
    ADDQ $8, SI
    ADDQ $8, DX
    DECQ CX
    JNZ  addsc_scalar

addsc_done:
    VZEROUPPER
    RET

// func sumAVX(a []float64) float64
TEXT ·sumAVX(SB), NOSPLIT, $0-32
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX

    VXORPD Y0, Y0, Y0

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   sum_remainder

sum_loop4:
    VMOVUPD (SI), Y1
    VADDPD Y0, Y1, Y0
    ADDQ $32, SI
    DECQ AX
    JNZ  sum_loop4

sum_remainder:
    // Reduce Y0 to X0 BEFORE scalar ops
    VEXTRACTF128 $1, Y0, X1
    VADDPD X1, X0, X0
    VHADDPD X0, X0, X0

    ANDQ $3, CX
    JZ   sum_done

sum_scalar:
    VMOVSD (SI), X1
    VADDSD X0, X1, X0
    ADDQ $8, SI
    DECQ CX
    JNZ  sum_scalar

sum_done:
    VMOVSD X0, ret+24(FP)
    VZEROUPPER
    RET

// func minAVX(a []float64) float64
TEXT ·minAVX(SB), NOSPLIT, $0-32
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX

    // Initialize with first 4 elements
    VMOVUPD (SI), Y0
    ADDQ $32, SI
    SUBQ $4, CX

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   min_reduce_first

min_loop4:
    VMOVUPD (SI), Y1
    VMINPD Y0, Y1, Y0
    ADDQ $32, SI
    DECQ AX
    JNZ  min_loop4

min_reduce_first:
    // Reduce Y0 to scalar BEFORE scalar ops (VEX scalar zeroes upper YMM)
    VEXTRACTF128 $1, Y0, X1
    VMINPD X0, X1, X0
    VPERMILPD $1, X0, X1
    VMINSD X0, X1, X0          // X0[0] = min of all 4 lanes

    ANDQ $3, CX
    JZ   min_done

min_scalar:
    VMOVSD (SI), X1
    VMINSD X0, X1, X0
    ADDQ $8, SI
    DECQ CX
    JNZ  min_scalar

min_done:
    VMOVSD X0, ret+24(FP)
    VZEROUPPER
    RET

// func maxAVX(a []float64) float64
TEXT ·maxAVX(SB), NOSPLIT, $0-32
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX

    VMOVUPD (SI), Y0
    ADDQ $32, SI
    SUBQ $4, CX

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   max_reduce_first

max_loop4:
    VMOVUPD (SI), Y1
    VMAXPD Y0, Y1, Y0
    ADDQ $32, SI
    DECQ AX
    JNZ  max_loop4

max_reduce_first:
    // Reduce Y0 to scalar BEFORE scalar ops
    VEXTRACTF128 $1, Y0, X1
    VMAXPD X0, X1, X0
    VPERMILPD $1, X0, X1
    VMAXSD X0, X1, X0

    ANDQ $3, CX
    JZ   max_done

max_scalar:
    VMOVSD (SI), X1
    VMAXSD X0, X1, X0
    ADDQ $8, SI
    DECQ CX
    JNZ  max_scalar

max_done:
    VMOVSD X0, ret+24(FP)
    VZEROUPPER
    RET

// func absAVX(dst, a []float64)
TEXT ·absAVX(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    VMOVUPD absf64mask<>(SB), Y2  // Load abs mask

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   abs_remainder

abs_loop4:
    VMOVUPD (SI), Y0
    VANDPD Y0, Y2, Y1           // Clear sign bit
    VMOVUPD Y1, (DX)
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  abs_loop4

abs_remainder:
    ANDQ $3, CX
    JZ   abs_done

abs_scalar:
    VMOVSD (SI), X0
    VANDPD X0, X2, X0
    VMOVSD X0, (DX)
    ADDQ $8, SI
    ADDQ $8, DX
    DECQ CX
    JNZ  abs_scalar

abs_done:
    VZEROUPPER
    RET

// func negAVX(dst, a []float64)
TEXT ·negAVX(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    VXORPD Y2, Y2, Y2           // Y2 = 0

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   neg_remainder

neg_loop4:
    VMOVUPD (SI), Y0
    VSUBPD Y0, Y2, Y1           // 0 - Y0 = -Y0
    VMOVUPD Y1, (DX)
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  neg_loop4

neg_remainder:
    ANDQ $3, CX
    JZ   neg_done

neg_scalar:
    VMOVSD (SI), X0
    VSUBSD X0, X2, X0
    VMOVSD X0, (DX)
    ADDQ $8, SI
    ADDQ $8, DX
    DECQ CX
    JNZ  neg_scalar

neg_done:
    VZEROUPPER
    RET

// func fmaAVX(dst, a, b, c []float64)
TEXT ·fmaAVX(SB), NOSPLIT, $0-96
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI
    MOVQ c_base+72(FP), R8

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   fma_remainder

fma_loop4:
    VMOVUPD (SI), Y0           // a
    VMOVUPD (DI), Y1           // b
    VMOVUPD (R8), Y2           // c
    VFMADD213PD Y2, Y1, Y0     // Y0 = Y0 * Y1 + Y2 = a*b+c
    VMOVUPD Y0, (DX)
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, R8
    ADDQ $32, DX
    DECQ AX
    JNZ  fma_loop4

fma_remainder:
    ANDQ $3, CX
    JZ   fma_done

fma_scalar:
    VMOVSD (SI), X0
    VMOVSD (DI), X1
    VMOVSD (R8), X2
    VFMADD213SD X2, X1, X0
    VMOVSD X0, (DX)
    ADDQ $8, SI
    ADDQ $8, DI
    ADDQ $8, R8
    ADDQ $8, DX
    DECQ CX
    JNZ  fma_scalar

fma_done:
    VZEROUPPER
    RET

// func clampAVX(dst, a []float64, minVal, maxVal float64)
TEXT ·clampAVX(SB), NOSPLIT, $0-64
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    VBROADCASTSD minVal+48(FP), Y1
    VBROADCASTSD maxVal+56(FP), Y2

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   clamp_remainder

clamp_loop4:
    VMOVUPD (SI), Y0
    VMAXPD Y0, Y1, Y0          // max(val, min)
    VMINPD Y0, Y2, Y0          // min(result, max)
    VMOVUPD Y0, (DX)
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  clamp_loop4

clamp_remainder:
    ANDQ $3, CX
    JZ   clamp_done
    VMOVSD minVal+48(FP), X1
    VMOVSD maxVal+56(FP), X2

clamp_scalar:
    VMOVSD (SI), X0
    VMAXSD X0, X1, X0
    VMINSD X0, X2, X0
    VMOVSD X0, (DX)
    ADDQ $8, SI
    ADDQ $8, DX
    DECQ CX
    JNZ  clamp_scalar

clamp_done:
    VZEROUPPER
    RET

// Stub implementations for new operations (TODO: optimize with AVX)
TEXT ·sqrtAVX(SB), $0-48
    JMP ·sqrt64Go(SB)

TEXT ·reciprocalAVX(SB), $0-48
    JMP ·reciprocal64Go(SB)

TEXT ·varianceAVX(SB), $0-32
    JMP ·variance64Go(SB)

TEXT ·euclideanDistanceAVX(SB), $0-56
    JMP ·euclideanDistance64Go(SB)
