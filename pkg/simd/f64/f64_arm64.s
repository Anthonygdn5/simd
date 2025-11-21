//go:build arm64

#include "textflag.h"

// ARM64 NEON implementation for float64 operations
// NEON processes 2 x float64 per vector register (128-bit)

// func dotProductNEON(a, b []float64) float64
TEXT ·dotProductNEON(SB), NOSPLIT, $0-56
    MOVD a_base+0(FP), R0      // R0 = &a[0]
    MOVD a_len+8(FP), R2       // R2 = len(a)
    MOVD b_base+24(FP), R1     // R1 = &b[0]

    // V0, V1 = dual accumulators for ILP
    VEOR V0.B16, V0.B16, V0.B16
    VEOR V1.B16, V1.B16, V1.B16

    // Process 4 elements (2 NEON ops) per iteration
    LSR $2, R2, R3             // R3 = len / 4
    CBZ R3, dot_remainder2

dot_loop4:
    VLD1.P 16(R0), [V2.D2]     // Load a[i:i+2]
    VLD1.P 16(R0), [V3.D2]     // Load a[i+2:i+4]
    VLD1.P 16(R1), [V4.D2]     // Load b[i:i+2]
    VLD1.P 16(R1), [V5.D2]     // Load b[i+2:i+4]
    VFMLA V2.D2, V4.D2, V0.D2  // V0 += V2 * V4
    VFMLA V3.D2, V5.D2, V1.D2  // V1 += V3 * V5
    SUB $1, R3
    CBNZ R3, dot_loop4

    // Combine accumulators
    VFADD V0.D2, V1.D2, V0.D2

dot_remainder2:
    // Check for 2-3 remaining elements
    AND $3, R2, R3
    LSR $1, R3, R4             // R4 = remainder / 2
    CBZ R4, dot_remainder1

    VLD1.P 16(R0), [V2.D2]
    VLD1.P 16(R1), [V4.D2]
    VFMLA V2.D2, V4.D2, V0.D2

dot_remainder1:
    // Check for final single element
    AND $1, R3, R4
    CBZ R4, dot_reduce

    FMOVD (R0), F2
    FMOVD (R1), F4
    FMADDD F2, F4, F0, F0

dot_reduce:
    // Horizontal sum: V0[0] + V0[1]
    FADDP V0.D2, V0.D2, V0.D2

    FMOVD F0, ret+48(FP)
    RET

// func addNEON(dst, a, b []float64)
TEXT ·addNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1
    MOVD b_base+48(FP), R2

    LSR $1, R3, R4
    CBZ R4, add_scalar

add_loop2:
    VLD1.P 16(R1), [V0.D2]
    VLD1.P 16(R2), [V1.D2]
    VFADD V0.D2, V1.D2, V2.D2
    VST1.P [V2.D2], 16(R0)
    SUB $1, R4
    CBNZ R4, add_loop2

add_scalar:
    AND $1, R3
    CBZ R3, add_done
    FMOVD (R1), F0
    FMOVD (R2), F1
    FADDD F0, F1, F0
    FMOVD F0, (R0)

add_done:
    RET

// func subNEON(dst, a, b []float64)
TEXT ·subNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1
    MOVD b_base+48(FP), R2

    LSR $1, R3, R4
    CBZ R4, sub_scalar

sub_loop2:
    VLD1.P 16(R1), [V0.D2]
    VLD1.P 16(R2), [V1.D2]
    VFSUB V1.D2, V0.D2, V2.D2
    VST1.P [V2.D2], 16(R0)
    SUB $1, R4
    CBNZ R4, sub_loop2

sub_scalar:
    AND $1, R3
    CBZ R3, sub_done
    FMOVD (R1), F0
    FMOVD (R2), F1
    FSUBD F1, F0, F0
    FMOVD F0, (R0)

sub_done:
    RET

// func mulNEON(dst, a, b []float64)
TEXT ·mulNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1
    MOVD b_base+48(FP), R2

    LSR $1, R3, R4
    CBZ R4, mul_scalar

mul_loop2:
    VLD1.P 16(R1), [V0.D2]
    VLD1.P 16(R2), [V1.D2]
    VFMUL V0.D2, V1.D2, V2.D2
    VST1.P [V2.D2], 16(R0)
    SUB $1, R4
    CBNZ R4, mul_loop2

mul_scalar:
    AND $1, R3
    CBZ R3, mul_done
    FMOVD (R1), F0
    FMOVD (R2), F1
    FMULD F0, F1, F0
    FMOVD F0, (R0)

mul_done:
    RET

// func divNEON(dst, a, b []float64)
TEXT ·divNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1
    MOVD b_base+48(FP), R2

    LSR $1, R3, R4
    CBZ R4, div_scalar

div_loop2:
    VLD1.P 16(R1), [V0.D2]
    VLD1.P 16(R2), [V1.D2]
    VFDIV V1.D2, V0.D2, V2.D2
    VST1.P [V2.D2], 16(R0)
    SUB $1, R4
    CBNZ R4, div_loop2

div_scalar:
    AND $1, R3
    CBZ R3, div_done
    FMOVD (R1), F0
    FMOVD (R2), F1
    FDIVD F1, F0, F0
    FMOVD F0, (R0)

div_done:
    RET

// func scaleNEON(dst, a []float64, s float64)
TEXT ·scaleNEON(SB), NOSPLIT, $0-56
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R2
    MOVD a_base+24(FP), R1
    FMOVD s+48(FP), F3
    VDUP F3, V3.D2             // Broadcast scalar

    LSR $1, R2, R3
    CBZ R3, scale_scalar

scale_loop2:
    VLD1.P 16(R1), [V0.D2]
    VFMUL V0.D2, V3.D2, V1.D2
    VST1.P [V1.D2], 16(R0)
    SUB $1, R3
    CBNZ R3, scale_loop2

scale_scalar:
    AND $1, R2
    CBZ R2, scale_done
    FMOVD (R1), F0
    FMULD F0, F3, F0
    FMOVD F0, (R0)

scale_done:
    RET

// func addScalarNEON(dst, a []float64, s float64)
TEXT ·addScalarNEON(SB), NOSPLIT, $0-56
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R2
    MOVD a_base+24(FP), R1
    FMOVD s+48(FP), F3
    VDUP F3, V3.D2

    LSR $1, R2, R3
    CBZ R3, addsc_scalar

addsc_loop2:
    VLD1.P 16(R1), [V0.D2]
    VFADD V0.D2, V3.D2, V1.D2
    VST1.P [V1.D2], 16(R0)
    SUB $1, R3
    CBNZ R3, addsc_loop2

addsc_scalar:
    AND $1, R2
    CBZ R2, addsc_done
    FMOVD (R1), F0
    FADDD F0, F3, F0
    FMOVD F0, (R0)

addsc_done:
    RET

// func sumNEON(a []float64) float64
TEXT ·sumNEON(SB), NOSPLIT, $0-32
    MOVD a_base+0(FP), R0
    MOVD a_len+8(FP), R1

    VEOR V0.B16, V0.B16, V0.B16

    LSR $1, R1, R2
    CBZ R2, sum_scalar

sum_loop2:
    VLD1.P 16(R0), [V1.D2]
    VFADD V0.D2, V1.D2, V0.D2
    SUB $1, R2
    CBNZ R2, sum_loop2

sum_scalar:
    AND $1, R1
    CBZ R1, sum_reduce
    FMOVD (R0), F1
    FADDD F0, F1, F0

sum_reduce:
    FADDP V0.D2, V0.D2, V0.D2
    FMOVD F0, ret+24(FP)
    RET

// func minNEON(a []float64) float64
TEXT ·minNEON(SB), NOSPLIT, $0-32
    MOVD a_base+0(FP), R0
    MOVD a_len+8(FP), R1

    VLD1.P 16(R0), [V0.D2]     // Initialize with first 2
    SUB $2, R1

    LSR $1, R1, R2
    CBZ R2, min_scalar

min_loop2:
    VLD1.P 16(R0), [V1.D2]
    VFMIN V0.D2, V1.D2, V0.D2
    SUB $1, R2
    CBNZ R2, min_loop2

min_scalar:
    AND $1, R1
    CBZ R1, min_reduce
    FMOVD (R0), F1
    FMIND F0, F1, F0

min_reduce:
    // Horizontal min
    VDUP V0.D[1], V1.D2
    FMIND F0, F1, F0
    FMOVD F0, ret+24(FP)
    RET

// func maxNEON(a []float64) float64
TEXT ·maxNEON(SB), NOSPLIT, $0-32
    MOVD a_base+0(FP), R0
    MOVD a_len+8(FP), R1

    VLD1.P 16(R0), [V0.D2]
    SUB $2, R1

    LSR $1, R1, R2
    CBZ R2, max_scalar

max_loop2:
    VLD1.P 16(R0), [V1.D2]
    VFMAX V0.D2, V1.D2, V0.D2
    SUB $1, R2
    CBNZ R2, max_loop2

max_scalar:
    AND $1, R1
    CBZ R1, max_reduce
    FMOVD (R0), F1
    FMAXD F0, F1, F0

max_reduce:
    VDUP V0.D[1], V1.D2
    FMAXD F0, F1, F0
    FMOVD F0, ret+24(FP)
    RET

// func absNEON(dst, a []float64)
TEXT ·absNEON(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R2
    MOVD a_base+24(FP), R1

    LSR $1, R2, R3
    CBZ R3, abs_scalar

abs_loop2:
    VLD1.P 16(R1), [V0.D2]
    VFABS V0.D2, V1.D2
    VST1.P [V1.D2], 16(R0)
    SUB $1, R3
    CBNZ R3, abs_loop2

abs_scalar:
    AND $1, R2
    CBZ R2, abs_done
    FMOVD (R1), F0
    FABSD F0, F0
    FMOVD F0, (R0)

abs_done:
    RET

// func negNEON(dst, a []float64)
TEXT ·negNEON(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R2
    MOVD a_base+24(FP), R1

    LSR $1, R2, R3
    CBZ R3, neg_scalar

neg_loop2:
    VLD1.P 16(R1), [V0.D2]
    VFNEG V0.D2, V1.D2
    VST1.P [V1.D2], 16(R0)
    SUB $1, R3
    CBNZ R3, neg_loop2

neg_scalar:
    AND $1, R2
    CBZ R2, neg_done
    FMOVD (R1), F0
    FNEGD F0, F0
    FMOVD F0, (R0)

neg_done:
    RET

// func fmaNEON(dst, a, b, c []float64)
TEXT ·fmaNEON(SB), NOSPLIT, $0-96
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R4
    MOVD a_base+24(FP), R1
    MOVD b_base+48(FP), R2
    MOVD c_base+72(FP), R3

    LSR $1, R4, R5
    CBZ R5, fma_scalar

fma_loop2:
    VLD1.P 16(R1), [V0.D2]     // a
    VLD1.P 16(R2), [V1.D2]     // b
    VLD1.P 16(R3), [V2.D2]     // c
    VFMLA V0.D2, V1.D2, V2.D2  // V2 = a*b + c
    VST1.P [V2.D2], 16(R0)
    SUB $1, R5
    CBNZ R5, fma_loop2

fma_scalar:
    AND $1, R4
    CBZ R4, fma_done
    FMOVD (R1), F0
    FMOVD (R2), F1
    FMOVD (R3), F2
    FMADDD F0, F1, F2, F2
    FMOVD F2, (R0)

fma_done:
    RET

// func clampNEON(dst, a []float64, minVal, maxVal float64)
TEXT ·clampNEON(SB), NOSPLIT, $0-64
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R2
    MOVD a_base+24(FP), R1
    FMOVD minVal+48(FP), F2
    FMOVD maxVal+56(FP), F3
    VDUP F2, V2.D2
    VDUP F3, V3.D2

    LSR $1, R2, R3
    CBZ R3, clamp_scalar

clamp_loop2:
    VLD1.P 16(R1), [V0.D2]
    VFMAX V0.D2, V2.D2, V0.D2  // max(val, min)
    VFMIN V0.D2, V3.D2, V0.D2  // min(result, max)
    VST1.P [V0.D2], 16(R0)
    SUB $1, R3
    CBNZ R3, clamp_loop2

clamp_scalar:
    AND $1, R2
    CBZ R2, clamp_done
    FMOVD (R1), F0
    FMAXD F0, F2, F0
    FMIND F0, F3, F0
    FMOVD F0, (R0)

clamp_done:
    RET

// Stub implementations for new operations (TODO: optimize with NEON)
TEXT ·sqrtNEON(SB), $0-48
    B ·sqrt64Go(SB)

TEXT ·reciprocalNEON(SB), $0-48
    B ·reciprocal64Go(SB)

TEXT ·varianceNEON(SB), $0-32
    B ·variance64Go(SB)

TEXT ·euclideanDistanceNEON(SB), $0-56
    B ·euclideanDistance64Go(SB)
