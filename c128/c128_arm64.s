//go:build arm64

#include "textflag.h"

// ============================================================================
// ARM64 NEON IMPLEMENTATIONS (128-bit, 1x complex128 per iteration)
// ============================================================================
//
// complex128 layout: [real, imag] pairs
// NEON processes 128 bits = 2 float64 = 1 complex128 per iteration
//
// Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
//
// NEON opcode formulas for float64 (2D arrangement):
// FMUL Vd.2D, Vn.2D, Vm.2D: 0x6E60DC00 | (Vm << 16) | (Vn << 5) | Vd
// FADD Vd.2D, Vn.2D, Vm.2D: 0x4E60D400 | (Vm << 16) | (Vn << 5) | Vd
// FSUB Vd.2D, Vn.2D, Vm.2D: 0x4EE0D400 | (Vm << 16) | (Vn << 5) | Vd
//
// Note: In Go ARM64 asm:
// - F0-F31 alias the low 64 bits of V0-V31 (F0 = V0.D[0])
// - To access V0.D[1], use VDUP V0.D[1], V1.D2, then F1 has that value
// - VMOV Vn.D[x], Rm moves vector element to general-purpose register

// func mulNEON(dst, a, b []complex128)
TEXT ·mulNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0      // dst pointer
    MOVD dst_len+8(FP), R1       // length
    MOVD a_base+24(FP), R2       // a pointer
    MOVD b_base+48(FP), R3       // b pointer

    CBZ  R1, mul_neon_done

mul_neon_loop:
    // Load one complex128 from a and b
    VLD1 (R2), [V0.D2]           // V0 = [ar, ai], F0 = ar
    VLD1 (R3), [V1.D2]           // V1 = [br, bi], F1 = br

    // Extract all four components to scalar registers
    // ar is already in F0 (low 64 bits of V0)
    // br is already in F1 (low 64 bits of V1)
    VDUP V0.D[1], V2.D2          // V2 = [ai, ai], F2 = ai
    VDUP V1.D[1], V3.D2          // V3 = [bi, bi], F3 = bi

    // Complex multiplication: (ar + ai*i)(br + bi*i) = (ar*br - ai*bi) + (ar*bi + ai*br)*i
    // Compute products using scalar FP registers
    FMULD F0, F1, F4             // F4 = ar * br
    FMULD F2, F3, F5             // F5 = ai * bi
    FMULD F0, F3, F6             // F6 = ar * bi
    FMULD F2, F1, F7             // F7 = ai * br

    // result_real = ar*br - ai*bi
    FSUBD F5, F4, F4             // F4 = ar*br - ai*bi

    // result_imag = ar*bi + ai*br
    FADDD F6, F7, F5             // F5 = ar*bi + ai*br

    // Store result directly to memory
    FMOVD F4, (R0)               // Store real part
    FMOVD F5, 8(R0)              // Store imag part

    ADD  $16, R2
    ADD  $16, R3
    ADD  $16, R0
    SUB  $1, R1
    CBNZ R1, mul_neon_loop

mul_neon_done:
    RET

// func mulConjNEON(dst, a, b []complex128)
// a * conj(b) = (ar + ai*i)(br - bi*i) = (ar*br + ai*bi) + (ai*br - ar*bi)*i
TEXT ·mulConjNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R1
    MOVD a_base+24(FP), R2
    MOVD b_base+48(FP), R3

    CBZ  R1, mulconj_neon_done

mulconj_neon_loop:
    VLD1 (R2), [V0.D2]           // V0 = [ar, ai], F0 = ar
    VLD1 (R3), [V1.D2]           // V1 = [br, bi], F1 = br

    // Extract components
    VDUP V0.D[1], V2.D2          // F2 = ai
    VDUP V1.D[1], V3.D2          // F3 = bi

    // Compute products
    FMULD F0, F1, F4             // F4 = ar * br
    FMULD F2, F3, F5             // F5 = ai * bi
    FMULD F2, F1, F6             // F6 = ai * br
    FMULD F0, F3, F7             // F7 = ar * bi

    // result_real = ar*br + ai*bi
    FADDD F4, F5, F4             // F4 = ar*br + ai*bi

    // result_imag = ai*br - ar*bi
    FSUBD F7, F6, F5             // F5 = ai*br - ar*bi

    // Store result
    FMOVD F4, (R0)
    FMOVD F5, 8(R0)

    ADD  $16, R2
    ADD  $16, R3
    ADD  $16, R0
    SUB  $1, R1
    CBNZ R1, mulconj_neon_loop

mulconj_neon_done:
    RET

// func scaleNEON(dst, a []complex128, s complex128)
// a * s = (ar + ai*i)(sr + si*i) = (ar*sr - ai*si) + (ar*si + ai*sr)*i
TEXT ·scaleNEON(SB), NOSPLIT, $0-64
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R1
    MOVD a_base+24(FP), R2

    // Load scalar s once
    FMOVD s+48(FP), F20          // F20 = sr (kept across loop)
    FMOVD s+56(FP), F21          // F21 = si (kept across loop)

    CBZ  R1, scale_neon_done

scale_neon_loop:
    VLD1 (R2), [V0.D2]           // V0 = [ar, ai], F0 = ar
    VDUP V0.D[1], V1.D2          // F1 = ai

    // Compute products
    FMULD F0, F20, F2            // F2 = ar * sr
    FMULD F1, F21, F3            // F3 = ai * si
    FMULD F0, F21, F4            // F4 = ar * si
    FMULD F1, F20, F5            // F5 = ai * sr

    // result_real = ar*sr - ai*si
    FSUBD F3, F2, F2             // F2 = ar*sr - ai*si

    // result_imag = ar*si + ai*sr
    FADDD F4, F5, F3             // F3 = ar*si + ai*sr

    // Store result
    FMOVD F2, (R0)
    FMOVD F3, 8(R0)

    ADD  $16, R2
    ADD  $16, R0
    SUB  $1, R1
    CBNZ R1, scale_neon_loop

scale_neon_done:
    RET

// func addNEON(dst, a, b []complex128)
// Vector add - can use NEON FADD since we're adding both real and imag
TEXT ·addNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R1
    MOVD a_base+24(FP), R2
    MOVD b_base+48(FP), R3

    CBZ  R1, add_neon_done

add_neon_loop:
    VLD1 (R2), [V0.D2]
    VLD1 (R3), [V1.D2]
    WORD $0x4E61D402             // FADD V2.2D, V0.2D, V1.2D
    VST1 [V2.D2], (R0)
    ADD  $16, R2
    ADD  $16, R3
    ADD  $16, R0
    SUB  $1, R1
    CBNZ R1, add_neon_loop

add_neon_done:
    RET

// func subNEON(dst, a, b []complex128)
// Vector sub - can use NEON FSUB since we're subtracting both real and imag
TEXT ·subNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R1
    MOVD a_base+24(FP), R2
    MOVD b_base+48(FP), R3

    CBZ  R1, sub_neon_done

sub_neon_loop:
    VLD1 (R2), [V0.D2]
    VLD1 (R3), [V1.D2]
    WORD $0x4EE1D402             // FSUB V2.2D, V0.2D, V1.2D
    VST1 [V2.D2], (R0)
    ADD  $16, R2
    ADD  $16, R3
    ADD  $16, R0
    SUB  $1, R1
    CBNZ R1, sub_neon_loop

sub_neon_done:
    RET

// ============================================================================
// ABS - COMPLEX MAGNITUDE: |a + bi| = sqrt(a² + b²)
// ============================================================================

// func absNEON(dst []float64, a []complex128)
TEXT ·absNEON(SB), NOSPLIT, $0-56
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R1
    MOVD a_base+24(FP), R2

    CBZ  R1, abs_neon_done

abs_neon_loop:
    VLD1 (R2), [V0.D2]           // V0 = [real, imag], F0 = real
    VDUP V0.D[1], V1.D2          // F1 = imag

    // F2 = real * real
    FMULD F0, F0, F2

    // F3 = imag * imag
    FMULD F1, F1, F3

    // F4 = real² + imag²
    FADDD F2, F3, F4

    // F5 = sqrt(real² + imag²)
    FSQRTD F4, F5

    // Store result (single float64)
    FMOVD F5, (R0)

    ADD  $16, R2
    ADD  $8, R0
    SUB  $1, R1
    CBNZ R1, abs_neon_loop

abs_neon_done:
    RET

// ============================================================================
// ABSSQ - MAGNITUDE SQUARED: |a + bi|² = a² + b²
// ============================================================================

// func absSqNEON(dst []float64, a []complex128)
TEXT ·absSqNEON(SB), NOSPLIT, $0-56
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R1
    MOVD a_base+24(FP), R2

    CBZ  R1, abssq_neon_done

abssq_neon_loop:
    VLD1 (R2), [V0.D2]           // V0 = [real, imag], F0 = real
    VDUP V0.D[1], V1.D2          // F1 = imag

    // F2 = real * real
    FMULD F0, F0, F2

    // F3 = imag * imag
    FMULD F1, F1, F3

    // F4 = real² + imag²
    FADDD F2, F3, F4

    // Store result (single float64)
    FMOVD F4, (R0)

    ADD  $16, R2
    ADD  $8, R0
    SUB  $1, R1
    CBNZ R1, abssq_neon_loop

abssq_neon_done:
    RET

// ============================================================================
// CONJ - COMPLEX CONJUGATE: conj(a + bi) = a - bi
// ============================================================================

// func conjNEON(dst, a []complex128)
TEXT ·conjNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R1
    MOVD a_base+24(FP), R2

    CBZ  R1, conj_neon_done

conj_neon_loop:
    VLD1 (R2), [V0.D2]           // V0 = [real, imag], F0 = real
    VDUP V0.D[1], V1.D2          // F1 = imag

    // Negate imaginary part
    FNEGD F1, F1

    // Store result - real unchanged, imag negated
    FMOVD F0, (R0)               // Store real
    FMOVD F1, 8(R0)              // Store -imag

    ADD  $16, R2
    ADD  $16, R0
    SUB  $1, R1
    CBNZ R1, conj_neon_loop

conj_neon_done:
    RET
