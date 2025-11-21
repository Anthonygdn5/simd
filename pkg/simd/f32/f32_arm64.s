//go:build arm64

#include "textflag.h"

// ARM64 NEON for float32: 4 elements per 128-bit register

// func dotProductNEON(a, b []float32) float32
TEXT ·dotProductNEON(SB), NOSPLIT, $0-28
    MOVD a_base+0(FP), R0
    MOVD a_len+8(FP), R2
    MOVD b_base+24(FP), R1

    VEOR V0.B16, V0.B16, V0.B16
    VEOR V1.B16, V1.B16, V1.B16

    // Process 8 elements (2 NEON ops) per iteration
    LSR $3, R2, R3
    CBZ R3, dot32_remainder4

dot32_loop8:
    VLD1.P 16(R0), [V2.S4]
    VLD1.P 16(R0), [V3.S4]
    VLD1.P 16(R1), [V4.S4]
    VLD1.P 16(R1), [V5.S4]
    WORD $0x4E24CC40           // FMLA V0.4S, V2.4S, V4.4S
    WORD $0x4E25CC61           // FMLA V1.4S, V3.4S, V5.4S
    SUB $1, R3
    CBNZ R3, dot32_loop8

    VFADD V0.S4, V1.S4, V0.S4

dot32_remainder4:
    AND $7, R2, R3
    LSR $2, R3, R4
    CBZ R4, dot32_remainder

    VLD1.P 16(R0), [V2.S4]
    VLD1.P 16(R1), [V4.S4]
    WORD $0x4E24CC40           // FMLA V0.4S, V2.4S, V4.4S

dot32_remainder:
    AND $3, R3, R4
    CBZ R4, dot32_reduce

dot32_scalar:
    FMOVS (R0), F2
    FMOVS (R1), F4
    FMADDS F2, F4, F0, F0
    ADD $4, R0
    ADD $4, R1
    SUB $1, R4
    CBNZ R4, dot32_scalar

dot32_reduce:
    // Horizontal sum of V0.S4
    WORD $0x7E30D800           // FADDP V0.2S, V0.2S -> actually need 4S reduction
    // Proper horizontal reduction for 4S
    VFADDP V0.S4, V0.S4, V0.S4
    VFADDP V0.S4, V0.S4, V0.S4

    FMOVS F0, ret+48(FP)
    RET

// func addNEON(dst, a, b []float32)
TEXT ·addNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1
    MOVD b_base+48(FP), R2

    LSR $2, R3, R4
    CBZ R4, add32_scalar

add32_loop4:
    VLD1.P 16(R1), [V0.S4]
    VLD1.P 16(R2), [V1.S4]
    VFADD V0.S4, V1.S4, V2.S4
    VST1.P [V2.S4], 16(R0)
    SUB $1, R4
    CBNZ R4, add32_loop4

add32_scalar:
    AND $3, R3
    CBZ R3, add32_done

add32_loop1:
    FMOVS (R1), F0
    FMOVS (R2), F1
    FADDS F0, F1, F0
    FMOVS F0, (R0)
    ADD $4, R0
    ADD $4, R1
    ADD $4, R2
    SUB $1, R3
    CBNZ R3, add32_loop1

add32_done:
    RET

// func subNEON(dst, a, b []float32)
TEXT ·subNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1
    MOVD b_base+48(FP), R2

    LSR $2, R3, R4
    CBZ R4, sub32_scalar

sub32_loop4:
    VLD1.P 16(R1), [V0.S4]
    VLD1.P 16(R2), [V1.S4]
    VFSUB V1.S4, V0.S4, V2.S4
    VST1.P [V2.S4], 16(R0)
    SUB $1, R4
    CBNZ R4, sub32_loop4

sub32_scalar:
    AND $3, R3
    CBZ R3, sub32_done

sub32_loop1:
    FMOVS (R1), F0
    FMOVS (R2), F1
    FSUBS F1, F0, F0
    FMOVS F0, (R0)
    ADD $4, R0
    ADD $4, R1
    ADD $4, R2
    SUB $1, R3
    CBNZ R3, sub32_loop1

sub32_done:
    RET

// func mulNEON(dst, a, b []float32)
TEXT ·mulNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1
    MOVD b_base+48(FP), R2

    LSR $2, R3, R4
    CBZ R4, mul32_scalar

mul32_loop4:
    VLD1.P 16(R1), [V0.S4]
    VLD1.P 16(R2), [V1.S4]
    VFMUL V0.S4, V1.S4, V2.S4
    VST1.P [V2.S4], 16(R0)
    SUB $1, R4
    CBNZ R4, mul32_loop4

mul32_scalar:
    AND $3, R3
    CBZ R3, mul32_done

mul32_loop1:
    FMOVS (R1), F0
    FMOVS (R2), F1
    FMULS F0, F1, F0
    FMOVS F0, (R0)
    ADD $4, R0
    ADD $4, R1
    ADD $4, R2
    SUB $1, R3
    CBNZ R3, mul32_loop1

mul32_done:
    RET

// func divNEON(dst, a, b []float32)
TEXT ·divNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1
    MOVD b_base+48(FP), R2

    LSR $2, R3, R4
    CBZ R4, div32_scalar

div32_loop4:
    VLD1.P 16(R1), [V0.S4]
    VLD1.P 16(R2), [V1.S4]
    VFDIV V1.S4, V0.S4, V2.S4
    VST1.P [V2.S4], 16(R0)
    SUB $1, R4
    CBNZ R4, div32_loop4

div32_scalar:
    AND $3, R3
    CBZ R3, div32_done

div32_loop1:
    FMOVS (R1), F0
    FMOVS (R2), F1
    FDIVS F1, F0, F0
    FMOVS F0, (R0)
    ADD $4, R0
    ADD $4, R1
    ADD $4, R2
    SUB $1, R3
    CBNZ R3, div32_loop1

div32_done:
    RET

// func scaleNEON(dst, a []float32, s float32)
TEXT ·scaleNEON(SB), NOSPLIT, $0-52
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R2
    MOVD a_base+24(FP), R1
    FMOVS s+48(FP), F3
    VDUP F3, V3.S4

    LSR $2, R2, R3
    CBZ R3, scale32_scalar

scale32_loop4:
    VLD1.P 16(R1), [V0.S4]
    VFMUL V0.S4, V3.S4, V1.S4
    VST1.P [V1.S4], 16(R0)
    SUB $1, R3
    CBNZ R3, scale32_loop4

scale32_scalar:
    AND $3, R2
    CBZ R2, scale32_done

scale32_loop1:
    FMOVS (R1), F0
    FMULS F0, F3, F0
    FMOVS F0, (R0)
    ADD $4, R0
    ADD $4, R1
    SUB $1, R2
    CBNZ R2, scale32_loop1

scale32_done:
    RET

// func addScalarNEON(dst, a []float32, s float32)
TEXT ·addScalarNEON(SB), NOSPLIT, $0-52
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R2
    MOVD a_base+24(FP), R1
    FMOVS s+48(FP), F3
    VDUP F3, V3.S4

    LSR $2, R2, R3
    CBZ R3, addsc32_scalar

addsc32_loop4:
    VLD1.P 16(R1), [V0.S4]
    VFADD V0.S4, V3.S4, V1.S4
    VST1.P [V1.S4], 16(R0)
    SUB $1, R3
    CBNZ R3, addsc32_loop4

addsc32_scalar:
    AND $3, R2
    CBZ R2, addsc32_done

addsc32_loop1:
    FMOVS (R1), F0
    FADDS F0, F3, F0
    FMOVS F0, (R0)
    ADD $4, R0
    ADD $4, R1
    SUB $1, R2
    CBNZ R2, addsc32_loop1

addsc32_done:
    RET

// func sumNEON(a []float32) float32
TEXT ·sumNEON(SB), NOSPLIT, $0-28
    MOVD a_base+0(FP), R0
    MOVD a_len+8(FP), R1

    VEOR V0.B16, V0.B16, V0.B16

    LSR $2, R1, R2
    CBZ R2, sum32_scalar

sum32_loop4:
    VLD1.P 16(R0), [V1.S4]
    VFADD V0.S4, V1.S4, V0.S4
    SUB $1, R2
    CBNZ R2, sum32_loop4

sum32_scalar:
    AND $3, R1
    CBZ R1, sum32_reduce

sum32_loop1:
    FMOVS (R0), F1
    FADDS F0, F1, F0
    ADD $4, R0
    SUB $1, R1
    CBNZ R1, sum32_loop1

sum32_reduce:
    VFADDP V0.S4, V0.S4, V0.S4
    VFADDP V0.S4, V0.S4, V0.S4
    FMOVS F0, ret+24(FP)
    RET

// func minNEON(a []float32) float32
TEXT ·minNEON(SB), NOSPLIT, $0-28
    MOVD a_base+0(FP), R0
    MOVD a_len+8(FP), R1

    VLD1.P 16(R0), [V0.S4]
    SUB $4, R1

    LSR $2, R1, R2
    CBZ R2, min32_scalar

min32_loop4:
    VLD1.P 16(R0), [V1.S4]
    VFMIN V0.S4, V1.S4, V0.S4
    SUB $1, R2
    CBNZ R2, min32_loop4

min32_scalar:
    AND $3, R1
    CBZ R1, min32_reduce

min32_loop1:
    FMOVS (R0), F1
    FMINS F0, F1, F0
    ADD $4, R0
    SUB $1, R1
    CBNZ R1, min32_loop1

min32_reduce:
    VFMINP V0.S4, V0.S4, V0.S4
    VFMINP V0.S4, V0.S4, V0.S4
    FMOVS F0, ret+24(FP)
    RET

// func maxNEON(a []float32) float32
TEXT ·maxNEON(SB), NOSPLIT, $0-28
    MOVD a_base+0(FP), R0
    MOVD a_len+8(FP), R1

    VLD1.P 16(R0), [V0.S4]
    SUB $4, R1

    LSR $2, R1, R2
    CBZ R2, max32_scalar

max32_loop4:
    VLD1.P 16(R0), [V1.S4]
    VFMAX V0.S4, V1.S4, V0.S4
    SUB $1, R2
    CBNZ R2, max32_loop4

max32_scalar:
    AND $3, R1
    CBZ R1, max32_reduce

max32_loop1:
    FMOVS (R0), F1
    FMAXS F0, F1, F0
    ADD $4, R0
    SUB $1, R1
    CBNZ R1, max32_loop1

max32_reduce:
    VFMAXP V0.S4, V0.S4, V0.S4
    VFMAXP V0.S4, V0.S4, V0.S4
    FMOVS F0, ret+24(FP)
    RET

// func absNEON(dst, a []float32)
TEXT ·absNEON(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R2
    MOVD a_base+24(FP), R1

    LSR $2, R2, R3
    CBZ R3, abs32_scalar

abs32_loop4:
    VLD1.P 16(R1), [V0.S4]
    VFABS V0.S4, V1.S4
    VST1.P [V1.S4], 16(R0)
    SUB $1, R3
    CBNZ R3, abs32_loop4

abs32_scalar:
    AND $3, R2
    CBZ R2, abs32_done

abs32_loop1:
    FMOVS (R1), F0
    FABSS F0, F0
    FMOVS F0, (R0)
    ADD $4, R0
    ADD $4, R1
    SUB $1, R2
    CBNZ R2, abs32_loop1

abs32_done:
    RET

// func negNEON(dst, a []float32)
TEXT ·negNEON(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R2
    MOVD a_base+24(FP), R1

    LSR $2, R2, R3
    CBZ R3, neg32_scalar

neg32_loop4:
    VLD1.P 16(R1), [V0.S4]
    VFNEG V0.S4, V1.S4
    VST1.P [V1.S4], 16(R0)
    SUB $1, R3
    CBNZ R3, neg32_loop4

neg32_scalar:
    AND $3, R2
    CBZ R2, neg32_done

neg32_loop1:
    FMOVS (R1), F0
    FNEGS F0, F0
    FMOVS F0, (R0)
    ADD $4, R0
    ADD $4, R1
    SUB $1, R2
    CBNZ R2, neg32_loop1

neg32_done:
    RET

// func fmaNEON(dst, a, b, c []float32)
TEXT ·fmaNEON(SB), NOSPLIT, $0-96
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R4
    MOVD a_base+24(FP), R1
    MOVD b_base+48(FP), R2
    MOVD c_base+72(FP), R3

    LSR $2, R4, R5
    CBZ R5, fma32_scalar

fma32_loop4:
    VLD1.P 16(R1), [V0.S4]
    VLD1.P 16(R2), [V1.S4]
    VLD1.P 16(R3), [V2.S4]
    WORD $0x4E21CC02           // FMLA V2.4S, V0.4S, V1.4S
    VST1.P [V2.S4], 16(R0)
    SUB $1, R5
    CBNZ R5, fma32_loop4

fma32_scalar:
    AND $3, R4
    CBZ R4, fma32_done

fma32_loop1:
    FMOVS (R1), F0
    FMOVS (R2), F1
    FMOVS (R3), F2
    FMADDS F0, F1, F2, F2
    FMOVS F2, (R0)
    ADD $4, R0
    ADD $4, R1
    ADD $4, R2
    ADD $4, R3
    SUB $1, R4
    CBNZ R4, fma32_loop1

fma32_done:
    RET

// func clampNEON(dst, a []float32, minVal, maxVal float32)
TEXT ·clampNEON(SB), NOSPLIT, $0-56
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R2
    MOVD a_base+24(FP), R1
    FMOVS minVal+48(FP), F2
    FMOVS maxVal+52(FP), F3
    VDUP F2, V2.S4
    VDUP F3, V3.S4

    LSR $2, R2, R3
    CBZ R3, clamp32_scalar

clamp32_loop4:
    VLD1.P 16(R1), [V0.S4]
    VFMAX V0.S4, V2.S4, V0.S4
    VFMIN V0.S4, V3.S4, V0.S4
    VST1.P [V0.S4], 16(R0)
    SUB $1, R3
    CBNZ R3, clamp32_loop4

clamp32_scalar:
    AND $3, R2
    CBZ R2, clamp32_done

clamp32_loop1:
    FMOVS (R1), F0
    FMAXS F0, F2, F0
    FMINS F0, F3, F0
    FMOVS F0, (R0)
    ADD $4, R0
    ADD $4, R1
    SUB $1, R2
    CBNZ R2, clamp32_loop1

clamp32_done:
    RET
