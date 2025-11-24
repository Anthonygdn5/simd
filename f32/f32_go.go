package f32

import "math"

// Loop unroll factors for SIMD-width matching
const (
	unrollFactor = 8 // Match AVX 256-bit = 8 x float32
	unrollMask   = unrollFactor - 1
)

// Numerical stability thresholds
const (
	sigmoidClampThreshold = 20.0 // sigmoid(x > 20) ≈ 1, sigmoid(x < -20) ≈ 0
	tanhClampThreshold    = 2.5  // tanh(x > 2.5) ≈ 1, tanh(x < -2.5) ≈ -1
	expOverflowThreshold  = 88.0 // exp(88) ≈ max float32, prevents overflow
)

// Pure Go implementations

func dotProductGo(a, b []float32) float32 {
	var sum float32
	n := min(len(a), len(b))
	n8 := n &^ unrollMask

	for i := 0; i < n8; i += 8 {
		sum += a[i] * b[i]
		sum += a[i+1] * b[i+1]
		sum += a[i+2] * b[i+2]
		sum += a[i+3] * b[i+3]
		sum += a[i+4] * b[i+4]
		sum += a[i+5] * b[i+5]
		sum += a[i+6] * b[i+6]
		sum += a[i+7] * b[i+7]
	}

	for i := n8; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

func addGo(dst, a, b []float32) {
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}

func subGo(dst, a, b []float32) {
	for i := range dst {
		dst[i] = a[i] - b[i]
	}
}

func mulGo(dst, a, b []float32) {
	for i := range dst {
		dst[i] = a[i] * b[i]
	}
}

func divGo(dst, a, b []float32) {
	for i := range dst {
		dst[i] = a[i] / b[i]
	}
}

func scaleGo(dst, a []float32, s float32) {
	for i := range dst {
		dst[i] = a[i] * s
	}
}

func addScalarGo(dst, a []float32, s float32) {
	for i := range dst {
		dst[i] = a[i] + s
	}
}

func sumGo(a []float32) float32 {
	var sum float32
	for _, v := range a {
		sum += v
	}
	return sum
}

func minGo(a []float32) float32 {
	m := a[0]
	for _, v := range a[1:] {
		if v < m {
			m = v
		}
	}
	return m
}

func maxGo(a []float32) float32 {
	m := a[0]
	for _, v := range a[1:] {
		if v > m {
			m = v
		}
	}
	return m
}

func absGo(dst, a []float32) {
	for i := range dst {
		dst[i] = float32(math.Abs(float64(a[i])))
	}
}

func negGo(dst, a []float32) {
	for i := range dst {
		dst[i] = -a[i]
	}
}

func fmaGo(dst, a, b, c []float32) {
	for i := range dst {
		dst[i] = a[i]*b[i] + c[i]
	}
}

func clampGo(dst, a []float32, minVal, maxVal float32) {
	for i := range dst {
		v := a[i]
		if v < minVal {
			v = minVal
		} else if v > maxVal {
			v = maxVal
		}
		dst[i] = v
	}
}

func dotProductBatch32Go(results []float32, rows [][]float32, vec []float32) {
	vecLen := len(vec)
	for i, row := range rows {
		n := min(len(row), vecLen)
		if n == 0 {
			results[i] = 0
			continue
		}
		results[i] = dotProductGo(row[:n], vec[:n])
	}
}

func convolveValid32Go(dst, signal, kernel []float32) {
	kLen := len(kernel)
	for i := range dst {
		dst[i] = dotProductGo(signal[i:i+kLen], kernel)
	}
}

func accumulateAdd32Go(dst, src []float32) {
	for i := range dst {
		dst[i] += src[i]
	}
}

func interleave2Go(dst, a, b []float32) {
	for i := range a {
		dst[i*2] = a[i]
		dst[i*2+1] = b[i]
	}
}

func deinterleave2Go(a, b, src []float32) {
	for i := range a {
		a[i] = src[i*2]
		b[i] = src[i*2+1]
	}
}

func convolveValidMultiGo(dsts [][]float32, signal []float32, kernels [][]float32, n, _ int) {
	// Kernel-major loop order: each kernel stays hot in cache for entire signal pass
	for k, kernel := range kernels {
		convolveValid32Go(dsts[k][:n], signal, kernel)
	}
}

func sqrt32Go(dst, a []float32) {
	for i := range dst {
		dst[i] = float32(math.Sqrt(float64(a[i])))
	}
}

func reciprocal32Go(dst, a []float32) {
	for i := range dst {
		dst[i] = 1.0 / a[i]
	}
}

func minIdxGo(a []float32) int {
	if len(a) == 0 {
		return -1
	}
	idx := 0
	m := a[0]
	for i, v := range a[1:] {
		if v < m {
			m = v
			idx = i + 1
		}
	}
	return idx
}

func maxIdxGo(a []float32) int {
	if len(a) == 0 {
		return -1
	}
	idx := 0
	m := a[0]
	for i, v := range a[1:] {
		if v > m {
			m = v
			idx = i + 1
		}
	}
	return idx
}

func addScaledGo(dst []float32, alpha float32, s []float32) {
	for i := range dst {
		dst[i] += alpha * s[i]
	}
}

func cumulativeSum32Go(dst, a []float32) {
	if len(dst) == 0 {
		return
	}
	sum := float32(0)
	for i := range dst {
		sum += a[i]
		dst[i] = sum
	}
}

func variance32Go(a []float32, mean float32) float32 {
	var sum float32
	for _, v := range a {
		diff := v - mean
		sum += diff * diff
	}
	return sum / float32(len(a))
}

func euclideanDistance32Go(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum)))
}

// cubicInterpDotGo computes: Σ hist[i] * (a[i] + x*(b[i] + x*(c[i] + x*d[i])))
// Uses Horner's method for numerical stability.
func cubicInterpDotGo(hist, a, b, c, d []float32, x float32) float32 {
	var sum float32
	n := len(hist)
	n8 := n &^ unrollMask // Round down to multiple of 8

	// Unrolled loop: 8 elements per iteration (match AVX width)
	for i := 0; i < n8; i += 8 {
		// Horner's method: coef = a + x*(b + x*(c + x*d))
		coef0 := a[i] + x*(b[i]+x*(c[i]+x*d[i]))
		coef1 := a[i+1] + x*(b[i+1]+x*(c[i+1]+x*d[i+1]))
		coef2 := a[i+2] + x*(b[i+2]+x*(c[i+2]+x*d[i+2]))
		coef3 := a[i+3] + x*(b[i+3]+x*(c[i+3]+x*d[i+3]))
		coef4 := a[i+4] + x*(b[i+4]+x*(c[i+4]+x*d[i+4]))
		coef5 := a[i+5] + x*(b[i+5]+x*(c[i+5]+x*d[i+5]))
		coef6 := a[i+6] + x*(b[i+6]+x*(c[i+6]+x*d[i+6]))
		coef7 := a[i+7] + x*(b[i+7]+x*(c[i+7]+x*d[i+7]))

		sum += hist[i]*coef0 + hist[i+1]*coef1 + hist[i+2]*coef2 + hist[i+3]*coef3
		sum += hist[i+4]*coef4 + hist[i+5]*coef5 + hist[i+6]*coef6 + hist[i+7]*coef7
	}

	// Handle remainder
	for i := n8; i < n; i++ {
		coef := a[i] + x*(b[i]+x*(c[i]+x*d[i]))
		sum += hist[i] * coef
	}

	return sum
}

// sigmoid32Go computes sigmoid(x) = 1 / (1 + e^(-x)) using math.Exp.
// This is accurate but slower than SIMD approximations.
func sigmoid32Go(dst, src []float32) {
	for i := range dst {
		x := src[i]
		// Clamp extreme values for numerical stability
		switch {
		case x > sigmoidClampThreshold:
			dst[i] = 1.0
		case x < -sigmoidClampThreshold:
			dst[i] = 0.0
		default:
			dst[i] = float32(1.0 / (1.0 + math.Exp(float64(-x))))
		}
	}
}

// relu32Go computes ReLU(x) = max(0, x).
func relu32Go(dst, src []float32) {
	for i := range dst {
		if src[i] > 0 {
			dst[i] = src[i]
		} else {
			dst[i] = 0
		}
	}
}

// clampScale32Go performs fused clamp and scale operation.
func clampScale32Go(dst, src []float32, minVal, maxVal, scale float32) {
	for i := range dst {
		v := src[i]
		if v < minVal {
			v = minVal
		} else if v > maxVal {
			v = maxVal
		}
		dst[i] = (v - minVal) * scale
	}
}

// tanh32Go computes tanh(x) using fast approximation.
func tanh32Go(dst, src []float32) {
	for i := range dst {
		x := src[i]
		// Fast approximation: tanh(x) ≈ x / (1 + |x|) with clamping
		switch {
		case x > tanhClampThreshold:
			dst[i] = 1.0
		case x < -tanhClampThreshold:
			dst[i] = -1.0
		default:
			// tanh(x) ≈ x / (1 + |x|) for moderate values
			absX := float32(math.Abs(float64(x)))
			dst[i] = x / (1.0 + absX)
		}
	}
}

// exp32Go computes e^x using math.Exp.
func exp32Go(dst, src []float32) {
	for i := range dst {
		x := src[i]
		// Clamp extreme values
		switch {
		case x > expOverflowThreshold:
			dst[i] = float32(math.Exp(expOverflowThreshold)) // Prevent overflow
		case x < -expOverflowThreshold:
			dst[i] = 0.0 // Prevent underflow
		default:
			dst[i] = float32(math.Exp(float64(x)))
		}
	}
}
