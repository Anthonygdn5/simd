//go:build arm64

package f16

import "github.com/tphakala/simd/cpu"

var (
	hasFP16 = cpu.ARM64.FP16
	hasNEON = cpu.ARM64.NEON
)

func toFloat32(h Float16) float32 {
	return toFloat32Go(h)
}

func fromFloat32(f float32) Float16 {
	return fromFloat32Go(f)
}

func toFloat32Slice(dst []float32, src []Float16) {
	n := len(dst)
	if hasFP16 && n >= 8 {
		// Process vectorized portion (multiples of 8)
		n8 := (n / 8) * 8
		toFloat32SliceNEON(dst[:n8], src[:n8])
		// Handle remainder with Go
		toFloat32SliceGo(dst[n8:], src[n8:])
		return
	}
	toFloat32SliceGo(dst, src)
}

func fromFloat32Slice(dst []Float16, src []float32) {
	n := len(dst)
	if hasFP16 && n >= 8 {
		// Process vectorized portion (multiples of 8)
		n8 := (n / 8) * 8
		fromFloat32SliceNEON(dst[:n8], src[:n8])
		// Handle remainder with Go
		fromFloat32SliceGo(dst[n8:], src[n8:])
		return
	}
	fromFloat32SliceGo(dst, src)
}

func dotProduct(a, b []Float16) float32 {
	n := min(len(a), len(b))
	if hasFP16 && n >= 8 {
		// Process vectorized portion
		n8 := (n / 8) * 8
		result := dotProductNEON(a[:n8], b[:n8])
		// Handle remainder with Go
		result += dotProductGo(a[n8:n], b[n8:n])
		return result
	}
	return dotProductGo(a, b)
}

func add(dst, a, b []Float16) {
	n := len(dst)
	if hasFP16 && n >= 8 {
		n8 := (n / 8) * 8
		addNEON(dst[:n8], a[:n8], b[:n8])
		addGo(dst[n8:], a[n8:], b[n8:])
		return
	}
	addGo(dst, a, b)
}

func sub(dst, a, b []Float16) {
	n := len(dst)
	if hasFP16 && n >= 8 {
		n8 := (n / 8) * 8
		subNEON(dst[:n8], a[:n8], b[:n8])
		subGo(dst[n8:], a[n8:], b[n8:])
		return
	}
	subGo(dst, a, b)
}

func mul(dst, a, b []Float16) {
	n := len(dst)
	if hasFP16 && n >= 8 {
		n8 := (n / 8) * 8
		mulNEON(dst[:n8], a[:n8], b[:n8])
		mulGo(dst[n8:], a[n8:], b[n8:])
		return
	}
	mulGo(dst, a, b)
}

func scale(dst, a []Float16, s Float16) {
	n := len(dst)
	if hasFP16 && n >= 8 {
		n8 := (n / 8) * 8
		scaleNEON(dst[:n8], a[:n8], s)
		scaleGo(dst[n8:], a[n8:], s)
		return
	}
	scaleGo(dst, a, s)
}

func fma16(dst, a, b, c []Float16) {
	n := len(dst)
	if hasFP16 && n >= 8 {
		n8 := (n / 8) * 8
		fmaNEON(dst[:n8], a[:n8], b[:n8], c[:n8])
		fmaGo(dst[n8:], a[n8:], b[n8:], c[n8:])
		return
	}
	fmaGo(dst, a, b, c)
}

func sum(a []Float16) float32 {
	n := len(a)
	if hasFP16 && n >= 8 {
		n8 := (n / 8) * 8
		result := sumNEON(a[:n8])
		result += sumGo(a[n8:])
		return result
	}
	return sumGo(a)
}

func abs16(dst, a []Float16) {
	n := len(dst)
	if hasFP16 && n >= 8 {
		n8 := (n / 8) * 8
		absNEON(dst[:n8], a[:n8])
		absGo(dst[n8:], a[n8:])
		return
	}
	absGo(dst, a)
}

func neg16(dst, a []Float16) {
	n := len(dst)
	if hasFP16 && n >= 8 {
		n8 := (n / 8) * 8
		negNEON(dst[:n8], a[:n8])
		negGo(dst[n8:], a[n8:])
		return
	}
	negGo(dst, a)
}

func relu16(dst, src []Float16) {
	n := len(dst)
	if hasFP16 && n >= 8 {
		n8 := (n / 8) * 8
		reluNEON(dst[:n8], src[:n8])
		reluGo(dst[n8:], src[n8:])
		return
	}
	reluGo(dst, src)
}

func sigmoid16(dst, src []Float16) {
	// Sigmoid is complex - use Go fallback for now
	sigmoidGo(dst, src)
}

func min16(a []Float16) Float16 {
	n := len(a)
	if hasFP16 && n >= 8 {
		n8 := (n / 8) * 8
		minVec := minNEON(a[:n8])
		if n8 < n {
			minRem := minGo(a[n8:])
			if toFloat32Go(minRem) < toFloat32Go(minVec) {
				return minRem
			}
		}
		return minVec
	}
	return minGo(a)
}

func max16(a []Float16) Float16 {
	n := len(a)
	if hasFP16 && n >= 8 {
		n8 := (n / 8) * 8
		maxVec := maxNEON(a[:n8])
		if n8 < n {
			maxRem := maxGo(a[n8:])
			if toFloat32Go(maxRem) > toFloat32Go(maxVec) {
				return maxRem
			}
		}
		return maxVec
	}
	return maxGo(a)
}

func div16(dst, a, b []Float16) {
	n := len(dst)
	if hasFP16 && n >= 8 {
		n8 := (n / 8) * 8
		divNEON(dst[:n8], a[:n8], b[:n8])
		divGo(dst[n8:], a[n8:], b[n8:])
		return
	}
	divGo(dst, a, b)
}

func addScalar16(dst, a []Float16, s Float16) {
	n := len(dst)
	if hasFP16 && n >= 8 {
		n8 := (n / 8) * 8
		addScalarNEON(dst[:n8], a[:n8], s)
		addScalarGo(dst[n8:], a[n8:], s)
		return
	}
	addScalarGo(dst, a, s)
}

func clamp16(dst, a []Float16, minVal, maxVal Float16) {
	n := len(dst)
	if hasFP16 && n >= 8 {
		n8 := (n / 8) * 8
		clampNEON(dst[:n8], a[:n8], minVal, maxVal)
		clampGo(dst[n8:], a[n8:], minVal, maxVal)
		return
	}
	clampGo(dst, a, minVal, maxVal)
}

func sqrt16(dst, a []Float16) {
	n := len(dst)
	if hasFP16 && n >= 8 {
		n8 := (n / 8) * 8
		sqrtNEON(dst[:n8], a[:n8])
		sqrtGo(dst[n8:], a[n8:])
		return
	}
	sqrtGo(dst, a)
}

func reciprocal16(dst, a []Float16) {
	n := len(dst)
	if hasFP16 && n >= 8 {
		n8 := (n / 8) * 8
		reciprocalNEON(dst[:n8], a[:n8])
		reciprocalGo(dst[n8:], a[n8:])
		return
	}
	reciprocalGo(dst, a)
}

func exp16(dst, src []Float16) {
	// Exp requires polynomial approximation - use Go for now
	expGo(dst, src)
}

func tanh16(dst, src []Float16) {
	// Tanh requires polynomial approximation - use Go for now
	tanhGo(dst, src)
}

func minIdx16(a []Float16) int {
	// MinIdx doesn't vectorize well - use Go
	return minIdxGo(a)
}

func maxIdx16(a []Float16) int {
	// MaxIdx doesn't vectorize well - use Go
	return maxIdxGo(a)
}

func addScaled16(dst []Float16, alpha Float16, s []Float16) {
	n := len(dst)
	if hasFP16 && n >= 8 {
		n8 := (n / 8) * 8
		addScaledNEON(dst[:n8], alpha, s[:n8])
		addScaledGo(dst[n8:], alpha, s[n8:])
		return
	}
	addScaledGo(dst, alpha, s)
}

func euclideanDistance16(a, b []Float16) float32 {
	// Use Go implementation - could optimize later
	return euclideanDistanceGo(a, b)
}

func variance16(a []Float16, mean float32) float32 {
	return varianceGo(a, mean)
}

func cumulativeSum16(dst, a []Float16) {
	// Cumulative sum is inherently sequential
	cumulativeSumGo(dst, a)
}

func dotProductBatch16(results []float32, rows [][]Float16, vec []Float16) {
	// Use per-row SIMD via dotProduct
	for i, row := range rows {
		results[i] = dotProduct(row, vec)
	}
}

func accumulateAdd16(dst, src []Float16) {
	n := len(src)
	if hasFP16 && n >= 8 {
		n8 := (n / 8) * 8
		accumulateAddNEON(dst[:n8], src[:n8])
		accumulateAddGo(dst[n8:], src[n8:])
		return
	}
	accumulateAddGo(dst, src)
}

func convolveValid16(dst []Float16, signal, kernel []Float16) {
	convolveValidGo(dst, signal, kernel)
}

func interleave2_16(dst, a, b []Float16) {
	interleave2Go(dst, a, b)
}

func deinterleave2_16(a, b, src []Float16) {
	deinterleave2Go(a, b, src)
}

func clampScale16(dst, src []Float16, minVal, maxVal, scale Float16) {
	clampScaleGo(dst, src, minVal, maxVal, scale)
}

//go:noescape
func toFloat32SliceNEON(dst []float32, src []Float16)

//go:noescape
func fromFloat32SliceNEON(dst []Float16, src []float32)

//go:noescape
func dotProductNEON(a, b []Float16) float32

//go:noescape
func addNEON(dst, a, b []Float16)

//go:noescape
func subNEON(dst, a, b []Float16)

//go:noescape
func mulNEON(dst, a, b []Float16)

//go:noescape
func scaleNEON(dst, a []Float16, s Float16)

//go:noescape
func fmaNEON(dst, a, b, c []Float16)

//go:noescape
func sumNEON(a []Float16) float32

//go:noescape
func absNEON(dst, a []Float16)

//go:noescape
func negNEON(dst, a []Float16)

//go:noescape
func reluNEON(dst, src []Float16)

//go:noescape
func minNEON(a []Float16) Float16

//go:noescape
func maxNEON(a []Float16) Float16

//go:noescape
func divNEON(dst, a, b []Float16)

//go:noescape
func addScalarNEON(dst, a []Float16, s Float16)

//go:noescape
func clampNEON(dst, a []Float16, minVal, maxVal Float16)

//go:noescape
func sqrtNEON(dst, a []Float16)

//go:noescape
func reciprocalNEON(dst, a []Float16)

//go:noescape
func addScaledNEON(dst []Float16, alpha Float16, s []Float16)

//go:noescape
func accumulateAddNEON(dst, src []Float16)
