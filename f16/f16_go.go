package f16

import "math"

// IEEE 754 half-precision format:
// Sign: 1 bit (bit 15)
// Exponent: 5 bits (bits 14-10), bias = 15
// Mantissa: 10 bits (bits 9-0), implicit leading 1 for normalized numbers

// toFloat32Go converts Float16 to float32 using pure Go.
func toFloat32Go(h Float16) float32 {
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1F
	mant := uint32(h) & 0x3FF

	var f32bits uint32

	switch exp {
	case 0:
		if mant == 0 {
			// Zero (positive or negative)
			f32bits = sign << 31
		} else {
			// Denormalized number - convert to normalized float32
			// Shift mantissa left until we get a leading 1
			exp = 1
			for mant&0x400 == 0 {
				mant <<= 1
				exp--
			}
			mant &= 0x3FF                   // Remove the leading 1
			f32bits = (sign << 31) | ((127 - 15 + exp) << 23) | (mant << 13)
		}
	case 31:
		// Infinity or NaN
		if mant == 0 {
			// Infinity
			f32bits = (sign << 31) | 0x7F800000
		} else {
			// NaN - preserve payload
			f32bits = (sign << 31) | 0x7F800000 | (mant << 13)
		}
	default:
		// Normalized number
		// Convert exponent: subtract FP16 bias (15), add FP32 bias (127)
		f32bits = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
	}

	return math.Float32frombits(f32bits)
}

// fromFloat32Go converts float32 to Float16 using pure Go.
func fromFloat32Go(f float32) Float16 {
	f32bits := math.Float32bits(f)
	sign := uint16((f32bits >> 16) & 0x8000)
	exp := int((f32bits >> 23) & 0xFF)
	mant := f32bits & 0x7FFFFF

	switch {
	case exp == 255:
		// Infinity or NaN
		if mant == 0 {
			// Infinity
			return sign | 0x7C00
		}
		// NaN - preserve some payload bits
		return sign | 0x7C00 | uint16(mant>>13)

	case exp > 142:
		// Overflow to infinity (exp > 127 + 15)
		return sign | 0x7C00

	case exp < 103:
		// Underflow to zero (exp < 127 - 24, too small even for denorms)
		return sign

	case exp < 113:
		// Denormalized result
		// Add implicit 1 to mantissa
		mant |= 0x800000
		// Shift right to create denormalized value
		shift := uint(113 - exp)
		// Round to nearest even
		round := uint32(1) << (shift - 1)
		if mant&round != 0 && (mant&(round-1) != 0 || mant&(round<<1) != 0) {
			mant += round
		}
		mant >>= shift
		return sign | uint16(mant)

	default:
		// Normalized result
		// Round to nearest even
		round := uint32(0x1000) // bit 12
		if mant&round != 0 && (mant&0xFFF != 0 || mant&0x2000 != 0) {
			mant += round
			if mant >= 0x800000 {
				// Mantissa overflow, increment exponent
				mant = 0
				exp++
			}
		}
		mant >>= 13
		exp -= 127 - 15
		if exp > 30 {
			// Overflow to infinity
			return sign | 0x7C00
		}
		return sign | uint16(exp<<10) | uint16(mant)
	}
}

// toFloat32SliceGo converts a slice of Float16 to float32.
func toFloat32SliceGo(dst []float32, src []Float16) {
	for i := range dst {
		dst[i] = toFloat32Go(src[i])
	}
}

// fromFloat32SliceGo converts a slice of float32 to Float16.
func fromFloat32SliceGo(dst []Float16, src []float32) {
	for i := range dst {
		dst[i] = fromFloat32Go(src[i])
	}
}

// dotProductGo computes dot product using pure Go (converts to float32).
func dotProductGo(a, b []Float16) float32 {
	n := min(len(a), len(b))
	var sum float32
	for i := 0; i < n; i++ {
		sum += toFloat32Go(a[i]) * toFloat32Go(b[i])
	}
	return sum
}

// addGo computes element-wise addition.
func addGo(dst, a, b []Float16) {
	for i := range dst {
		dst[i] = fromFloat32Go(toFloat32Go(a[i]) + toFloat32Go(b[i]))
	}
}

// subGo computes element-wise subtraction.
func subGo(dst, a, b []Float16) {
	for i := range dst {
		dst[i] = fromFloat32Go(toFloat32Go(a[i]) - toFloat32Go(b[i]))
	}
}

// mulGo computes element-wise multiplication.
func mulGo(dst, a, b []Float16) {
	for i := range dst {
		dst[i] = fromFloat32Go(toFloat32Go(a[i]) * toFloat32Go(b[i]))
	}
}

// scaleGo multiplies each element by a scalar.
func scaleGo(dst, a []Float16, s Float16) {
	sf := toFloat32Go(s)
	for i := range dst {
		dst[i] = fromFloat32Go(toFloat32Go(a[i]) * sf)
	}
}

// fmaGo computes fused multiply-add.
func fmaGo(dst, a, b, c []Float16) {
	for i := range dst {
		dst[i] = fromFloat32Go(toFloat32Go(a[i])*toFloat32Go(b[i]) + toFloat32Go(c[i]))
	}
}

// sumGo returns the sum of all elements.
func sumGo(a []Float16) float32 {
	var sum float32
	for i := range a {
		sum += toFloat32Go(a[i])
	}
	return sum
}

// absGo computes element-wise absolute value.
func absGo(dst, a []Float16) {
	for i := range dst {
		// Clear sign bit
		dst[i] = a[i] & 0x7FFF
	}
}

// negGo computes element-wise negation.
func negGo(dst, a []Float16) {
	for i := range dst {
		// Flip sign bit
		dst[i] = a[i] ^ 0x8000
	}
}

// reluGo computes ReLU activation.
func reluGo(dst, src []Float16) {
	for i := range dst {
		if src[i]&0x8000 != 0 { // negative (sign bit set)
			dst[i] = 0
		} else {
			dst[i] = src[i]
		}
	}
}

// sigmoidGo computes sigmoid activation.
func sigmoidGo(dst, src []Float16) {
	for i := range dst {
		x := toFloat32Go(src[i])
		// sigmoid(x) = 1 / (1 + exp(-x))
		dst[i] = fromFloat32Go(1.0 / (1.0 + float32(math.Exp(float64(-x)))))
	}
}

// minGo returns the minimum value.
func minGo(a []Float16) Float16 {
	minVal := a[0]
	minF := toFloat32Go(minVal)
	for i := 1; i < len(a); i++ {
		f := toFloat32Go(a[i])
		if f < minF {
			minF = f
			minVal = a[i]
		}
	}
	return minVal
}

// maxGo returns the maximum value.
func maxGo(a []Float16) Float16 {
	maxVal := a[0]
	maxF := toFloat32Go(maxVal)
	for i := 1; i < len(a); i++ {
		f := toFloat32Go(a[i])
		if f > maxF {
			maxF = f
			maxVal = a[i]
		}
	}
	return maxVal
}

// divGo computes element-wise division.
func divGo(dst, a, b []Float16) {
	for i := range dst {
		dst[i] = fromFloat32Go(toFloat32Go(a[i]) / toFloat32Go(b[i]))
	}
}

// addScalarGo adds a scalar to each element.
func addScalarGo(dst, a []Float16, s Float16) {
	sf := toFloat32Go(s)
	for i := range dst {
		dst[i] = fromFloat32Go(toFloat32Go(a[i]) + sf)
	}
}

// clampGo clamps each element to [minVal, maxVal].
func clampGo(dst, a []Float16, minVal, maxVal Float16) {
	minF := toFloat32Go(minVal)
	maxF := toFloat32Go(maxVal)
	for i := range dst {
		f := toFloat32Go(a[i])
		if f < minF {
			f = minF
		} else if f > maxF {
			f = maxF
		}
		dst[i] = fromFloat32Go(f)
	}
}

// sqrtGo computes element-wise square root.
func sqrtGo(dst, a []Float16) {
	for i := range dst {
		dst[i] = fromFloat32Go(float32(math.Sqrt(float64(toFloat32Go(a[i])))))
	}
}

// reciprocalGo computes element-wise reciprocal.
func reciprocalGo(dst, a []Float16) {
	for i := range dst {
		dst[i] = fromFloat32Go(1.0 / toFloat32Go(a[i]))
	}
}

// expGo computes element-wise exponential.
func expGo(dst, src []Float16) {
	for i := range dst {
		dst[i] = fromFloat32Go(float32(math.Exp(float64(toFloat32Go(src[i])))))
	}
}

// tanhGo computes element-wise hyperbolic tangent.
func tanhGo(dst, src []Float16) {
	for i := range dst {
		dst[i] = fromFloat32Go(float32(math.Tanh(float64(toFloat32Go(src[i])))))
	}
}

// minIdxGo returns the index of the minimum value.
func minIdxGo(a []Float16) int {
	minIdx := 0
	minF := toFloat32Go(a[0])
	for i := 1; i < len(a); i++ {
		f := toFloat32Go(a[i])
		if f < minF {
			minF = f
			minIdx = i
		}
	}
	return minIdx
}

// maxIdxGo returns the index of the maximum value.
func maxIdxGo(a []Float16) int {
	maxIdx := 0
	maxF := toFloat32Go(a[0])
	for i := 1; i < len(a); i++ {
		f := toFloat32Go(a[i])
		if f > maxF {
			maxF = f
			maxIdx = i
		}
	}
	return maxIdx
}

// addScaledGo computes dst[i] += alpha * s[i].
func addScaledGo(dst []Float16, alpha Float16, s []Float16) {
	alphaF := toFloat32Go(alpha)
	for i := range dst {
		dst[i] = fromFloat32Go(toFloat32Go(dst[i]) + alphaF*toFloat32Go(s[i]))
	}
}

// euclideanDistanceGo computes Euclidean distance.
func euclideanDistanceGo(a, b []Float16) float32 {
	var sum float32
	for i := range a {
		d := toFloat32Go(a[i]) - toFloat32Go(b[i])
		sum += d * d
	}
	return float32(math.Sqrt(float64(sum)))
}

// varianceGo computes variance given the mean.
func varianceGo(a []Float16, mean float32) float32 {
	var sum float32
	for i := range a {
		d := toFloat32Go(a[i]) - mean
		sum += d * d
	}
	return sum / float32(len(a))
}

// cumulativeSumGo computes cumulative sum.
func cumulativeSumGo(dst, a []Float16) {
	var sum float32
	for i := range dst {
		sum += toFloat32Go(a[i])
		dst[i] = fromFloat32Go(sum)
	}
}

// dotProductBatchGo computes batch dot products.
func dotProductBatchGo(results []float32, rows [][]Float16, vec []Float16) {
	for i, row := range rows {
		results[i] = dotProductGo(row, vec)
	}
}

// accumulateAddGo adds src to dst[offset:].
func accumulateAddGo(dst, src []Float16) {
	for i := range src {
		dst[i] = fromFloat32Go(toFloat32Go(dst[i]) + toFloat32Go(src[i]))
	}
}

// convolveValidGo computes valid convolution.
func convolveValidGo(dst []Float16, signal, kernel []Float16) {
	kLen := len(kernel)
	for i := range dst {
		var sum float32
		for j := 0; j < kLen; j++ {
			sum += toFloat32Go(signal[i+j]) * toFloat32Go(kernel[j])
		}
		dst[i] = fromFloat32Go(sum)
	}
}

// interleave2Go interleaves two slices.
func interleave2Go(dst, a, b []Float16) {
	for i := range a {
		dst[i*2] = a[i]
		dst[i*2+1] = b[i]
	}
}

// deinterleave2Go deinterleaves a slice.
func deinterleave2Go(a, b, src []Float16) {
	for i := range a {
		a[i] = src[i*2]
		b[i] = src[i*2+1]
	}
}

// clampScaleGo performs fused clamp and scale.
func clampScaleGo(dst, src []Float16, minVal, maxVal, scale Float16) {
	minF := toFloat32Go(minVal)
	maxF := toFloat32Go(maxVal)
	scaleF := toFloat32Go(scale)
	for i := range dst {
		f := toFloat32Go(src[i])
		if f < minF {
			f = minF
		} else if f > maxF {
			f = maxF
		}
		dst[i] = fromFloat32Go((f - minF) * scaleF)
	}
}
