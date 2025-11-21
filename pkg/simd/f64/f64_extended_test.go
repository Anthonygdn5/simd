package f64

import (
	"math"
	"testing"
)

// TestSqrt validates Sqrt operation
func TestSqrt(t *testing.T) {
	tests := []struct {
		name string
		a    []float64
		want []float64
	}{
		{"empty", nil, nil},
		{"single", []float64{4}, []float64{2}},
		{"perfect_squares", []float64{0, 1, 4, 9, 16, 25}, []float64{0, 1, 2, 3, 4, 5}},
		{"decimals", []float64{0.25, 0.5, 2, 8}, []float64{0.5, 0.7071067811865476, 1.4142135623730951, 2.8284271247461903}},
		{"large", []float64{1e6, 1e8, 1e10}, []float64{1000, 10000, 100000}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, len(tt.a))
			Sqrt(dst, tt.a)
			for i := range dst {
				if i < len(tt.want) && !almostEqual(dst[i], tt.want[i], 1e-14) {
					t.Errorf("Sqrt()[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

// TestReciprocal validates Reciprocal operation
func TestReciprocal(t *testing.T) {
	tests := []struct {
		name string
		a    []float64
		want []float64
	}{
		{"simple", []float64{1, 2, 4, 5, 10}, []float64{1, 0.5, 0.25, 0.2, 0.1}},
		{"negative", []float64{-1, -2, -4}, []float64{-1, -0.5, -0.25}},
		{"decimals", []float64{0.5, 0.25, 0.1}, []float64{2, 4, 10}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, len(tt.a))
			Reciprocal(dst, tt.a)
			for i := range dst {
				if i < len(tt.want) && !almostEqual(dst[i], tt.want[i], 1e-14) {
					t.Errorf("Reciprocal()[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

// TestMean validates Mean operation
func TestMean(t *testing.T) {
	tests := []struct {
		name string
		a    []float64
		want float64
	}{
		{"empty", nil, 0},
		{"single", []float64{5}, 5},
		{"simple", []float64{1, 2, 3, 4, 5}, 3},
		{"negative", []float64{-5, -3, -1, 1, 3, 5}, 0},
		{"decimals", []float64{1.5, 2.5, 3.5}, 2.5},
		{"large", make100(), 50.5}, // 1 to 100, mean = 50.5
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Mean(tt.a)
			if !almostEqual(got, tt.want, 1e-14) {
				t.Errorf("Mean() = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestVariance validates Variance operation
func TestVariance(t *testing.T) {
	tests := []struct {
		name string
		a    []float64
		want float64
	}{
		{"empty", nil, 0},
		{"single", []float64{5}, 0},
		{"constant", []float64{3, 3, 3, 3}, 0},
		{"simple", []float64{1, 2, 3, 4, 5}, 2}, // Population variance
		{"binary", []float64{0, 0, 1, 1}, 0.25},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Variance(tt.a)
			if !almostEqual(got, tt.want, 1e-14) {
				t.Errorf("Variance() = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestStdDev validates StdDev operation
func TestStdDev(t *testing.T) {
	tests := []struct {
		name string
		a    []float64
		want float64
	}{
		{"empty", nil, 0},
		{"single", []float64{5}, 0},
		{"constant", []float64{3, 3, 3, 3}, 0},
		{"simple", []float64{1, 2, 3, 4, 5}, math.Sqrt(2)}, // sqrt(variance)
		{"binary", []float64{0, 0, 1, 1}, 0.5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := StdDev(tt.a)
			if !almostEqual(got, tt.want, 1e-14) {
				t.Errorf("StdDev() = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestEuclideanDistance validates EuclideanDistance operation
func TestEuclideanDistance(t *testing.T) {
	tests := []struct {
		name string
		a, b []float64
		want float64
	}{
		{"empty", nil, nil, 0},
		{"same", []float64{1, 2, 3}, []float64{1, 2, 3}, 0},
		{"2d", []float64{0, 0}, []float64{3, 4}, 5},                     // 3-4-5 triangle
		{"3d", []float64{1, 2, 3}, []float64{4, 6, 8}, math.Sqrt(50)},   // sqrt(9+16+25)
		{"negative", []float64{-1, -2}, []float64{1, 2}, math.Sqrt(20)}, // sqrt(4+16)
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := EuclideanDistance(tt.a, tt.b)
			if !almostEqual(got, tt.want, 1e-14) {
				t.Errorf("EuclideanDistance() = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestNormalize validates Normalize operation
func TestNormalize(t *testing.T) {
	tests := []struct {
		name      string
		a         []float64
		wantNorm  float64 // Expected norm of result (should be 1 for non-zero vectors)
		checkZero bool    // If true, check that output equals input (for zero vectors)
	}{
		{"empty", nil, 0, false},
		{"unit", []float64{1, 0, 0}, 1, false},
		{"2d", []float64{3, 4}, 1, false},    // Will become {0.6, 0.8}
		{"3d", []float64{2, 2, 1}, 1, false}, // Norm = 3, normalized = {2/3, 2/3, 1/3}
		{"negative", []float64{-1, 0, 1}, 1, false},
		{"zero", []float64{0, 0, 0}, 0, true},      // Should remain zero
		{"tiny", []float64{1e-15, 1e-15}, 0, true}, // Below threshold, should remain unchanged
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, len(tt.a))
			Normalize(dst, tt.a)

			if tt.checkZero {
				// For zero/tiny vectors, output should equal input
				for i := range dst {
					if i < len(tt.a) && dst[i] != tt.a[i] {
						t.Errorf("Normalize()[%d] = %v, want %v (zero vector)", i, dst[i], tt.a[i])
					}
				}
			} else if len(dst) > 0 {
				// Calculate norm of result
				norm := 0.0
				for _, v := range dst {
					norm += v * v
				}
				norm = math.Sqrt(norm)

				if !almostEqual(norm, tt.wantNorm, 1e-10) {
					t.Errorf("Normalize() norm = %v, want %v", norm, tt.wantNorm)
				}
			}
		})
	}
}

// TestCumulativeSum validates CumulativeSum operation
func TestCumulativeSum(t *testing.T) {
	tests := []struct {
		name string
		a    []float64
		want []float64
	}{
		{"empty", nil, nil},
		{"single", []float64{5}, []float64{5}},
		{"simple", []float64{1, 2, 3, 4, 5}, []float64{1, 3, 6, 10, 15}},
		{"negative", []float64{5, -3, 2, -1}, []float64{5, 2, 4, 3}},
		{"zeros", []float64{0, 1, 0, 2, 0}, []float64{0, 1, 1, 3, 3}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, len(tt.a))
			CumulativeSum(dst, tt.a)
			for i := range dst {
				if i < len(tt.want) && !almostEqual(dst[i], tt.want[i], 1e-14) {
					t.Errorf("CumulativeSum()[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

// TestSqrt_Large tests Sqrt with large arrays
func TestSqrt_Large(t *testing.T) {
	// Test different sizes to exercise SIMD paths
	sizes := []int{7, 8, 9, 15, 16, 17, 31, 32, 33, 100, 256, 277, 1000}

	for _, n := range sizes {
		t.Run(string(rune('0'+n%10)), func(t *testing.T) {
			a := make([]float64, n)
			dst := make([]float64, n)

			// Fill with perfect squares
			for i := range a {
				a[i] = float64((i + 1) * (i + 1))
			}

			Sqrt(dst, a)

			// Verify
			for i := range dst {
				expected := float64(i + 1)
				if !almostEqual(dst[i], expected, 1e-10) {
					t.Errorf("Sqrt[%d] = %v, want %v", i, dst[i], expected)
				}
			}
		})
	}
}

// TestEuclideanDistance_HighDim tests with high-dimensional vectors
func TestEuclideanDistance_HighDim(t *testing.T) {
	// Create two high-dimensional vectors
	n := 1000
	a := make([]float64, n)
	b := make([]float64, n)

	// Unit vectors in different directions
	a[0] = 1.0 // [1, 0, 0, ...]
	b[1] = 1.0 // [0, 1, 0, ...]

	got := EuclideanDistance(a, b)
	want := math.Sqrt(2) // sqrt(1^2 + 1^2)

	if !almostEqual(got, want, 1e-14) {
		t.Errorf("EuclideanDistance(high-dim) = %v, want %v", got, want)
	}

	// Parallel vectors
	for i := range a {
		a[i] = float64(i)
		b[i] = float64(i)
	}

	got = EuclideanDistance(a, b)
	if got != 0 {
		t.Errorf("EuclideanDistance(same vectors) = %v, want 0", got)
	}
}

// Benchmarks for new operations

func BenchmarkSqrt_100(b *testing.B) {
	a := make([]float64, 100)
	dst := make([]float64, 100)
	for i := range a {
		a[i] = float64(i + 1)
	}

	b.SetBytes(100 * 8 * 2)

	for b.Loop() {
		Sqrt(dst, a)
	}
}

func BenchmarkReciprocal_100(b *testing.B) {
	a := make([]float64, 100)
	dst := make([]float64, 100)
	for i := range a {
		a[i] = float64(i + 1)
	}

	b.SetBytes(100 * 8 * 2)

	for b.Loop() {
		Reciprocal(dst, a)
	}
}

func BenchmarkMean_1000(b *testing.B) {
	a := make([]float64, 1000)
	for i := range a {
		a[i] = float64(i)
	}

	b.SetBytes(1000 * 8)

	var result float64
	for b.Loop() {
		result = Mean(a)
	}
	_ = result
}

func BenchmarkVariance_1000(b *testing.B) {
	a := make([]float64, 1000)
	for i := range a {
		a[i] = float64(i)
	}

	b.SetBytes(1000 * 8)

	var result float64
	for b.Loop() {
		result = Variance(a)
	}
	_ = result
}

func BenchmarkEuclideanDistance_100(b *testing.B) {
	a := make([]float64, 100)
	c := make([]float64, 100)
	for i := range a {
		a[i] = float64(i)
		c[i] = float64(i + 1)
	}

	b.SetBytes(100 * 8 * 2)

	var result float64
	for b.Loop() {
		result = EuclideanDistance(a, c)
	}
	_ = result
}

func BenchmarkNormalize_100(b *testing.B) {
	a := make([]float64, 100)
	dst := make([]float64, 100)
	for i := range a {
		a[i] = float64(i + 1)
	}

	b.SetBytes(100 * 8 * 2)

	for b.Loop() {
		Normalize(dst, a)
	}
}

func BenchmarkCumulativeSum_1000(b *testing.B) {
	a := make([]float64, 1000)
	dst := make([]float64, 1000)
	for i := range a {
		a[i] = float64(i)
	}

	b.SetBytes(1000 * 8 * 2)

	for b.Loop() {
		CumulativeSum(dst, a)
	}
}
