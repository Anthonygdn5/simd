package f64

// Tests generated from C reference implementation output.
// See testdata/generate_expectations.c for the reference implementation.

import (
	"math"
	"testing"
)

const tolerance = 1e-14

func almostEqual(a, b, tol float64) bool {
	if math.IsNaN(a) && math.IsNaN(b) {
		return true
	}
	if math.IsInf(a, 1) && math.IsInf(b, 1) {
		return true
	}
	if math.IsInf(a, -1) && math.IsInf(b, -1) {
		return true
	}
	return math.Abs(a-b) <= tol
}

func slicesAlmostEqual(got, want []float64) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if !almostEqual(got[i], want[i], tolerance) {
			return false
		}
	}
	return true
}

// TestDotProduct_CRef validates DotProduct against C reference output
func TestDotProduct_CRef(t *testing.T) {
	tests := []struct {
		name string
		a, b []float64
		want float64
	}{
		// SIMD boundary tests
		{"size_1", []float64{1}, []float64{1}, 1},
		{"size_3", []float64{1, 2, 3}, []float64{3, 2, 1}, 10},
		{"size_4", []float64{1, 2, 3, 4}, []float64{4, 3, 2, 1}, 20},
		{"size_5", []float64{1, 2, 3, 4, 5}, []float64{5, 4, 3, 2, 1}, 35},
		{"size_7", []float64{1, 2, 3, 4, 5, 6, 7}, []float64{7, 6, 5, 4, 3, 2, 1}, 84},
		{"size_8", []float64{1, 2, 3, 4, 5, 6, 7, 8}, []float64{8, 7, 6, 5, 4, 3, 2, 1}, 120},
		{"size_9", []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}, []float64{9, 8, 7, 6, 5, 4, 3, 2, 1}, 165},
		{"size_15", makeSeq(15), makeRevSeq(15), 680},
		{"size_16", makeSeq(16), makeRevSeq(16), 816},
		{"size_17", makeSeq(17), makeRevSeq(17), 969},
		{"size_31", makeSeq(31), makeRevSeq(31), 5456},
		{"size_32", makeSeq(32), makeRevSeq(32), 5984},
		{"size_33", makeSeq(33), makeRevSeq(33), 6545},
		// Mixed signs
		{"mixed_signs", []float64{-5, -4, -3, -2, -1, 1, 2, 3, 4, 5}, []float64{5, 4, 3, 2, 1, -1, -2, -3, -4, -5}, -110},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := DotProduct(tt.a, tt.b)
			if !almostEqual(got, tt.want, tolerance) {
				t.Errorf("DotProduct() = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestSum_CRef validates Sum against C reference output
func TestSum_CRef(t *testing.T) {
	tests := []struct {
		name string
		a    []float64
		want float64
	}{
		{"size_1", []float64{1}, 1},
		{"size_3", []float64{1, 2, 3}, 6},
		{"size_4", []float64{1, 2, 3, 4}, 10},
		{"size_5", []float64{1, 2, 3, 4, 5}, 15},
		{"size_7", []float64{1, 2, 3, 4, 5, 6, 7}, 28},
		{"size_8", []float64{1, 2, 3, 4, 5, 6, 7, 8}, 36},
		{"size_9", []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}, 45},
		{"size_15", makeSeq(15), 120},
		{"size_16", makeSeq(16), 136},
		{"size_17", makeSeq(17), 153},
		{"size_31", makeSeq(31), 496},
		{"size_32", makeSeq(32), 528},
		{"size_33", makeSeq(33), 561},
		{"mixed_signs", []float64{-5, -4, -3, -2, -1, 1, 2, 3, 4, 5}, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Sum(tt.a)
			if !almostEqual(got, tt.want, tolerance) {
				t.Errorf("Sum() = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestMinMax_CRef validates Min and Max against C reference output
func TestMinMax_CRef(t *testing.T) {
	tests := []struct {
		name    string
		a       []float64
		wantMin float64
		wantMax float64
	}{
		{"size_8", []float64{1, 2, 3, 4, 5, 6, 7, 8}, 1, 8},
		{"size_9", []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}, 1, 9},
		{"size_16", makeSeq(16), 1, 16},
		{"size_17", makeSeq(17), 1, 17},
		{"mixed_signs", []float64{-5, -4, -3, -2, -1, 1, 2, 3, 4, 5}, -5, 5},
		// Special values test
		{"special", []float64{0, 0, 2.2250738585072014e-308, 2.2250738585072014e-308, 8.9884656743115785e+307, 8.9884656743115785e+307, 1, 1}, 0, 8.9884656743115785e+307},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotMin := Min(tt.a)
			if !almostEqual(gotMin, tt.wantMin, tolerance) {
				t.Errorf("Min() = %v, want %v", gotMin, tt.wantMin)
			}
			gotMax := Max(tt.a)
			if !almostEqual(gotMax, tt.wantMax, tolerance) {
				t.Errorf("Max() = %v, want %v", gotMax, tt.wantMax)
			}
		})
	}
}

// TestAdd_CRef validates Add against C reference output
func TestAdd_CRef(t *testing.T) {
	tests := []struct {
		name string
		a, b []float64
		want []float64
	}{
		{"size_4", []float64{1, 2, 3, 4}, []float64{4, 3, 2, 1}, []float64{5, 5, 5, 5}},
		{"size_5", []float64{1, 2, 3, 4, 5}, []float64{5, 4, 3, 2, 1}, []float64{6, 6, 6, 6, 6}},
		{"size_8", []float64{1, 2, 3, 4, 5, 6, 7, 8}, []float64{8, 7, 6, 5, 4, 3, 2, 1}, []float64{9, 9, 9, 9, 9, 9, 9, 9}},
		{"size_9", []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}, []float64{9, 8, 7, 6, 5, 4, 3, 2, 1}, []float64{10, 10, 10, 10, 10, 10, 10, 10, 10}},
		{"mixed_signs", []float64{-5, -4, -3, -2, -1, 1, 2, 3, 4, 5}, []float64{5, 4, 3, 2, 1, -1, -2, -3, -4, -5}, []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, len(tt.a))
			Add(dst, tt.a, tt.b)
			if !slicesAlmostEqual(dst, tt.want) {
				t.Errorf("Add() = %v, want %v", dst, tt.want)
			}
		})
	}
}

// TestSub_CRef validates Sub against C reference output
func TestSub_CRef(t *testing.T) {
	tests := []struct {
		name string
		a, b []float64
		want []float64
	}{
		{"size_4", []float64{1, 2, 3, 4}, []float64{4, 3, 2, 1}, []float64{-3, -1, 1, 3}},
		{"size_5", []float64{1, 2, 3, 4, 5}, []float64{5, 4, 3, 2, 1}, []float64{-4, -2, 0, 2, 4}},
		{"size_8", []float64{1, 2, 3, 4, 5, 6, 7, 8}, []float64{8, 7, 6, 5, 4, 3, 2, 1}, []float64{-7, -5, -3, -1, 1, 3, 5, 7}},
		{"mixed_signs", []float64{-5, -4, -3, -2, -1, 1, 2, 3, 4, 5}, []float64{5, 4, 3, 2, 1, -1, -2, -3, -4, -5}, []float64{-10, -8, -6, -4, -2, 2, 4, 6, 8, 10}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, len(tt.a))
			Sub(dst, tt.a, tt.b)
			if !slicesAlmostEqual(dst, tt.want) {
				t.Errorf("Sub() = %v, want %v", dst, tt.want)
			}
		})
	}
}

// TestMul_CRef validates Mul against C reference output
func TestMul_CRef(t *testing.T) {
	tests := []struct {
		name string
		a, b []float64
		want []float64
	}{
		{"size_4", []float64{1, 2, 3, 4}, []float64{4, 3, 2, 1}, []float64{4, 6, 6, 4}},
		{"size_5", []float64{1, 2, 3, 4, 5}, []float64{5, 4, 3, 2, 1}, []float64{5, 8, 9, 8, 5}},
		{"size_8", []float64{1, 2, 3, 4, 5, 6, 7, 8}, []float64{8, 7, 6, 5, 4, 3, 2, 1}, []float64{8, 14, 18, 20, 20, 18, 14, 8}},
		{"size_9", []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}, []float64{9, 8, 7, 6, 5, 4, 3, 2, 1}, []float64{9, 16, 21, 24, 25, 24, 21, 16, 9}},
		{"mixed_signs", []float64{-5, -4, -3, -2, -1, 1, 2, 3, 4, 5}, []float64{5, 4, 3, 2, 1, -1, -2, -3, -4, -5}, []float64{-25, -16, -9, -4, -1, -1, -4, -9, -16, -25}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, len(tt.a))
			Mul(dst, tt.a, tt.b)
			if !slicesAlmostEqual(dst, tt.want) {
				t.Errorf("Mul() = %v, want %v", dst, tt.want)
			}
		})
	}
}

// TestDiv_CRef validates Div against C reference output
func TestDiv_CRef(t *testing.T) {
	tests := []struct {
		name string
		a, b []float64
		want []float64
	}{
		{
			"edge_cases",
			[]float64{10, -10, 1, -1, 100, 0.01, 10000000000, 1e-10},
			[]float64{2, -2, 3, -3, 0.10000000000000001, 100, 1.0000000000000001e-05, 100000},
			[]float64{5, 5, 0.33333333333333331, 0.33333333333333331, 1000, 0.0001, 999999999999999.88, 1.0000000000000001e-15},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, len(tt.a))
			Div(dst, tt.a, tt.b)
			if !slicesAlmostEqual(dst, tt.want) {
				t.Errorf("Div() = %v, want %v", dst, tt.want)
			}
		})
	}
}

// TestScale_CRef validates Scale against C reference output
func TestScale_CRef(t *testing.T) {
	tests := []struct {
		name   string
		a      []float64
		scalar float64
		want   []float64
	}{
		{"size_4", []float64{1, 2, 3, 4}, 2.5, []float64{2.5, 5, 7.5, 10}},
		{"size_5", []float64{1, 2, 3, 4, 5}, 2.5, []float64{2.5, 5, 7.5, 10, 12.5}},
		{"size_8", []float64{1, 2, 3, 4, 5, 6, 7, 8}, 2.5, []float64{2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, len(tt.a))
			Scale(dst, tt.a, tt.scalar)
			if !slicesAlmostEqual(dst, tt.want) {
				t.Errorf("Scale() = %v, want %v", dst, tt.want)
			}
		})
	}
}

// TestAddScalar_CRef validates AddScalar against C reference output
func TestAddScalar_CRef(t *testing.T) {
	tests := []struct {
		name   string
		a      []float64
		scalar float64
		want   []float64
	}{
		{"positive", []float64{1, 2, 3, 4, 5, 6, 7, 8}, 10.5, []float64{11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5}},
		{"negative", []float64{1, 2, 3, 4, 5, 6, 7, 8}, -3.0, []float64{-2, -1, 0, 1, 2, 3, 4, 5}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, len(tt.a))
			AddScalar(dst, tt.a, tt.scalar)
			if !slicesAlmostEqual(dst, tt.want) {
				t.Errorf("AddScalar() = %v, want %v", dst, tt.want)
			}
		})
	}
}

// TestAbs_CRef validates Abs against C reference output
func TestAbs_CRef(t *testing.T) {
	tests := []struct {
		name string
		a    []float64
		want []float64
	}{
		{"size_4", []float64{1, 2, 3, 4}, []float64{1, 2, 3, 4}},
		{"size_8", []float64{1, 2, 3, 4, 5, 6, 7, 8}, []float64{1, 2, 3, 4, 5, 6, 7, 8}},
		{"mixed_signs", []float64{-5, -4, -3, -2, -1, 1, 2, 3, 4, 5}, []float64{5, 4, 3, 2, 1, 1, 2, 3, 4, 5}},
		{"special", []float64{0, 0, 2.2250738585072014e-308, 2.2250738585072014e-308, 8.9884656743115785e+307, 8.9884656743115785e+307, 1, 1}, []float64{0, 0, 2.2250738585072014e-308, 2.2250738585072014e-308, 8.9884656743115785e+307, 8.9884656743115785e+307, 1, 1}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, len(tt.a))
			Abs(dst, tt.a)
			if !slicesAlmostEqual(dst, tt.want) {
				t.Errorf("Abs() = %v, want %v", dst, tt.want)
			}
		})
	}
}

// TestNeg_CRef validates Neg against C reference output
func TestNeg_CRef(t *testing.T) {
	tests := []struct {
		name string
		a    []float64
		want []float64
	}{
		{"size_4", []float64{1, 2, 3, 4}, []float64{-1, -2, -3, -4}},
		{"size_8", []float64{1, 2, 3, 4, 5, 6, 7, 8}, []float64{-1, -2, -3, -4, -5, -6, -7, -8}},
		{"mixed_signs", []float64{-5, -4, -3, -2, -1, 1, 2, 3, 4, 5}, []float64{5, 4, 3, 2, 1, -1, -2, -3, -4, -5}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, len(tt.a))
			Neg(dst, tt.a)
			if !slicesAlmostEqual(dst, tt.want) {
				t.Errorf("Neg() = %v, want %v", dst, tt.want)
			}
		})
	}
}

// TestFMA_CRef validates FMA against C reference output
func TestFMA_CRef(t *testing.T) {
	tests := []struct {
		name    string
		a, b, c []float64
		want    []float64
	}{
		{
			"size_4",
			[]float64{1, 2, 3, 4}, []float64{4, 3, 2, 1}, []float64{0.5, 0.5, 0.5, 0.5},
			[]float64{4.5, 6.5, 6.5, 4.5},
		},
		{
			"size_5",
			[]float64{1, 2, 3, 4, 5}, []float64{5, 4, 3, 2, 1}, []float64{0.5, 0.5, 0.5, 0.5, 0.5},
			[]float64{5.5, 8.5, 9.5, 8.5, 5.5},
		},
		{
			"size_8",
			[]float64{1, 2, 3, 4, 5, 6, 7, 8}, []float64{8, 7, 6, 5, 4, 3, 2, 1}, []float64{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5},
			[]float64{8.5, 14.5, 18.5, 20.5, 20.5, 18.5, 14.5, 8.5},
		},
		{
			"mixed_signs",
			[]float64{-5, -4, -3, -2, -1, 1, 2, 3, 4, 5},
			[]float64{5, 4, 3, 2, 1, -1, -2, -3, -4, -5},
			[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0},
			[]float64{-24.9, -15.8, -8.7, -3.6, -0.5, -0.4, -3.3, -8.2, -15.1, -24},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, len(tt.a))
			FMA(dst, tt.a, tt.b, tt.c)
			if !slicesAlmostEqual(dst, tt.want) {
				t.Errorf("FMA() = %v, want %v", dst, tt.want)
			}
		})
	}
}

// TestClamp_CRef validates Clamp against C reference output
func TestClamp_CRef(t *testing.T) {
	tests := []struct {
		name     string
		a        []float64
		min, max float64
		want     []float64
	}{
		{
			"0_to_10",
			[]float64{-100, -10, -1, -0.5, 0, 0.5, 1, 5, 10, 50, 100, 1000},
			0, 10,
			[]float64{0, 0, 0, 0, 0, 0.5, 1, 5, 10, 10, 10, 10},
		},
		{
			"neg5_to_5",
			[]float64{-100, -10, -1, -0.5, 0, 0.5, 1, 5, 10, 50, 100, 1000},
			-5, 5,
			[]float64{-5, -5, -1, -0.5, 0, 0.5, 1, 5, 5, 5, 5, 5},
		},
		{
			"1_to_100",
			[]float64{-100, -10, -1, -0.5, 0, 0.5, 1, 5, 10, 50, 100, 1000},
			1, 100,
			[]float64{1, 1, 1, 1, 1, 1, 1, 5, 10, 50, 100, 100},
		},
		{
			"simd_boundary",
			[]float64{1, 2, 3, 4, 5, 6, 7, 8},
			2, 5,
			[]float64{2, 2, 3, 4, 5, 5, 5, 5},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, len(tt.a))
			Clamp(dst, tt.a, tt.min, tt.max)
			if !slicesAlmostEqual(dst, tt.want) {
				t.Errorf("Clamp() = %v, want %v", dst, tt.want)
			}
		})
	}
}

// TestDotProduct_LargeSineCosine validates dot product with sine/cosine pattern
func TestDotProduct_LargeSineCosine(t *testing.T) {
	tests := []struct {
		n    int
		want float64
		tol  float64
	}{
		// C reference: dotProduct_256 := 5.4535340475840655e-15
		{256, 0, 1e-12},
		// C reference: dotProduct_277 := -3.7749937761775022e-15
		{277, 0, 1e-12},
		// C reference: dotProduct_512 := 1.4310986009095443e-14
		{512, 0, 1e-11},
		// C reference: dotProduct_1024 := 3.3491389345627806e-14
		{1024, 0, 1e-10},
	}

	for _, tt := range tests {
		t.Run(string(rune('0'+tt.n%10)), func(t *testing.T) {
			a := make([]float64, tt.n)
			b := make([]float64, tt.n)
			for i := 0; i < tt.n; i++ {
				a[i] = math.Sin(2.0 * math.Pi * float64(i) / float64(tt.n))
				b[i] = math.Cos(2.0 * math.Pi * float64(i) / float64(tt.n))
			}

			got := DotProduct(a, b)
			// For sine*cosine, the result should be near zero
			if math.Abs(got-tt.want) > tt.tol {
				t.Errorf("DotProduct(sin, cos, n=%d) = %v, want ~%v (within %v)", tt.n, got, tt.want, tt.tol)
			}
		})
	}
}

// Helper functions for generating test data
func makeSeq(n int) []float64 {
	a := make([]float64, n)
	for i := range a {
		a[i] = float64(i + 1)
	}
	return a
}

func makeRevSeq(n int) []float64 {
	a := make([]float64, n)
	for i := range a {
		a[i] = float64(n - i)
	}
	return a
}
