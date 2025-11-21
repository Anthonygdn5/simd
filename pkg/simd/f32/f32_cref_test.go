package f32

// Tests generated from C reference implementation output.
// See testdata/generate_expectations.c for the reference implementation.

import (
	"math"
	"testing"
)

const tolerance32 = 1e-5

func almostEqual32(a, b float32) bool {
	if math.IsNaN(float64(a)) && math.IsNaN(float64(b)) {
		return true
	}
	if math.IsInf(float64(a), 1) && math.IsInf(float64(b), 1) {
		return true
	}
	if math.IsInf(float64(a), -1) && math.IsInf(float64(b), -1) {
		return true
	}
	diff := a - b
	if diff < 0 {
		diff = -diff
	}
	return diff <= tolerance32
}

func slicesAlmostEqual32(got, want []float32) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if !almostEqual32(got[i], want[i]) {
			return false
		}
	}
	return true
}

// TestDotProduct_CRef validates DotProduct against C reference output
func TestDotProduct_CRef(t *testing.T) {
	tests := []struct {
		name string
		a, b []float32
		want float32
	}{
		// SIMD boundary tests
		{"size_1", []float32{1}, []float32{1}, 1},
		{"size_3", []float32{1, 2, 3}, []float32{3, 2, 1}, 10},
		{"size_4", []float32{1, 2, 3, 4}, []float32{4, 3, 2, 1}, 20},
		{"size_5", []float32{1, 2, 3, 4, 5}, []float32{5, 4, 3, 2, 1}, 35},
		{"size_7", []float32{1, 2, 3, 4, 5, 6, 7}, []float32{7, 6, 5, 4, 3, 2, 1}, 84},
		{"size_8", []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{8, 7, 6, 5, 4, 3, 2, 1}, 120},
		{"size_9", []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}, []float32{9, 8, 7, 6, 5, 4, 3, 2, 1}, 165},
		{"size_15", makeSeq32(15), makeRevSeq32(15), 680},
		{"size_16", makeSeq32(16), makeRevSeq32(16), 816},
		{"size_17", makeSeq32(17), makeRevSeq32(17), 969},
		{"size_31", makeSeq32(31), makeRevSeq32(31), 5456},
		{"size_32", makeSeq32(32), makeRevSeq32(32), 5984},
		{"size_33", makeSeq32(33), makeRevSeq32(33), 6545},
		// Mixed signs
		{"mixed_signs", []float32{-5, -4, -3, -2, -1, 1, 2, 3, 4, 5}, []float32{5, 4, 3, 2, 1, -1, -2, -3, -4, -5}, -110},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := DotProduct(tt.a, tt.b)
			if !almostEqual32(got, tt.want) {
				t.Errorf("DotProduct() = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestSum_CRef validates Sum against C reference output
func TestSum_CRef(t *testing.T) {
	tests := []struct {
		name string
		a    []float32
		want float32
	}{
		{"size_1", []float32{1}, 1},
		{"size_3", []float32{1, 2, 3}, 6},
		{"size_4", []float32{1, 2, 3, 4}, 10},
		{"size_5", []float32{1, 2, 3, 4, 5}, 15},
		{"size_7", []float32{1, 2, 3, 4, 5, 6, 7}, 28},
		{"size_8", []float32{1, 2, 3, 4, 5, 6, 7, 8}, 36},
		{"size_9", []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}, 45},
		{"size_15", makeSeq32(15), 120},
		{"size_16", makeSeq32(16), 136},
		{"size_17", makeSeq32(17), 153},
		{"size_31", makeSeq32(31), 496},
		{"size_32", makeSeq32(32), 528},
		{"size_33", makeSeq32(33), 561},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Sum(tt.a)
			if !almostEqual32(got, tt.want) {
				t.Errorf("Sum() = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestMinMax_CRef validates Min and Max against C reference output
func TestMinMax_CRef(t *testing.T) {
	tests := []struct {
		name    string
		a       []float32
		wantMin float32
		wantMax float32
	}{
		{"size_8", []float32{1, 2, 3, 4, 5, 6, 7, 8}, 1, 8},
		{"size_9", []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}, 1, 9},
		{"size_16", makeSeq32(16), 1, 16},
		{"size_17", makeSeq32(17), 1, 17},
		// Special values test from C reference
		{"special", []float32{0, 0, 1.17549435e-38, 1.17549435e-38, 1.70141173e+38, 1.70141173e+38, 1, 1}, 0, 1.70141173e+38},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotMin := Min(tt.a)
			if !almostEqual32(gotMin, tt.wantMin) {
				t.Errorf("Min() = %v, want %v", gotMin, tt.wantMin)
			}
			gotMax := Max(tt.a)
			if !almostEqual32(gotMax, tt.wantMax) {
				t.Errorf("Max() = %v, want %v", gotMax, tt.wantMax)
			}
		})
	}
}

// TestAdd_CRef validates Add against C reference output
func TestAdd_CRef(t *testing.T) {
	tests := []struct {
		name string
		a, b []float32
		want []float32
	}{
		{"size_4", []float32{1, 2, 3, 4}, []float32{4, 3, 2, 1}, []float32{5, 5, 5, 5}},
		{"size_5", []float32{1, 2, 3, 4, 5}, []float32{5, 4, 3, 2, 1}, []float32{6, 6, 6, 6, 6}},
		{"size_8", []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{8, 7, 6, 5, 4, 3, 2, 1}, []float32{9, 9, 9, 9, 9, 9, 9, 9}},
		{"size_9", []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}, []float32{9, 8, 7, 6, 5, 4, 3, 2, 1}, []float32{10, 10, 10, 10, 10, 10, 10, 10, 10}},
		{"mixed_signs", []float32{-5, -4, -3, -2, -1, 1, 2, 3, 4, 5}, []float32{5, 4, 3, 2, 1, -1, -2, -3, -4, -5}, []float32{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.a))
			Add(dst, tt.a, tt.b)
			if !slicesAlmostEqual32(dst, tt.want) {
				t.Errorf("Add() = %v, want %v", dst, tt.want)
			}
		})
	}
}

// TestSub_CRef validates Sub against C reference output
func TestSub_CRef(t *testing.T) {
	tests := []struct {
		name string
		a, b []float32
		want []float32
	}{
		{"size_4", []float32{1, 2, 3, 4}, []float32{4, 3, 2, 1}, []float32{-3, -1, 1, 3}},
		{"size_5", []float32{1, 2, 3, 4, 5}, []float32{5, 4, 3, 2, 1}, []float32{-4, -2, 0, 2, 4}},
		{"size_8", []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{8, 7, 6, 5, 4, 3, 2, 1}, []float32{-7, -5, -3, -1, 1, 3, 5, 7}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.a))
			Sub(dst, tt.a, tt.b)
			if !slicesAlmostEqual32(dst, tt.want) {
				t.Errorf("Sub() = %v, want %v", dst, tt.want)
			}
		})
	}
}

// TestMul_CRef validates Mul against C reference output
func TestMul_CRef(t *testing.T) {
	tests := []struct {
		name string
		a, b []float32
		want []float32
	}{
		{"size_4", []float32{1, 2, 3, 4}, []float32{4, 3, 2, 1}, []float32{4, 6, 6, 4}},
		{"size_5", []float32{1, 2, 3, 4, 5}, []float32{5, 4, 3, 2, 1}, []float32{5, 8, 9, 8, 5}},
		{"size_8", []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{8, 7, 6, 5, 4, 3, 2, 1}, []float32{8, 14, 18, 20, 20, 18, 14, 8}},
		{"size_9", []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}, []float32{9, 8, 7, 6, 5, 4, 3, 2, 1}, []float32{9, 16, 21, 24, 25, 24, 21, 16, 9}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.a))
			Mul(dst, tt.a, tt.b)
			if !slicesAlmostEqual32(dst, tt.want) {
				t.Errorf("Mul() = %v, want %v", dst, tt.want)
			}
		})
	}
}

// TestDiv_CRef validates Div against C reference output
func TestDiv_CRef(t *testing.T) {
	tests := []struct {
		name string
		a, b []float32
		want []float32
	}{
		{
			"edge_cases",
			[]float32{10, -10, 1, -1, 100, 0.01, 1e+10, 1e-10},
			[]float32{2, -2, 3, -3, 0.1, 100, 1e-5, 1e5},
			// C reference: divResult32 := []float32{5, 5, 0.333333343, 0.333333343, 1000, 9.99999975e-05, 1.00000005e+15, 1e-15}
			[]float32{5, 5, 0.333333343, 0.333333343, 1000, 0.0001, 1.00000005e+15, 1e-15},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.a))
			Div(dst, tt.a, tt.b)
			if !slicesAlmostEqual32(dst, tt.want) {
				t.Errorf("Div() = %v, want %v", dst, tt.want)
			}
		})
	}
}

// TestScale_CRef validates Scale against C reference output
func TestScale_CRef(t *testing.T) {
	tests := []struct {
		name   string
		a      []float32
		scalar float32
		want   []float32
	}{
		{"size_4", []float32{1, 2, 3, 4}, 2.5, []float32{2.5, 5, 7.5, 10}},
		{"size_5", []float32{1, 2, 3, 4, 5}, 2.5, []float32{2.5, 5, 7.5, 10, 12.5}},
		{"size_8", []float32{1, 2, 3, 4, 5, 6, 7, 8}, 2.5, []float32{2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.a))
			Scale(dst, tt.a, tt.scalar)
			if !slicesAlmostEqual32(dst, tt.want) {
				t.Errorf("Scale() = %v, want %v", dst, tt.want)
			}
		})
	}
}

// TestAddScalar_CRef validates AddScalar against C reference output
func TestAddScalar_CRef(t *testing.T) {
	tests := []struct {
		name   string
		a      []float32
		scalar float32
		want   []float32
	}{
		{"positive", []float32{1, 2, 3, 4, 5, 6, 7, 8}, 10.5, []float32{11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5}},
		{"negative", []float32{1, 2, 3, 4, 5, 6, 7, 8}, -3.0, []float32{-2, -1, 0, 1, 2, 3, 4, 5}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.a))
			AddScalar(dst, tt.a, tt.scalar)
			if !slicesAlmostEqual32(dst, tt.want) {
				t.Errorf("AddScalar() = %v, want %v", dst, tt.want)
			}
		})
	}
}

// TestAbs_CRef validates Abs against C reference output
func TestAbs_CRef(t *testing.T) {
	tests := []struct {
		name string
		a    []float32
		want []float32
	}{
		{"size_4", []float32{1, 2, 3, 4}, []float32{1, 2, 3, 4}},
		{"size_8", []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{1, 2, 3, 4, 5, 6, 7, 8}},
		{"mixed_signs", []float32{-5, -4, -3, -2, -1, 1, 2, 3, 4, 5}, []float32{5, 4, 3, 2, 1, 1, 2, 3, 4, 5}},
		{"special", []float32{0, 0, 1.17549435e-38, 1.17549435e-38, 1.70141173e+38, 1.70141173e+38, 1, 1}, []float32{0, 0, 1.17549435e-38, 1.17549435e-38, 1.70141173e+38, 1.70141173e+38, 1, 1}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.a))
			Abs(dst, tt.a)
			if !slicesAlmostEqual32(dst, tt.want) {
				t.Errorf("Abs() = %v, want %v", dst, tt.want)
			}
		})
	}
}

// TestNeg_CRef validates Neg against C reference output
func TestNeg_CRef(t *testing.T) {
	tests := []struct {
		name string
		a    []float32
		want []float32
	}{
		{"size_4", []float32{1, 2, 3, 4}, []float32{-1, -2, -3, -4}},
		{"size_8", []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{-1, -2, -3, -4, -5, -6, -7, -8}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.a))
			Neg(dst, tt.a)
			if !slicesAlmostEqual32(dst, tt.want) {
				t.Errorf("Neg() = %v, want %v", dst, tt.want)
			}
		})
	}
}

// TestFMA_CRef validates FMA against C reference output
func TestFMA_CRef(t *testing.T) {
	tests := []struct {
		name    string
		a, b, c []float32
		want    []float32
	}{
		{
			"size_4",
			[]float32{1, 2, 3, 4}, []float32{4, 3, 2, 1}, []float32{0.5, 0.5, 0.5, 0.5},
			[]float32{4.5, 6.5, 6.5, 4.5},
		},
		{
			"size_5",
			[]float32{1, 2, 3, 4, 5}, []float32{5, 4, 3, 2, 1}, []float32{0.5, 0.5, 0.5, 0.5, 0.5},
			[]float32{5.5, 8.5, 9.5, 8.5, 5.5},
		},
		{
			"size_8",
			[]float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{8, 7, 6, 5, 4, 3, 2, 1}, []float32{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5},
			[]float32{8.5, 14.5, 18.5, 20.5, 20.5, 18.5, 14.5, 8.5},
		},
		{
			"mixed_signs",
			[]float32{-5, -4, -3, -2, -1, 1, 2, 3, 4, 5},
			[]float32{5, 4, 3, 2, 1, -1, -2, -3, -4, -5},
			[]float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0},
			[]float32{-24.9, -15.8, -8.7, -3.6, -0.5, -0.4, -3.3, -8.2, -15.1, -24},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.a))
			FMA(dst, tt.a, tt.b, tt.c)
			if !slicesAlmostEqual32(dst, tt.want) {
				t.Errorf("FMA() = %v, want %v", dst, tt.want)
			}
		})
	}
}

// TestClamp_CRef validates Clamp against C reference output
func TestClamp_CRef(t *testing.T) {
	tests := []struct {
		name     string
		a        []float32
		min, max float32
		want     []float32
	}{
		{
			"0_to_10",
			[]float32{-100, -10, -1, -0.5, 0, 0.5, 1, 5, 10, 50, 100, 1000},
			0, 10,
			[]float32{0, 0, 0, 0, 0, 0.5, 1, 5, 10, 10, 10, 10},
		},
		{
			"simd_boundary",
			[]float32{1, 2, 3, 4, 5, 6, 7, 8},
			2, 5,
			[]float32{2, 2, 3, 4, 5, 5, 5, 5},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.a))
			Clamp(dst, tt.a, tt.min, tt.max)
			if !slicesAlmostEqual32(dst, tt.want) {
				t.Errorf("Clamp() = %v, want %v", dst, tt.want)
			}
		})
	}
}

// TestDotProduct_LargeSineCosine validates dot product with sine/cosine pattern
func TestDotProduct_LargeSineCosine(t *testing.T) {
	tests := []struct {
		n   int
		tol float32
	}{
		// For sine*cosine, the result should be near zero
		// float32 has less precision, so tolerance is higher
		{256, 1e-4},
		{277, 1e-4},
		{512, 1e-4},
		{1024, 1e-4},
	}

	for _, tt := range tests {
		t.Run(string(rune('0'+tt.n%10)), func(t *testing.T) {
			a := make([]float32, tt.n)
			b := make([]float32, tt.n)
			for i := 0; i < tt.n; i++ {
				a[i] = float32(math.Sin(2.0 * math.Pi * float64(i) / float64(tt.n)))
				b[i] = float32(math.Cos(2.0 * math.Pi * float64(i) / float64(tt.n)))
			}

			got := DotProduct(a, b)
			// Result should be near zero for orthogonal sin/cos
			if got > tt.tol || got < -tt.tol {
				t.Errorf("DotProduct(sin, cos, n=%d) = %v, want ~0 (within %v)", tt.n, got, tt.tol)
			}
		})
	}
}

// Helper functions for generating test data
func makeSeq32(n int) []float32 {
	a := make([]float32, n)
	for i := range a {
		a[i] = float32(i + 1)
	}
	return a
}

func makeRevSeq32(n int) []float32 {
	a := make([]float32, n)
	for i := range a {
		a[i] = float32(n - i)
	}
	return a
}
