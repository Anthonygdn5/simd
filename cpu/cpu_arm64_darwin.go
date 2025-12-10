//go:build arm64 && darwin

package cpu

// Apple Silicon (M1/M2/M3/M4) all support FEAT_FP16 (half-precision floating point).
// The golang.org/x/sys/cpu package doesn't properly detect this on macOS,
// so we enable it unconditionally on darwin/arm64.

func init() {
	// All Apple Silicon chips support FP16
	ARM64.FP16 = true
}
