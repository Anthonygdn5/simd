//go:build !amd64 && !arm64

package cpu

func cpuInfo() string {
	return "Generic (no SIMD)"
}
