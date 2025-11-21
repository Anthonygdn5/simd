//go:build arm64

package cpu

import "golang.org/x/sys/cpu"

func init() {
	ARM64.NEON = cpu.ARM64.HasASIMD
	ARM64.SVE = cpu.ARM64.HasSVE
	ARM64.SVE2 = cpu.ARM64.HasSVE2
}

func cpuInfo() string {
	switch {
	case ARM64.SVE2:
		return "ARM64 SVE2"
	case ARM64.SVE:
		return "ARM64 SVE"
	case ARM64.NEON:
		return "ARM64 NEON"
	default:
		return "ARM64 (no SIMD)"
	}
}
