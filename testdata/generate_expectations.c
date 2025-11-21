/*
 * Reference C implementation for generating SIMD test expectations.
 * Compile: gcc -O2 -march=native -o generate_expectations generate_expectations.c -lm
 * Run: ./generate_expectations > expectations.txt
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* ============================================================================
 * float64 operations
 * ============================================================================ */

double f64_sqrt(double *dst, const double *a, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = sqrt(a[i]);
    }
}

void f64_reciprocal(double *dst, const double *a, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = 1.0 / a[i];
    }
}

double f64_mean(const double *a, size_t n) {
    if (n == 0) return 0.0;
    return f64_sum(a, n) / (double)n;
}

double f64_variance(const double *a, size_t n) {
    if (n == 0) return 0.0;
    double mean = f64_mean(a, n);
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = a[i] - mean;
        sum += diff * diff;
    }
    return sum / (double)n;
}

double f64_stddev(const double *a, size_t n) {
    return sqrt(f64_variance(a, n));
}

double f64_euclidean_distance(const double *a, const double *b, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

void f64_normalize(double *dst, const double *a, size_t n) {
    double magnitude = 0.0;
    for (size_t i = 0; i < n; i++) {
        magnitude += a[i] * a[i];
    }
    magnitude = sqrt(magnitude);

    if (magnitude < 1e-10) {
        // Copy unchanged for zero/tiny vectors
        for (size_t i = 0; i < n; i++) {
            dst[i] = a[i];
        }
    } else {
        // Scale by 1/magnitude
        double inv_mag = 1.0 / magnitude;
        for (size_t i = 0; i < n; i++) {
            dst[i] = a[i] * inv_mag;
        }
    }
}

void f64_cumulative_sum(double *dst, const double *a, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += a[i];
        dst[i] = sum;
    }
}

double f64_dot_product(const double *a, const double *b, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

void f64_add(double *dst, const double *a, const double *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = a[i] + b[i];
    }
}

void f64_sub(double *dst, const double *a, const double *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = a[i] - b[i];
    }
}

void f64_mul(double *dst, const double *a, const double *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = a[i] * b[i];
    }
}

void f64_div(double *dst, const double *a, const double *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = a[i] / b[i];
    }
}

void f64_scale(double *dst, const double *a, double s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = a[i] * s;
    }
}

void f64_add_scalar(double *dst, const double *a, double s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = a[i] + s;
    }
}

double f64_sum(const double *a, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += a[i];
    }
    return sum;
}

double f64_min(const double *a, size_t n) {
    if (n == 0) return 0.0;
    double min = a[0];
    for (size_t i = 1; i < n; i++) {
        if (a[i] < min) min = a[i];
    }
    return min;
}

double f64_max(const double *a, size_t n) {
    if (n == 0) return 0.0;
    double max = a[0];
    for (size_t i = 1; i < n; i++) {
        if (a[i] > max) max = a[i];
    }
    return max;
}

void f64_abs(double *dst, const double *a, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = fabs(a[i]);
    }
}

void f64_neg(double *dst, const double *a, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = -a[i];
    }
}

void f64_fma(double *dst, const double *a, const double *b, const double *c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = fma(a[i], b[i], c[i]);
    }
}

void f64_clamp(double *dst, const double *a, double min_val, double max_val, size_t n) {
    for (size_t i = 0; i < n; i++) {
        double v = a[i];
        if (v < min_val) v = min_val;
        if (v > max_val) v = max_val;
        dst[i] = v;
    }
}

/* ============================================================================
 * float32 operations
 * ============================================================================ */

float f32_dot_product(const float *a, const float *b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

void f32_add(float *dst, const float *a, const float *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = a[i] + b[i];
    }
}

void f32_sub(float *dst, const float *a, const float *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = a[i] - b[i];
    }
}

void f32_mul(float *dst, const float *a, const float *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = a[i] * b[i];
    }
}

void f32_div(float *dst, const float *a, const float *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = a[i] / b[i];
    }
}

void f32_scale(float *dst, const float *a, float s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = a[i] * s;
    }
}

void f32_add_scalar(float *dst, const float *a, float s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = a[i] + s;
    }
}

float f32_sum(const float *a, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += a[i];
    }
    return sum;
}

float f32_min(const float *a, size_t n) {
    if (n == 0) return 0.0f;
    float min = a[0];
    for (size_t i = 1; i < n; i++) {
        if (a[i] < min) min = a[i];
    }
    return min;
}

float f32_max(const float *a, size_t n) {
    if (n == 0) return 0.0f;
    float max = a[0];
    for (size_t i = 1; i < n; i++) {
        if (a[i] > max) max = a[i];
    }
    return max;
}

void f32_abs(float *dst, const float *a, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = fabsf(a[i]);
    }
}

void f32_neg(float *dst, const float *a, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = -a[i];
    }
}

void f32_fma(float *dst, const float *a, const float *b, const float *c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = fmaf(a[i], b[i], c[i]);
    }
}

void f32_clamp(float *dst, const float *a, float min_val, float max_val, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float v = a[i];
        if (v < min_val) v = min_val;
        if (v > max_val) v = max_val;
        dst[i] = v;
    }
}

/* ============================================================================
 * Output helpers
 * ============================================================================ */

void print_f64_array(const char *name, const double *a, size_t n) {
    printf("%s: [", name);
    for (size_t i = 0; i < n; i++) {
        printf("%.17g", a[i]);
        if (i < n - 1) printf(", ");
    }
    printf("]\n");
}

void print_f32_array(const char *name, const float *a, size_t n) {
    printf("%s: [", name);
    for (size_t i = 0; i < n; i++) {
        printf("%.9g", a[i]);
        if (i < n - 1) printf(", ");
    }
    printf("]\n");
}

void print_go_f64_slice(const char *name, const double *a, size_t n) {
    printf("%s := []float64{", name);
    for (size_t i = 0; i < n; i++) {
        printf("%.17g", a[i]);
        if (i < n - 1) printf(", ");
    }
    printf("}\n");
}

void print_go_f32_slice(const char *name, const float *a, size_t n) {
    printf("%s := []float32{", name);
    for (size_t i = 0; i < n; i++) {
        printf("%.9g", a[i]);
        if (i < n - 1) printf(", ");
    }
    printf("}\n");
}

/* ============================================================================
 * Test vector generation
 * ============================================================================ */

int main(void) {
    printf("// Generated test expectations from C reference implementation\n");
    printf("// Compile with: gcc -O2 -march=native -o generate_expectations generate_expectations.c -lm\n\n");

    /* --------------------------------------------------------------------------
     * Test Case 1: Small vectors with exact values (SIMD boundary tests)
     * -------------------------------------------------------------------------- */
    printf("// =============================================================================\n");
    printf("// Test Case 1: SIMD boundary tests (sizes 1, 3, 4, 5, 7, 8, 9, 15, 16, 17)\n");
    printf("// =============================================================================\n\n");

    // Test sizes that exercise SIMD boundaries
    size_t test_sizes[] = {1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33};
    size_t num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);

    for (size_t si = 0; si < num_sizes; si++) {
        size_t n = test_sizes[si];
        printf("// --- Size %zu ---\n", n);

        double *a64 = malloc(n * sizeof(double));
        double *b64 = malloc(n * sizeof(double));
        double *c64 = malloc(n * sizeof(double));
        double *dst64 = malloc(n * sizeof(double));

        float *a32 = malloc(n * sizeof(float));
        float *b32 = malloc(n * sizeof(float));
        float *c32 = malloc(n * sizeof(float));
        float *dst32 = malloc(n * sizeof(float));

        // Initialize with predictable values
        for (size_t i = 0; i < n; i++) {
            a64[i] = (double)(i + 1);
            b64[i] = (double)(n - i);
            c64[i] = 0.5;

            a32[i] = (float)(i + 1);
            b32[i] = (float)(n - i);
            c32[i] = 0.5f;
        }

        printf("// float64, n=%zu\n", n);
        print_go_f64_slice("a", a64, n);
        print_go_f64_slice("b", b64, n);

        printf("dotProduct := %.17g\n", f64_dot_product(a64, b64, n));
        printf("sum := %.17g\n", f64_sum(a64, n));
        printf("min := %.17g\n", f64_min(a64, n));
        printf("max := %.17g\n", f64_max(a64, n));

        f64_add(dst64, a64, b64, n);
        print_go_f64_slice("add", dst64, n);

        f64_sub(dst64, a64, b64, n);
        print_go_f64_slice("sub", dst64, n);

        f64_mul(dst64, a64, b64, n);
        print_go_f64_slice("mul", dst64, n);

        f64_scale(dst64, a64, 2.5, n);
        print_go_f64_slice("scale", dst64, n);

        f64_abs(dst64, a64, n);
        print_go_f64_slice("abs", dst64, n);

        f64_neg(dst64, a64, n);
        print_go_f64_slice("neg", dst64, n);

        f64_fma(dst64, a64, b64, c64, n);
        print_go_f64_slice("fma", dst64, n);

        f64_clamp(dst64, a64, 2.0, 5.0, n);
        print_go_f64_slice("clamp", dst64, n);

        printf("\n// float32, n=%zu\n", n);
        print_go_f32_slice("a32", a32, n);
        print_go_f32_slice("b32", b32, n);

        printf("dotProduct32 := float32(%.9g)\n", f32_dot_product(a32, b32, n));
        printf("sum32 := float32(%.9g)\n", f32_sum(a32, n));
        printf("min32 := float32(%.9g)\n", f32_min(a32, n));
        printf("max32 := float32(%.9g)\n", f32_max(a32, n));

        f32_add(dst32, a32, b32, n);
        print_go_f32_slice("add32", dst32, n);

        f32_fma(dst32, a32, b32, c32, n);
        print_go_f32_slice("fma32", dst32, n);

        printf("\n");

        free(a64); free(b64); free(c64); free(dst64);
        free(a32); free(b32); free(c32); free(dst32);
    }

    /* --------------------------------------------------------------------------
     * Test Case 2: Negative values and mixed signs
     * -------------------------------------------------------------------------- */
    printf("// =============================================================================\n");
    printf("// Test Case 2: Negative values and mixed signs\n");
    printf("// =============================================================================\n\n");

    {
        size_t n = 10;
        double a64[] = {-5, -4, -3, -2, -1, 1, 2, 3, 4, 5};
        double b64[] = {5, 4, 3, 2, 1, -1, -2, -3, -4, -5};
        double c64[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
        double dst64[10];

        float a32[] = {-5, -4, -3, -2, -1, 1, 2, 3, 4, 5};
        float b32[] = {5, 4, 3, 2, 1, -1, -2, -3, -4, -5};
        float c32[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f};
        float dst32[10];

        printf("// float64 mixed signs\n");
        print_go_f64_slice("aMixed", a64, n);
        print_go_f64_slice("bMixed", b64, n);
        print_go_f64_slice("cMixed", c64, n);

        printf("dotProductMixed := %.17g\n", f64_dot_product(a64, b64, n));
        printf("sumMixed := %.17g\n", f64_sum(a64, n));
        printf("minMixed := %.17g\n", f64_min(a64, n));
        printf("maxMixed := %.17g\n", f64_max(a64, n));

        f64_add(dst64, a64, b64, n);
        print_go_f64_slice("addMixed", dst64, n);

        f64_sub(dst64, a64, b64, n);
        print_go_f64_slice("subMixed", dst64, n);

        f64_mul(dst64, a64, b64, n);
        print_go_f64_slice("mulMixed", dst64, n);

        f64_abs(dst64, a64, n);
        print_go_f64_slice("absMixed", dst64, n);

        f64_neg(dst64, a64, n);
        print_go_f64_slice("negMixed", dst64, n);

        f64_fma(dst64, a64, b64, c64, n);
        print_go_f64_slice("fmaMixed", dst64, n);

        printf("\n// float32 mixed signs\n");
        print_go_f32_slice("aMixed32", a32, n);
        print_go_f32_slice("bMixed32", b32, n);

        printf("dotProductMixed32 := float32(%.9g)\n", f32_dot_product(a32, b32, n));

        f32_add(dst32, a32, b32, n);
        print_go_f32_slice("addMixed32", dst32, n);

        f32_abs(dst32, a32, n);
        print_go_f32_slice("absMixed32", dst32, n);

        f32_fma(dst32, a32, b32, c32, n);
        print_go_f32_slice("fmaMixed32", dst32, n);

        printf("\n");
    }

    /* --------------------------------------------------------------------------
     * Test Case 3: Floating-point precision edge cases
     * -------------------------------------------------------------------------- */
    printf("// =============================================================================\n");
    printf("// Test Case 3: Floating-point precision edge cases\n");
    printf("// =============================================================================\n\n");

    {
        // Test with values that can cause precision issues
        size_t n = 8;
        double a64[] = {1e15, 1.0, -1e15, 1.0, 1e-15, 1e15, 1e-15, -1e15};
        double b64[] = {1.0, 1e15, 1.0, -1e15, 1e15, 1e-15, -1e15, 1e-15};
        double c64[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        double dst64[8];

        printf("// float64 precision edge cases\n");
        print_go_f64_slice("aPrecision", a64, n);
        print_go_f64_slice("bPrecision", b64, n);

        printf("dotProductPrecision := %.17g\n", f64_dot_product(a64, b64, n));
        printf("sumPrecision := %.17g\n", f64_sum(a64, n));

        f64_add(dst64, a64, b64, n);
        print_go_f64_slice("addPrecision", dst64, n);

        f64_mul(dst64, a64, b64, n);
        print_go_f64_slice("mulPrecision", dst64, n);

        f64_fma(dst64, a64, b64, c64, n);
        print_go_f64_slice("fmaPrecision", dst64, n);

        printf("\n");
    }

    /* --------------------------------------------------------------------------
     * Test Case 4: Special values (zeros, very small, very large)
     * -------------------------------------------------------------------------- */
    printf("// =============================================================================\n");
    printf("// Test Case 4: Special values\n");
    printf("// =============================================================================\n\n");

    {
        size_t n = 8;
        double a64[] = {0.0, -0.0, DBL_MIN, -DBL_MIN, DBL_MAX/2, -DBL_MAX/2, 1.0, -1.0};
        double b64[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, -0.0};
        double dst64[8];

        float a32[] = {0.0f, -0.0f, FLT_MIN, -FLT_MIN, FLT_MAX/2, -FLT_MAX/2, 1.0f, -1.0f};
        float b32[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, -0.0f};
        float dst32[8];

        printf("// float64 special values\n");
        print_go_f64_slice("aSpecial", a64, n);
        print_go_f64_slice("bSpecial", b64, n);

        printf("sumSpecial := %.17g\n", f64_sum(a64, n));
        printf("minSpecial := %.17g\n", f64_min(a64, n));
        printf("maxSpecial := %.17g\n", f64_max(a64, n));

        f64_add(dst64, a64, b64, n);
        print_go_f64_slice("addSpecial", dst64, n);

        f64_mul(dst64, a64, b64, n);
        print_go_f64_slice("mulSpecial", dst64, n);

        f64_abs(dst64, a64, n);
        print_go_f64_slice("absSpecial", dst64, n);

        printf("\n// float32 special values\n");
        print_go_f32_slice("aSpecial32", a32, n);
        print_go_f32_slice("bSpecial32", b32, n);

        printf("sumSpecial32 := float32(%.9g)\n", f32_sum(a32, n));
        printf("minSpecial32 := float32(%.9g)\n", f32_min(a32, n));
        printf("maxSpecial32 := float32(%.9g)\n", f32_max(a32, n));

        f32_abs(dst32, a32, n);
        print_go_f32_slice("absSpecial32", dst32, n);

        printf("\n");
    }

    /* --------------------------------------------------------------------------
     * Test Case 5: Large arrays (typical DSP sizes: 256, 512, 1024)
     * -------------------------------------------------------------------------- */
    printf("// =============================================================================\n");
    printf("// Test Case 5: Large arrays (DSP sizes)\n");
    printf("// =============================================================================\n\n");

    {
        size_t sizes[] = {256, 277, 512, 1024};
        for (size_t si = 0; si < 4; si++) {
            size_t n = sizes[si];

            double *a64 = malloc(n * sizeof(double));
            double *b64 = malloc(n * sizeof(double));

            float *a32 = malloc(n * sizeof(float));
            float *b32 = malloc(n * sizeof(float));

            // Initialize with sine-like pattern (common in DSP)
            for (size_t i = 0; i < n; i++) {
                a64[i] = sin(2.0 * 3.14159265358979323846 * (double)i / (double)n);
                b64[i] = cos(2.0 * 3.14159265358979323846 * (double)i / (double)n);

                a32[i] = sinf(2.0f * 3.14159265f * (float)i / (float)n);
                b32[i] = cosf(2.0f * 3.14159265f * (float)i / (float)n);
            }

            printf("// Large array n=%zu (sine/cosine pattern)\n", n);
            printf("// float64\n");
            printf("dotProduct_%zu := %.17g\n", n, f64_dot_product(a64, b64, n));
            printf("sum_%zu := %.17g\n", n, f64_sum(a64, n));
            printf("min_%zu := %.17g\n", n, f64_min(a64, n));
            printf("max_%zu := %.17g\n", n, f64_max(a64, n));

            printf("// float32\n");
            printf("dotProduct32_%zu := float32(%.9g)\n", n, f32_dot_product(a32, b32, n));
            printf("sum32_%zu := float32(%.9g)\n", n, f32_sum(a32, n));
            printf("min32_%zu := float32(%.9g)\n", n, f32_min(a32, n));
            printf("max32_%zu := float32(%.9g)\n", n, f32_max(a32, n));

            printf("\n");

            free(a64); free(b64);
            free(a32); free(b32);
        }
    }

    /* --------------------------------------------------------------------------
     * Test Case 6: Division edge cases
     * -------------------------------------------------------------------------- */
    printf("// =============================================================================\n");
    printf("// Test Case 6: Division edge cases\n");
    printf("// =============================================================================\n\n");

    {
        size_t n = 8;
        double a64[] = {10.0, -10.0, 1.0, -1.0, 100.0, 0.01, 1e10, 1e-10};
        double b64[] = {2.0, -2.0, 3.0, -3.0, 0.1, 100.0, 1e-5, 1e5};
        double dst64[8];

        float a32[] = {10.0f, -10.0f, 1.0f, -1.0f, 100.0f, 0.01f, 1e10f, 1e-10f};
        float b32[] = {2.0f, -2.0f, 3.0f, -3.0f, 0.1f, 100.0f, 1e-5f, 1e5f};
        float dst32[8];

        printf("// float64 division\n");
        print_go_f64_slice("aDiv", a64, n);
        print_go_f64_slice("bDiv", b64, n);

        f64_div(dst64, a64, b64, n);
        print_go_f64_slice("divResult", dst64, n);

        printf("\n// float32 division\n");
        print_go_f32_slice("aDiv32", a32, n);
        print_go_f32_slice("bDiv32", b32, n);

        f32_div(dst32, a32, b32, n);
        print_go_f32_slice("divResult32", dst32, n);

        printf("\n");
    }

    /* --------------------------------------------------------------------------
     * Test Case 7: Clamp edge cases
     * -------------------------------------------------------------------------- */
    printf("// =============================================================================\n");
    printf("// Test Case 7: Clamp edge cases\n");
    printf("// =============================================================================\n\n");

    {
        size_t n = 12;
        double a64[] = {-100, -10, -1, -0.5, 0, 0.5, 1, 5, 10, 50, 100, 1000};
        double dst64[12];

        float a32[] = {-100, -10, -1, -0.5f, 0, 0.5f, 1, 5, 10, 50, 100, 1000};
        float dst32[12];

        printf("// float64 clamp tests\n");
        print_go_f64_slice("aClamp", a64, n);

        // Clamp to [0, 10]
        f64_clamp(dst64, a64, 0.0, 10.0, n);
        print_go_f64_slice("clamp_0_10", dst64, n);

        // Clamp to [-5, 5]
        f64_clamp(dst64, a64, -5.0, 5.0, n);
        print_go_f64_slice("clamp_neg5_5", dst64, n);

        // Clamp to [1, 100]
        f64_clamp(dst64, a64, 1.0, 100.0, n);
        print_go_f64_slice("clamp_1_100", dst64, n);

        printf("\n// float32 clamp tests\n");
        print_go_f32_slice("aClamp32", a32, n);

        f32_clamp(dst32, a32, 0.0f, 10.0f, n);
        print_go_f32_slice("clamp32_0_10", dst32, n);

        printf("\n");
    }

    /* --------------------------------------------------------------------------
     * Test Case 8: AddScalar tests
     * -------------------------------------------------------------------------- */
    printf("// =============================================================================\n");
    printf("// Test Case 8: AddScalar tests\n");
    printf("// =============================================================================\n\n");

    {
        size_t n = 8;
        double a64[] = {1, 2, 3, 4, 5, 6, 7, 8};
        double dst64[8];

        float a32[] = {1, 2, 3, 4, 5, 6, 7, 8};
        float dst32[8];

        printf("// float64 AddScalar\n");
        print_go_f64_slice("aAddScalar", a64, n);

        f64_add_scalar(dst64, a64, 10.5, n);
        print_go_f64_slice("addScalar_10_5", dst64, n);

        f64_add_scalar(dst64, a64, -3.0, n);
        print_go_f64_slice("addScalar_neg3", dst64, n);

        printf("\n// float32 AddScalar\n");
        print_go_f32_slice("aAddScalar32", a32, n);

        f32_add_scalar(dst32, a32, 10.5f, n);
        print_go_f32_slice("addScalar32_10_5", dst32, n);

        printf("\n");
    }

    printf("// End of generated test expectations\n");

    return 0;
}
