# LOS Feedback Controller - C Reference Implementation

This code was provided by Joel Shields on December 1, 2025.

It exists for the purpose of checking accuracy of the Rust implementation
(`los_controller.rs`) against the original C reference.

## Building

```bash
make
```

## Running

```bash
./los_test > reference_output.csv 2>&1
```

## Files

- `CF_LOS_FB_40Hz.c` - Original controller implementation
- `CF_LOS_FB_40Hz.h` - Controller coefficients (A, B, C, D matrices)
- `main.c` - Test harness with step response, sinusoidal, and hold mode tests
- `reference_output.csv` - Generated output for comparison with Rust tests

## Modifications from Original

The original files were modified as follows:

1. **Formatting**: Applied `clang-format` with Google style and Linux braces:
   ```bash
   clang-format -style="{BasedOnStyle: Google, BreakBeforeBraces: Linux}" -i *.c *.h
   ```

2. **Header guards**: Added `#ifndef`/`#define`/`#endif` guards to `CF_LOS_FB_40Hz.h`

3. **Function prototype**: Added `CF_LOS_FB_function` declaration to header file

4. **main.c**: Created self-contained test harness with:
   - Controller function and coefficients inlined for standalone compilation
   - Step response test (100 iterations with constant error)
   - Sinusoidal disturbance test (200 iterations)
   - Hold mode test (20 iterations)
   - Performance benchmark (1000 iterations)
   - High-precision (15 decimal places) output for numerical comparison
