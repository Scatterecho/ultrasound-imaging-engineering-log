# 2026-05-20 — DAS kernel benchmark in PyBF

## What I worked on

- Returned to `bf_cores.py` and compared the two DAS implementations:
  - `delay_and_sum_numpy()`
  - `delay_and_sum_numba()`
- Confirmed that actual `@jit` usage in this project is concentrated in `delay_and_sum_numba()`.
- Added an independent benchmark script under the PyBF fork:
  - `tests/code/benchmark_das/main.py`

## Benchmark result

Test configuration used a reduced image size with `n_points = 21600` and varied the number of transmit modes.

```text
modes | points | NumPy med(s) | Numba med(s) | speedup | NumPy temp MB | rel err
    1 |  21600 |       0.0678 |       0.0045 |   15.04 |         110.7 | 2.84e-07
    3 |  21600 |       0.1810 |       0.0117 |   15.54 |         332.2 | 2.99e-07
    5 |  21600 |       0.2967 |       0.0218 |   13.64 |         553.7 | 2.99e-07
    9 |  21600 |       0.5167 |       0.0364 |   14.21 |         996.7 | 2.99e-07
```

## Key engineering takeaways

- The NumPy implementation is concise and useful as a reference implementation, but fancy indexing creates large temporary arrays with shape close to `(n_modes, n_points, n_elements)`.
- The Numba implementation keeps the explicit loop structure but compiles it to machine code and parallelizes over pixels with `prange`.
- For this DAS gather-and-reduce kernel, Numba is both faster and more memory-efficient than the NumPy fancy-indexing approach.
- The benchmark result scales roughly linearly with the number of modes, matching the expected complexity:

```text
O(n_modes × n_points × n_elements)
```

## Next step

Study `delay_calc.py` to understand the geometric transmit and receive delay model, especially for plane-wave imaging and future transcranial aberration-correction work.
