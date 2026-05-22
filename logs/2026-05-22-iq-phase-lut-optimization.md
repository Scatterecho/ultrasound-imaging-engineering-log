# 2026-05-22 - IQ phase rotation optimized with lookup tables

Today I optimized the direct IQ-domain beamforming path added earlier.

## Motivation

The first direct IQ implementation restored carrier phase inside the DAS kernel by evaluating:

```text
exp(j * 2*pi*fc*tau)
```

for every pixel-channel delay. This is physically clear, but expensive because the inner loop repeatedly calls trigonometric functions.

## Optimization

I added a phase lookup-table path:

```text
exp(j * phase(sample_idx + frac))
≈ integer_phase_lut[sample_idx] * fractional_phase_lut[frac_bin]
```

The beamforming path now supports:

```python
bf.beamform(
    rf_data,
    numba_active=True,
    iq_phase_correction=True,
    iq_phase_lut_size=256,
)
```

## Files changed

```text
pybf/bf_cores.py
scripts/beamformer_cartesian_realtime.py
tests/code/benchmark_iq_phase_rotator/main.py
```

## Benchmark

Command:

```powershell
python tests\code\benchmark_iq_phase_rotator\main.py --image-res 400 600 --repeats 5 --phase-lut-size 256
```

Result:

```text
Analytic RF median time:      0.3060 s
Direct IQ median time:        0.1609 s
Phase-LUT IQ median time:     0.1168 s

Speedup RF/direct-IQ:         1.90x
Speedup RF/phase-LUT-IQ:      2.62x
Speedup direct/phase-LUT:     1.38x
```

Image quality versus analytic-RF reference:

```text
Direct IQ dB correlation:     0.9667
Phase-LUT IQ dB correlation:  0.9667

Direct IQ mean dB diff:       1.9491 dB
Phase-LUT mean dB diff:       1.9491 dB
```

## Engineering lesson

Pre-rotating the IQ samples before fractional interpolation looked tempting, but it changed the order of operations and degraded image similarity. The LUT method is better because it preserves the original operation order:

```text
fractional-delay IQ interpolation -> phase rotation -> apodized DAS
```

This gave a meaningful speedup with essentially unchanged image quality.

## Code commit

```text
233135b Optimize IQ phase rotation with lookup tables
```
