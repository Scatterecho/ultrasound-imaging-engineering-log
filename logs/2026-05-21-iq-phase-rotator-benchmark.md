# 2026-05-21 - IQ phase-rotator beamforming benchmark

Today I added and tested a direct IQ-domain DAS path in `pybf`.

## Motivation

The original realtime path performs:

```text
RF -> IQ demodulate -> interpolate x10 -> remodulate to analytic RF -> DAS
```

This keeps the DAS kernel simple, but it expands the per-frame signal array significantly.

The new experimental path performs:

```text
RF -> IQ demodulate -> fractional-delay IQ interpolation -> phase rotator -> DAS
```

The phase rotator restores the missing carrier phase term directly inside the DAS kernel:

```text
exp(+j 2*pi*fc*tau)
```

## Implementation

Added:

```text
pybf/bf_cores.py
    delay_and_sum_iq_phase_rotator_numba(...)

scripts/beamformer_cartesian_realtime.py
    beamform(..., iq_phase_correction=True)

tests/code/benchmark_iq_phase_rotator/main.py
    Full-frame timing, memory estimate, and image-quality comparison script.
```

The first version used low-rate integer IQ delay indexing, which was fast but gave a larger image mismatch. I then upgraded it to use linear fractional-delay interpolation in IQ before applying the phase rotator.

## Benchmark command

```powershell
python tests\code\benchmark_iq_phase_rotator\main.py --image-res 400 600 --repeats 10
```

## Result

```text
Analytic RF median time: 0.2955 s
Direct IQ median time:   0.1442 s
Speedup RF/IQ:           2.05x

Analytic RF preprocessed memory estimate: 67.8 MB
Direct IQ preprocessed memory estimate:   3.4 MB
Memory ratio:                          20.0x

Complex relative L2 error: 0.1654
Magnitude relative L2 error: 0.1295
dB image correlation:        0.9667
Mean absolute dB diff:       1.9491 dB
95th percentile |dB diff|:   5.8799 dB
```

## Engineering lesson

Direct IQ beamforming can substantially reduce memory and improve speed, but it is not automatically equivalent to remodulated analytic-RF DAS. The quality depends heavily on how fractional delay and phase rotation are implemented.

For the next optimization step, likely directions are:

- precompute phase rotators or use lookup tables;
- replace per-sample `cos/sin` with recurrence or cached tables;
- test higher-order IQ interpolation;
- split kernel timing to isolate interpolation cost vs phase-rotation cost.

## Code commit

```text
1d4d75d Add IQ phase-rotator beamforming benchmark
```
