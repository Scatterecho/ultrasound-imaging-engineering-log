# 2026-05-19 — PyBF realtime beamformer study

## What I worked on

- Continued studying `BFCartesianRealTime` in PyBF.
- Reviewed the engineering split between initialization, preprocessing, and per-frame beamforming.
- Traced the main array shapes through the realtime beamforming pipeline.
- Studied the core DAS implementation in `delay_and_sum_numpy()`.

## Key engineering takeaways

- Geometry-dependent quantities such as pixel coordinates, propagation delays, sample indices, and receive apodization are precomputed in `__init__()` because they do not change frame by frame.
- The RF input is normalized from a single acquisition shape `(channels, samples)` into `(n_acq, channels, samples)` so that single-angle and multi-angle processing share the same internal interface.
- The DAS core uses NumPy fancy indexing to gather delayed RF samples for all pixels and channels, then sums along the channel axis.
- This vectorized NumPy implementation is concise and useful for learning, but it creates a large intermediate array and is not necessarily how a production realtime beamformer should be implemented.

## Shapes traced today

```text
rf_data:        (192, 2313)
reshape:        (1, 192, 2313)
preprocess:     (192, 23130)
transpose:      (23130, 192)
delays_samples: (192, 240000)
DAS output:     (1, 240000)
image:          (600, 400)
```

## Next step

Study `delay_calc.py`, especially how plane-wave transmit delay and receive delay are computed geometrically.
