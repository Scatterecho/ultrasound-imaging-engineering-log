# 2026-06-17 vbeam: JAX CPU benchmark and transmit reduction strategy

## Context

The previous vbeam notes had already covered the conceptual layers:

```text
Spec / ForAll / Apply / Reduce
apodization
wavefront models
speed of sound
interpolation / fastmath
spekk transformations
UFF importer
manual SignalForPointSetup
```

So I did not repeat the transformation-chain basics. The next useful step was to
move from reading the abstraction to measuring its engineering consequences on a
real PICMUS UFF dataset with the local Windows CPU/JAX environment.

The relevant local scripts are:

```text
friends/run_picmus_plane_wave.py
friends/manual_point_scatterer_demo.py
friends/benchmark_picmus_jax_cpu.py
friends/benchmark_picmus_transmit_strategy.py
```

## Manual point-scatterer closed loop

To make the manual setup more physical than the earlier shape-contract demo, I
created a point-scatterer simulation:

```text
friends/manual_point_scatterer_demo.py
```

The script bypasses UFF and directly constructs a `SignalForPointSetup` with:

```text
5 plane-wave transmits
32 receive elements
2048 time samples
one scatterer at x = 1.5 mm, z = 22.0 mm
```

The synthetic signal is generated using the same delay model later used by
`signal_for_point()`:

```text
tx_distance = x_scatterer * sin(azimuth) + z_scatterer * cos(azimuth)
rx_distance = norm(scatterer_position - receiver_position)
delay = (tx_distance + rx_distance) / speed_of_sound
```

Then vbeam is asked to beamform the image from that signal. The validated result
was:

```text
result shape: (96, 128)
expected scatterer x/z (m): 0.0015, 0.0220
beamformed peak x/z (m): 0.0014526, 0.0220079
absolute localization error: 0.048 mm
```

This confirmed that the manual setup is not only shape-compatible with vbeam,
but geometrically meaningful.

## PICMUS JAX CPU timing benchmark

I then added:

```text
friends/benchmark_picmus_jax_cpu.py
```

The goal was to separate several timing components that are often mixed together
when running a JAX demo:

```text
download/cache time
UFF read time
import_pyuff/setup time
get_das_beamformer build time
first JIT run time
second run time
third run time
```

The key point is that the first JAX run includes tracing, XLA compilation, and
execution. The second and third runs are a better estimate of the steady runtime
for one reconstructed frame with the same shapes.

The benchmark used the PICMUS carotid plane-wave dataset:

```text
raw channel_data.data shape = (1536, 128, 75)
native scan shape = 387 x 609
native num_points = 235683
```

The measured results were:

```text
case               scan    tx  points   first_run  second_run  third_run  steady_mean
resize_48x64_tx1   48x64   1   3072     0.117 s    0.002 s     0.002 s    0.002 s
resize_48x64_tx5   48x64   5   3072     0.192 s    0.006 s     0.005 s    0.005 s
resize_96x128_tx15 96x128  15  12288    0.231 s    0.034 s     0.032 s    0.033 s
resize_96x128_tx75 96x128  75  12288    0.349 s    0.146 s     0.152 s    0.149 s
native_scan_tx1    native  1   235683   0.231 s    0.075 s     0.080 s    0.077 s
native_scan_tx75   native  75  235683   3.302 s    2.605 s     2.616 s    2.611 s
```

The main takeaway:

```text
PICMUS native scan + 75 plane waves on CPU/JAX is about 2.6 s/frame after JIT.
96x128 resize + 75 plane waves is about 0.15 s/frame after JIT.
```

The first run should not be used as the normal frame time because it includes
JAX compilation overhead.

## Scaling with points and transmits

For fixed 128 receivers, the dominant work scale is approximately:

```text
points x receivers x transmits
```

For example:

```text
96x128 tx75:
12288 x 128 x 75 ~= 118 million delayed samples

native tx75:
235683 x 128 x 75 ~= 2.26 billion delayed samples
```

The compute scale increases by about 19.2x, while the steady runtime increased
from about 0.149 s to 2.611 s, about 17.5x. This is reasonable and suggests that
large CPU/JAX workloads amortize overhead better than very small image grids.

## Transmit reduction strategy benchmark

To test whether fully exposing the transmit dimension would be faster, I added:

```text
friends/benchmark_picmus_transmit_strategy.py
```

The script compares two strategies:

```text
default_reduce_sum:
    vbeam's default DAS path, using Reduce.Sum("transmits")

vmap_all_then_sum:
    vectorize over points, receivers, and transmits, then sum named axes
```

Both strategies reconstruct the same final image shape, but they have very
different memory behavior.

The benchmark results were:

```text
case         strategy            cube_GB   first_run   second_run   third_run   steady    RSS
96x128 tx75  default Reduce       0.879     0.351 s     0.148 s      0.146 s     0.147 s   0.45 GB
96x128 tx75  vmap-all then sum    0.879     0.868 s     0.687 s      0.692 s     0.689 s   1.32 GB

native tx75  default Reduce       16.857    2.847 s     2.614 s      2.632 s     2.623 s   1.09 GB
native tx75  vmap-all then sum    16.857    8.425 s     7.953 s      7.770 s     7.862 s   17.67 GB
```

Here `cube_GB` is the conceptual intermediate datacube size:

```text
points x receivers x transmits x sizeof(complex64)
```

For native scan + 75 transmits, this is approximately:

```text
235683 x 128 x 75 x 8 bytes ~= 16.86 GB
```

The measured RSS of the vmap-all strategy was about 17.67 GB, which is very close
to the conceptual datacube size. This indicates that, in this case, XLA did not
fully fuse away the large intermediate transmit/receiver/point cube.

## Why vmap-all was slower

The initial intuition was:

```text
If all transmits are vmapped, there is more parallelism, so it might be faster.
```

The benchmark showed the opposite on CPU:

```text
native tx75 default Reduce:    ~2.62 s/frame, ~1.09 GB RSS
native tx75 vmap-all then sum: ~7.86 s/frame, ~17.67 GB RSS
```

The reason is that default `Reduce.Sum("transmits")` is not simply non-parallel.
It still exposes a large amount of parallel work inside each transmit:

```text
points x receivers = 235683 x 128 ~= 30 million delayed samples per transmit
```

That is already enough to keep the CPU busy. Exposing the transmit dimension as
one more vmap axis creates more theoretical parallelism, but it also forces a huge
intermediate datacube to be written and read back for summation. On CPU, this
becomes memory-bandwidth and cache-pressure dominated.

A useful mental model is:

```text
Reduce.Sum("transmits"):
    for each transmit:
        compute points x receivers in parallel
        sum receivers
        accumulate into image
    keep memory low

vmap-all then sum:
    compute points x receivers x transmits
    store a large intermediate cube
    read it again for reduction
    memory pressure dominates
```

## Current takeaway

This benchmark is an important engineering checkpoint for vbeam:

```text
vbeam is not just rewriting DAS in JAX.
It uses named transformations to choose which dimensions are expanded in parallel
and which dimensions are reduced in a streaming or memory-saving way.
```

For the local CPU/JAX setup, the default transmit reduction strategy is both
faster and much more memory efficient than fully materializing the transmit
dimension.

This also suggests a practical future GPU lesson:

```text
More vmap axes are not automatically better.
The best strategy is usually tiled parallelism plus local reductions, not blindly
materializing points x receivers x transmits.
```

## Next step

Possible next directions:

1. Compare CPU/JAX behavior with a future GPU/JAX environment.
2. Explore point tiling or optimized scan strategies to reduce memory without
   sacrificing too much parallelism.
3. Profile where time is spent inside interpolation, phase correction,
   apodization, and reductions.
4. Start connecting these benchmarks to transcranial use cases where the scan,
   sound speed model, and aperture rule will be custom.