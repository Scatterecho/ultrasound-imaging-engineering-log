# 2026-06-24 vbeam: remote server GPU benchmark comparison

## Goal

Benchmark vbeam's JAX backend on the remote Linux server and compare it with the
previous Windows CPU and local WSL2 RTX 4060 results.

The server environment was:

```text
GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
VRAM: about 96 GB
JAX: 0.10.2
backend: gpu
physical GPU selected: GPU 1
process-visible device: CudaDevice(id=0)
```

Runtime environment variables:

```bash
export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

## PICMUS DAS benchmark command

The full benchmark command was:

```bash
cd /data1/wjy23bs/vbeam_project/code/vbeam
. /data1/wjy23bs/vbeam_project/venvs/vbeam-jax-gpu/bin/activate
export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python -s friends/benchmark_picmus_jax_cpu.py --include-native-full
```

The script name still contains `cpu` for historical reasons, but it records the
actual JAX backend in the CSV. On the server it ran with:

```text
backend: gpu
device: NVIDIA RTX PRO 6000 Blackwell Server Edition
```

The UFF data was read from the cached server path:

```text
friends/data/www.ustb.no/datasets/PICMUS_carotid_cross.uff
```

The raw channel data shape was:

```text
(1536, 128, 75)
```

## Server benchmark results

Steady time is the mean of the second and third runs, after JIT compilation.

```text
case                  points   tx   result    steady_mean_s
resize_48x64_tx1        3072    1   48x64        0.002252
resize_48x64_tx5        3072    5   48x64        0.002464
resize_96x128_tx15     12288   15   96x128       0.002247
resize_96x128_tx75     12288   75   96x128       0.003801
native_scan_tx1       235683    1   387x609      0.002644
native_scan_tx75      235683   75   387x609      0.022162
```

For the full native scan with all 75 plane waves:

```text
native_scan_tx75: about 0.0222 s/frame
throughput: about 45 compound frames/s
```

## Three-platform comparison

Previous baselines:

```text
Windows CPU:
    C:\Users\Admin\.conda\envs\vbeam_jax_cpu
    JAX 0.6.2

Local WSL2 GPU:
    NVIDIA GeForce RTX 4060 8 GB
    JAX 0.10.2

Remote server GPU:
    NVIDIA RTX PRO 6000 Blackwell Server Edition
    JAX 0.10.2
```

Steady-state comparison:

```text
case                 Windows CPU   RTX 4060     RTX PRO 6000   server/CPU   server/4060
resize_48x64_tx1      0.00215 s    0.00258 s     0.00225 s        1.0x        1.1x
resize_48x64_tx5      0.00506 s    0.00348 s     0.00246 s        2.1x        1.4x
resize_96x128_tx15    0.0343  s    0.0179  s     0.00225 s       15.2x        8.0x
resize_96x128_tx75    0.1547  s    0.0875  s     0.00380 s       40.7x       23.0x
native_scan_tx1       0.0856  s    0.00336 s     0.00264 s       32.4x        1.3x
native_scan_tx75      2.679   s    0.1199  s     0.0222  s      120.9x        5.4x
```

## Interpretation of native_scan_tx1 and native_scan_tx75

The server GPU does not improve over the RTX 4060 by a theoretical dense-matrix
FLOPS ratio. This is expected because pixel-based DAS is not a regular GEMM-like
workload.

For `native_scan_tx75`, the work scale is:

```text
points x receivers x transmits
= 235683 x 128 x 75
approximately 2.26 billion channel contributions per compound frame
```

Each contribution involves delay calculation, interpolation/gather from the
signal, and reductions. This access pattern is much less regular than dense
matrix multiplication. Performance can be limited by:

```text
memory access and gather/interpolation pattern
cache behavior
kernel fusion choices made by XLA
reduction structure
register pressure
kernel launch and scheduling overhead
host-to-device orchestration
```

The small `native_scan_tx1` case contains only:

```text
235683 x 128 x 1
approximately 30 million contributions
```

which is relatively small for a large server GPU. Fixed overheads and scheduling
become a larger part of the measured time, so the server only improves over the
4060 by about 1.3x. The 75-transmit case exposes much more parallel work and the
server improves by about 5.4x over the 4060.

## Is 45 fps real time?

For a compounded B-mode frame made from 75 plane waves, about 45 frames/s is a
usable real-time baseline. However, it should not be confused with ultrafast
single-plane-wave frame rate.

The throughput can also be viewed as:

```text
45 compound frames/s x 75 plane waves/frame
approximately 3375 plane waves/s equivalent reconstruction throughput
```

Whether this is sufficient depends on the imaging mode:

```text
Conventional displayed B-mode:
    45 compound frames/s can be considered real-time.

Ultrafast imaging or high-frame-rate functional imaging:
    45 compounded frames/s may be too low, especially if many additional
    processing stages are required.

Full engineering pipeline:
    acquisition PRF, data transfer, filtering, beamforming, scan conversion,
    display, and any aberration/sound-speed correction all matter.
```

The current result is therefore a strong baseline, not a final optimized real-time
system.

## Transmit strategy benchmark

Command:

```bash
python -s friends/benchmark_picmus_transmit_strategy.py
```

This compares:

```text
default_reduce_sum:
    vbeam's default Reduce.Sum over transmits

vmap_all_then_sum:
    expose the transmit dimension through vectorization, then sum
```

Server result for `96x128`, 128 receivers, and 75 transmits:

```text
strategy             first_run_s   second_run_s   third_run_s   steady_mean_s   host RSS after third
default_reduce_sum     1.4132        0.00422        0.00359        0.00391          1.206 GB
vmap_all_then_sum      0.8576        0.00346        0.00294        0.00320          1.224 GB
```

For this server and this moderate case, `vmap_all_then_sum` was about:

```text
0.00391 / 0.00320 = 1.22x faster
```

The conceptual complex64 datacube size for this case is:

```text
12288 x 128 x 75 x 8 bytes
approximately 0.879 GB
```

This is small relative to the server GPU's VRAM, which explains why explicitly
exposing more parallelism can help without causing memory pressure.

Important limitation: the script's `psutil` RSS reports host process memory, not
actual NVIDIA VRAM. GPU memory must be monitored separately with `nvidia-smi`.

## Engineering takeaways

1. Tiny cases are overhead-dominated, so large server GPUs do not show their full
   advantage.
2. Larger cases such as `96x128_tx75` and `native_scan_tx75` expose enough work
   for the server GPU to pull away from the RTX 4060.
3. Pixel-based DAS is memory/gather/reduction heavy, so dense-FLOPS expectations
   are misleading.
4. The full 75-angle native PICMUS frame runs at about 45 compound frames/s on the
   server GPU.
5. For real engineering acceleration, likely next directions are:

```text
fewer plane waves or adaptive angle selection
ROI or lower display grid resolution
precomputed delay/LUT strategies
IQ + phase rotator variants
tiling/blocking to improve memory behavior
specialized CUDA/CuPy/Numba-CUDA kernels for fixed scans
more robust benchmark methodology with warmups, repeats, medians, and VRAM logging
```
