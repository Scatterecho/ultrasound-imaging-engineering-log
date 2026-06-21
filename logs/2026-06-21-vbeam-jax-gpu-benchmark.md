# 2026-06-21 vbeam: local JAX GPU environment and benchmark

## Goal

Continue from the Windows CPU/JAX benchmark and build a separate WSL2 Linux
environment for running vbeam on the local NVIDIA GeForce RTX 4060 8 GB GPU.

The CPU and GPU environments are deliberately separate:

```text
Windows CPU environment:
C:\Users\Admin\.conda\envs\vbeam_jax_cpu
JAX 0.6.2

WSL2 GPU environment:
/home/scatter/venvs/vbeam-jax-gpu
JAX 0.10.2
```

The JAX version difference is an important experimental limitation: the current
CPU/GPU comparison includes both a device difference and a JAX version difference.

## WSL2 Python environment

Ubuntu 26.04 ships with Python 3.14.4, which is too new for a conservative
scientific Python environment. I used `uv` to install Python 3.11 and created the
virtual environment in Ubuntu's native Linux filesystem:

```text
/home/scatter/venvs/vbeam-jax-gpu
```

The environment was activated with:

```bash
source ~/venvs/vbeam-jax-gpu/bin/activate
```

A `uv venv` does not necessarily include the `pip` module. When:

```bash
python -m pip install --upgrade pip
```

returned:

```text
No module named pip
```

I installed pip into the active environment through uv:

```bash
source ~/.local/bin/env
uv pip install --upgrade pip setuptools wheel
python -m pip --version
```

Then vbeam and the CUDA JAX wheel were installed from the Windows-hosted project:

```bash
cd /mnt/g/Projects_Ultrasound/2_magnusdk_vbeam/vbeam
source ~/venvs/vbeam-jax-gpu/bin/activate
python -m pip install -e ".[test]" matplotlib ipykernel jupyterlab psutil
python -m pip install -U "jax[cuda13]"
```

## JAX GPU verification

For the 8 GB RTX 4060, automatic JAX preallocation was disabled in the current
terminal:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

The verification command reported:

```text
jax: 0.10.2
devices: [CudaDevice(id=0)]
default backend: gpu
```

This confirms the full execution path:

```text
Windows NVIDIA driver
-> WSL2
-> Ubuntu
-> Python 3.11 venv
-> JAX CUDA wheel
-> NVIDIA GeForce RTX 4060
```

Both of the following scripts also ran successfully and produced output:

```bash
python -s friends/manual_point_scatterer_demo.py
python -s friends/run_picmus_plane_wave.py
```

## Windows and Linux file locations

The source project is still stored on the Windows G drive:

```text
Windows:
G:\Projects_Ultrasound\2_magnusdk_vbeam\vbeam

Ubuntu/WSL:
/mnt/g/Projects_Ultrasound/2_magnusdk_vbeam/vbeam
```

Therefore output such as:

```text
/mnt/g/Projects_Ultrasound/2_magnusdk_vbeam/vbeam/friends/outputs/manual_point_scatterer_abs.png
```

is physically stored on the Windows G drive.

The Python/JAX environment is different. It is under:

```text
/home/scatter/venvs/vbeam-jax-gpu
```

and is therefore stored inside Ubuntu's virtual disk:

```text
G:\WSL\Ubuntu\ext4.vhdx
```

Jupyter remains useful in Linux workflows. A typical arrangement is:

```text
Windows browser: JupyterLab user interface
WSL Ubuntu: Jupyter server and Python kernel
RTX 4060: JAX computation
/mnt/g project path: notebook and output files on the Windows G drive
```

For repeatable performance measurements, standalone Python scripts are preferable
to notebooks because they control process state, execution order, and output more
reliably.

## Device-aware benchmark CSV files

The two benchmark scripts were updated to record:

```text
backend
jax_version
device
```

and to write backend-specific files instead of overwriting CPU results:

```text
friends/outputs/picmus_jax_benchmark_cpu.csv
friends/outputs/picmus_jax_benchmark_gpu.csv
friends/outputs/picmus_transmit_strategy_benchmark_cpu.csv
friends/outputs/picmus_transmit_strategy_benchmark_gpu.csv
```

The scripts automatically use `jax.default_backend()` in the output filename.

## CPU and GPU DAS benchmark

The benchmark command in Ubuntu was:

```bash
python -s friends/benchmark_picmus_jax_cpu.py --include-native-full
```

Despite the historical `cpu` word in the script filename, it runs on the current
JAX default backend. The new CSV explicitly identifies the backend and device.

The fresh CPU/GPU steady-state comparison is:

```text
case               CPU JAX 0.6.2   RTX 4060 JAX 0.10.2   CPU/GPU
resize_48x64_tx1      0.002146 s       0.002579 s          0.83x
resize_48x64_tx5      0.005058 s       0.003484 s          1.45x
resize_96x128_tx15    0.034262 s       0.017913 s          1.91x
resize_96x128_tx75    0.154737 s       0.087465 s          1.77x
native_scan_tx1       0.085587 s       0.003361 s         25.46x
native_scan_tx75      2.678789 s       0.119905 s         22.34x
```

For the full native PICMUS scan with 75 plane waves:

```text
CPU: approximately 2.68 s/frame
RTX 4060: approximately 0.120 s/frame
GPU throughput: approximately 8.3 frames/s
```

The GPU benefit is small for tiny images because launch and scheduling overhead
are comparable to the computation. The native scan exposes enough
point-receiver work to saturate the GPU and reaches more than 20x speedup.

## Transmit strategy benchmark

The strategy benchmark command was:

```bash
python -s friends/benchmark_picmus_transmit_strategy.py
```

It compares:

```text
default_reduce_sum:
    Reduce.Sum("transmits")

vmap_all_then_sum:
    expose the transmit dimension through vmap, then sum
```

For `96x128`, 128 receivers, and 75 transmits, the conceptual complex64 datacube
is approximately 0.879 GB and fits in the RTX 4060's 8 GB VRAM.

The latest backend-specific CSV results are:

```text
strategy             CPU steady       GPU steady       CPU/GPU
Default Reduce        0.155250 s        0.008772 s       17.70x
vmap-all then sum     0.662100 s        0.019256 s       34.38x
```

Within the latest GPU run, default Reduce was about 2.2x faster than vmap-all.
However, an earlier GPU run measured default Reduce at about 0.0306 s and
vmap-all at about 0.0202 s, where vmap-all was faster. The vmap-all result was
relatively stable across the two runs, while the Reduce result changed
substantially.

This means the current two-run experiment is not sufficient to state an absolute
GPU strategy winner. Likely factors include:

```text
GPU clock and warm state
JAX/XLA compilation and runtime caches
kernel-launch overhead
small number of timed repetitions
background GPU activity on the Windows host
```

A stronger next benchmark should use:

```text
explicit untimed warm-up runs
more repeated timed runs
median and percentile statistics
GPU device synchronization after every run
nvidia-smi or another method for VRAM measurement
```

## Memory measurement limitation

The strategy CSV uses `psutil` RSS. Under WSL GPU execution, RSS measures Linux
host-process memory, not NVIDIA VRAM. It must not be interpreted as GPU memory
usage.

Actual GPU memory should be observed separately, for example in another Ubuntu
terminal:

```bash
watch -n 0.2 nvidia-smi
```

The native vmap-all case is still intentionally avoided on the 8 GB RTX 4060:

```text
native conceptual datacube: approximately 16.86 GB complex64
RTX 4060 VRAM: approximately 8 GB
```

## Current takeaway

The local WSL2 GPU environment is now complete and functional. Real PICMUS vbeam
DAS runs correctly on the RTX 4060 and the full native 75-transmit reconstruction
is roughly 22x faster than the existing Windows CPU/JAX baseline.

The experiment also shows why benchmark design matters: large reconstruction
cases produce stable device-level conclusions, while small strategy microbenchmarks
need explicit warm-up and repeated statistics before drawing conclusions about
Reduce versus vmap-all on GPU.