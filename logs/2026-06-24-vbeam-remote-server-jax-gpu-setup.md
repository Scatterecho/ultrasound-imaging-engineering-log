# 2026-06-24 vbeam: remote Linux server JAX GPU setup

## Goal

Move from the local WSL2 RTX 4060 experiment to a shared Linux GPU server and
set up a safe, user-local vbeam/JAX GPU environment.

Because this is a shared server, the operating rules were:

```text
Do not use sudo.
Do not modify system Python or system CUDA.
Do not kill other users' processes.
Do not install packages globally.
Always check GPU occupancy before running JAX.
Use CUDA_VISIBLE_DEVICES to select an idle GPU.
Disable JAX preallocation with XLA_PYTHON_CLIENT_PREALLOCATE=false.
```

## Server snapshot

Login user and host:

```text
user: wjy23bs
host: fineserver
home: /home/wjy23bs
```

The server has two NVIDIA GPUs:

```text
GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
VRAM per GPU: 97887 MiB
NVIDIA driver: 580.65.06
CUDA version reported by nvidia-smi: 13.0
```

At one point GPU 0 was occupied by another user's training process:

```text
PID: 3309230
user: lqh25bs
command: python3 train.py --dataset Bubble --arch UNext ...
GPU memory: about 7280 MiB
```

GPU 1 was essentially idle, so the vbeam experiments used physical GPU 1:

```bash
export CUDA_VISIBLE_DEVICES=1
```

Inside JAX, this physical GPU 1 appears as:

```text
CudaDevice(id=0)
```

because `CUDA_VISIBLE_DEVICES` renumbers the visible devices within the process.

## Directory layout

A single project root was created under the user's `/data1` directory:

```text
/data1/wjy23bs/vbeam_project/
├── code/
├── venvs/
├── data/
├── outputs/
└── logs/
```

This keeps source code, virtual environments, data, and outputs together while
remaining completely inside the user's own storage area.

The creation command was:

```bash
cd /data1/wjy23bs
mkdir -p vbeam_project/code vbeam_project/venvs vbeam_project/data \
         vbeam_project/outputs vbeam_project/logs
```

## Conda and venv decision

`conda` and `mamba` were not available in the current shell:

```bash
which conda
which mamba
```

No usable conda executable was found under the checked locations. The system
Python was:

```text
/usr/bin/python3
Python 3.10.12
```

A first attempt to create a standard venv failed:

```bash
python3 -m venv venvs/vbeam-jax-gpu
```

with:

```text
ensurepip is not available
apt install python3.10-venv
```

Because this is a shared server, I did not use `sudo apt install`. Instead, I
installed `uv` in the user's home directory:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

The server default shell behaved like `sh`, not `bash`, so `source` was not
available. The compatible activation form is:

```bash
. "$HOME/.local/bin/env"
```

This makes the user-local `uv` executable available from any current directory:

```text
uv binary: /home/wjy23bs/.local/bin/uv
project root: /data1/wjy23bs/vbeam_project
```

So `uv` lives under `/home`, while the vbeam environment lives under `/data1`.
This is analogous to using a user-installed environment manager to create a
project-specific environment elsewhere.

## Python 3.10 attempt and why it was replaced

A Python 3.10 venv was first created under:

```text
/data1/wjy23bs/vbeam_project/venvs/vbeam-jax-gpu
```

but:

```bash
python -m pip install -U "jax[cuda13]"
```

resolved to:

```text
jax 0.6.2
jaxlib 0.6.2
```

and emitted:

```text
WARNING: jax 0.6.2 does not provide the extra 'cuda13'
```

The minimal test then reported:

```text
devices: [CpuDevice(id=0)]
default backend: cpu
```

and:

```text
An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed.
```

Therefore the Python 3.10 environment was removed and recreated with Python 3.11.

## Final Python 3.11 JAX GPU environment

The final environment was created with:

```bash
cd /data1/wjy23bs/vbeam_project
. "$HOME/.local/bin/env"
export UV_LINK_MODE=copy
uv venv venvs/vbeam-jax-gpu --python 3.11
. venvs/vbeam-jax-gpu/bin/activate
```

A `uv venv` did not initially include pip, so pip was installed into the active
environment with:

```bash
uv pip install pip setuptools wheel
python -m pip --version
```

The resulting pip path was inside the project environment:

```text
/data1/wjy23bs/vbeam_project/venvs/vbeam-jax-gpu/lib/python3.11/site-packages/pip
```

Then JAX CUDA 13 support was installed:

```bash
export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python -m pip install -U "jax[cuda13]"
```

This installed:

```text
jax 0.10.2
jaxlib 0.10.2
jax-cuda13-plugin 0.10.2
jax-cuda13-pjrt 0.10.2
CUDA/cuDNN/NCCL related nvidia-* wheels
```

The complete venv size was about:

```text
4.1 GB
```

which is expected because the pip CUDA wheel path installs CUDA runtime libraries
inside the environment rather than relying on system CUDA libraries directly.

## JAX GPU smoke test

The minimal test was:

```bash
python - <<'PY'
import jax
import jax.numpy as jnp

print("jax:", jax.__version__)
print("jaxlib:", jax.lib.__version__)
print("devices:", jax.devices())
print("default backend:", jax.default_backend())

x = jnp.ones((2048, 2048), dtype=jnp.float32)
y = (x @ x).block_until_ready()
print("result:", y.shape, y.dtype, float(y[0, 0]))
PY
```

It reported:

```text
jax: 0.10.2
jaxlib: 0.10.2
devices: [CudaDevice(id=0)]
default backend: gpu
result: (2048, 2048) float32 2048.0
```

## vbeam source checkout

The local vbeam learning scripts were pushed to the fork:

```text
https://github.com/Scatterecho/vbeam.git
branch: codex/vbeam-learning-benchmarks
commit: e687353 Add vbeam learning benchmarks
```

On the server, the branch was cloned into:

```text
/data1/wjy23bs/vbeam_project/code/vbeam
```

with:

```bash
cd /data1/wjy23bs/vbeam_project/code
git clone -b codex/vbeam-learning-benchmarks https://github.com/Scatterecho/vbeam.git
cd vbeam
```

Then vbeam was installed in editable mode into the active JAX GPU environment:

```bash
. /data1/wjy23bs/vbeam_project/venvs/vbeam-jax-gpu/bin/activate
export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export UV_LINK_MODE=copy
python -m pip install -e ".[test]" matplotlib psutil
```

The import check reported:

```text
vbeam: /data1/wjy23bs/vbeam_project/code/vbeam/vbeam/__init__.py
jax: 0.10.2
devices: [CudaDevice(id=0)]
backend: gpu
```

## PICMUS data issue

The benchmark script attempted to download:

```text
http://www.ustb.no/datasets/PICMUS_carotid_cross.uff
```

but the server received:

```text
HTTP Error 404: Not Found
```

The existing cached file from the local machine was uploaded manually to the
server path expected by the vbeam script:

```text
/data1/wjy23bs/vbeam_project/code/vbeam/friends/data/www.ustb.no/datasets/PICMUS_carotid_cross.uff
```

Verified server-side size:

```text
76705680 bytes
```

After this, the benchmark script reused the local cached file and no longer tried
to download from the unavailable USTB URL.

## Current state

The remote Linux server now has a working, user-local, GPU-enabled vbeam/JAX
environment:

```text
project root: /data1/wjy23bs/vbeam_project
source code:  /data1/wjy23bs/vbeam_project/code/vbeam
venv:         /data1/wjy23bs/vbeam_project/venvs/vbeam-jax-gpu
JAX:          0.10.2
backend:      gpu
selected GPU: physical GPU 1 through CUDA_VISIBLE_DEVICES=1
```
