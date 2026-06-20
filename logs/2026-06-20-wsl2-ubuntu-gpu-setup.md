# 2026-06-20 vbeam: WSL2 Ubuntu GPU setup notes

## Context

After finishing the CPU/JAX benchmark stage, the next goal is to try vbeam with
JAX on a local NVIDIA GPU before moving to a Linux server.

The local GPU machine has:

```text
GPU: NVIDIA GeForce RTX 4060
VRAM: 8188 MiB
Windows NVIDIA driver: 591.86
Windows nvidia-smi reported CUDA version: 13.1
CPU: 13th Gen Intel Core i7-13700KF
```

The CPU/JAX vbeam environment remains on Windows:

```text
C:\Users\Admin\.conda\envs\vbeam_jax_cpu
```

The GPU/JAX environment will be separate and built inside WSL2 Ubuntu.

## Why WSL2 is needed

JAX's official installation documentation currently lists Windows x86_64 NVIDIA
GPU support as `no`, while Windows WSL2 x86_64 NVIDIA GPU support is
`experimental`. The recommended NVIDIA GPU installation path is the Linux CUDA
pip wheel, for example:

```bash
pip install -U "jax[cuda13]"
```

The important distinction is:

```text
Windows native Python:
    good for CPU/JAX in the existing conda environment

WSL2 Ubuntu Python:
    used for Linux JAX CUDA wheels and GPU execution
```

## WSL2 installation location

To avoid C drive pressure, Ubuntu was installed to:

```text
G:\WSL\Ubuntu
```

The first `wsl --install -d Ubuntu --location G:\WSL\Ubuntu` call installed WSL
2.7.8 and enabled the `VirtualMachinePlatform` Windows optional component. After
reboot, the same command was run again and deployed Ubuntu itself.

Current relevant files in the installation directory:

```text
G:\WSL\Ubuntu\ext4.vhdx
G:\WSL\Ubuntu\shortcut.ico
```

`ext4.vhdx` is the Ubuntu Linux virtual disk. It contains the Linux filesystem
such as `/home/scatter`, `/usr`, `/etc`, and installed Linux packages. Windows
Explorer only sees it as one virtual disk file, while WSL mounts it as Ubuntu's
root filesystem.

The Start Menu Ubuntu shortcut under:

```text
C:\Users\Admin\AppData\Roaming\Microsoft\Windows\Start Menu
```

is only a small launcher shortcut, not the Ubuntu system itself.

## Linux entry in Windows Explorer

Windows Explorer shows a `Linux` entry with something like:

```text
Linux
  localhost
    Ubuntu
```

This is a special Explorer namespace for browsing WSL files, equivalent to paths
such as:

```text
\\wsl.localhost\Ubuntu
```

It is not a normal application and is therefore not found by searching for
"linux" as an app.

## Ubuntu terminal basics

The Ubuntu terminal prompt looks like:

```text
scatter@DESKTOP-PTP37C2:/mnt/g/Projects_Ultrasound/2_magnusdk_vbeam/vbeam$
```

Meaning:

```text
scatter: Linux username
DESKTOP-PTP37C2: host name
/mnt/g/...: current Linux path
$: normal user shell
```

Windows drives are mounted in Ubuntu under `/mnt`:

```text
C:\ -> /mnt/c
G:\ -> /mnt/g
G:\Projects_Ultrasound -> /mnt/g/Projects_Ultrasound
```

Ubuntu's own Linux home directory is:

```text
/home/scatter
```

For Python virtual environments, `/home/scatter` is preferred over `/mnt/g`
because Linux filesystem IO is faster and avoids Windows/WSL path translation.

## Verified WSL2 state

Inside Ubuntu:

```bash
lsb_release -a
```

reported:

```text
Ubuntu 26.04 LTS
Codename: resolute
```

```bash
uname -a
```

reported a Microsoft WSL2 kernel:

```text
6.18.33.1-microsoft-standard-WSL2 x86_64 GNU/Linux
```

```bash
nvidia-smi
```

successfully detected:

```text
NVIDIA GeForce RTX 4060
Driver Version: 591.86
CUDA Version: 13.1
Memory: 8188 MiB
```

This confirms that WSL2 can see the NVIDIA GPU through the Windows host driver.
No Linux NVIDIA driver should be installed inside WSL.

## Python version note

Ubuntu 26.04 currently reports:

```text
Python 3.14.4
```

This is too new for a conservative scientific Python/JAX setup. For vbeam GPU
experiments, the safer target is Python 3.11 or 3.12. The planned environment is:

```text
/home/scatter/venvs/vbeam-jax-gpu
```

The vbeam project declares:

```text
requires-python >= 3.9
dependencies: numpy, scipy, spekk >= 1.0.6, pyuff_ustb >= 2.0.7
optional backends: jax, tensorflow
```

The next step is to install or provide Python 3.11 inside Ubuntu, create the venv
under `/home/scatter/venvs/vbeam-jax-gpu`, install vbeam dependencies, and then
install JAX with CUDA 13 support.

## Planned command locations

Ubuntu terminal commands should be run from:

```bash
cd /home/scatter
```

or simply:

```bash
cd ~
```

The Windows vbeam project can be reached from Ubuntu at:

```bash
cd /mnt/g/Projects_Ultrasound/2_magnusdk_vbeam/vbeam
```

For the virtual environment itself, use the Linux filesystem:

```text
/home/scatter/venvs/vbeam-jax-gpu
```