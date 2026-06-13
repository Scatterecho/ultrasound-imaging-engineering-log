# 2026-06-06 vbeam: running the official plane-wave notebook locally

## Context

After building a command-line PICMUS plane-wave example for vbeam, I tested the
official `docs/examples/plane_wave_dataset.ipynb` workflow in a local Windows
CPU/JAX setup.

The goal was not only to run a demo, but to understand how a notebook should be
opened, which Python kernel it uses, and why PyCharm/Jupyter warnings may appear
even when the beamformer itself is working.

## PyCharm run result

Running `friends/run_picmus_plane_wave.py` from PyCharm completed successfully:

```text
Result shape: (96, 128)
Result dtype: float32
JAX devices: [CpuDevice(id=0)]
Saved image: friends/outputs/picmus_plane_wave_das.png
Exit code: 0
```

The red messages were warnings rather than fatal errors:

- `RuntimeWarning: invalid value encountered in multiply` from `pyuff_ustb`
  appears while reading PICMUS plane-wave metadata.
- `point_position will be overwritten by the scan` is expected because the
  imported UFF scan defines the vbeam scan grid.

One real environment issue was found: PyCharm originally loaded `pyuff_ustb`
from the user-level package directory:

```text
C:\Users\Admin\AppData\Roaming\Python\Python310\site-packages
```

The intended package path is inside the conda environment:

```text
C:\Users\Admin\.conda\envs\vbeam_jax_cpu\lib\site-packages
```

The practical fix is to run with interpreter option `-s` or set
`PYTHONNOUSERSITE=1`, so user-site packages cannot leak into the conda
environment.

## How to run `.ipynb` files

For this vbeam learning stage, the recommended split is:

- Use PyCharm for `.py` scripts.
- Use Jupyter Lab for `.ipynb` notebooks.

PyCharm can run notebooks in some editions/configurations, but Jupyter Lab makes
the kernel, cell outputs, plots, and execution order more explicit. This is useful
when learning vbeam's data flow step by step.

## Jupyter root cause analysis

Typing `jupyter lab` directly used base Anaconda:

```text
D:\Anaconda\Scripts\jupyter.exe
```

This caused two problems:

1. Jupyter tried to use base/global paths rather than the `vbeam_jax_cpu`
   environment.
2. Global Jupyter/IPython tried to write under user directories such as
   `C:\Users\Admin\.jupyter` and `C:\Users\Admin\.ipython`, which produced
   permission errors.

The stable approach is to launch Jupyter through the exact environment Python:

```powershell
cd G:\Projects_Ultrasound\2_magnusdk_vbeam\vbeam

$env:PYTHONNOUSERSITE="1"
$env:JUPYTER_CONFIG_DIR="$PWD\.jupyter_config"
$env:JUPYTER_DATA_DIR="$PWD\.jupyter_data"
$env:JUPYTER_RUNTIME_DIR="$PWD\.jupyter_runtime"
$env:IPYTHONDIR="$PWD\.ipython"
$env:JUPYTER_PATH="C:\Users\Admin\.conda\envs\vbeam_jax_cpu\share\jupyter"

C:\Users\Admin\.conda\envs\vbeam_jax_cpu\python.exe -s -m jupyter lab --no-browser --port=8899 --ServerApp.root_dir="$PWD"
```

A helper script was also added in the vbeam project:

```text
friends/launch_jupyter_vbeam.ps1
```

## Local version of the official notebook

The official notebook starts with:

```python
!pip install vbeam
```

This is suitable for Colab or a fresh environment, but not for an editable local
source checkout. Running it locally may install a different vbeam package from
PyPI and hide the source tree that is being studied.

Therefore, I created a local learning copy instead of modifying the original
notebook directly:

```text
friends/notebooks/plane_wave_dataset_local_cpu.ipynb
```

The local copy changes four things:

1. Skips `!pip install vbeam`.
2. Uses the `vbeam_jax_cpu` kernel explicitly.
3. Reads the already downloaded PICMUS UFF file from `friends/data`.
4. Resizes the scan to `96 x 128` so CPU/JAX execution is fast enough for
   interactive learning.

The executed notebook output was saved as:

```text
friends/notebooks/plane_wave_dataset_local_cpu_executed.ipynb
```

## Verified notebook output

The notebook was executed successfully with `nbconvert` using the
`vbeam_jax_cpu` kernel.

Key output:

```text
Python: C:\Users\Admin\.conda\envs\vbeam_jax_cpu\python.exe
JAX devices: [CpuDevice(id=0)]
Setup sizes: {'receivers': 128, 'points': 12288, 'signal_time': 1536, 'transmits': 75}
Setup spec: Spec({'signal': ['transmits', 'receivers', 'signal_time'], 'receiver': ['receivers'], 'point_position': ['points'], 'wave_data': ['transmits']})

Result shape: (96, 128)
Result dtype: float32
75 transmits DAS: 154 ms
1 transmit DAS: 4.82 ms
```

This confirms that the notebook version and script version are following the
same vbeam pipeline:

```text
UFF channel_data + scan
-> import_pyuff(...)
-> SignalForPointSetup
-> get_das_beamformer(setup)
-> jax.jit(...)
-> beamformer(**setup.data)
```

## Notes on `setup.size`

During notebook localization, I initially tried to print `setup.sizes`, which
failed because `SignalForPointSetup` exposes sizes through a method:

```python
setup.size()
setup.size("points")
setup.size(["transmits", "receivers", "points"])
```

This matches vbeam/spekk's semantic-dimension design: dimensions are queried by
field names such as `"points"`, `"receivers"`, and `"transmits"` rather than by
anonymous array axes.

## GUI control status

Computer Use / GUI control did not work in this session.

The first failure was a Windows permission issue:

```text
CreateProcessAsUserW failed: 5
```

After resetting the helper, the failure changed to a plugin/runtime packaging
issue:

```text
Package subpath ... is not defined by "exports" in @oai/sky/package.json
```

So the current blocker is not vbeam, PyCharm, or Jupyter itself. It is the Codex
Computer Use runtime on this machine. GUI control can be revisited later after
re-authorizing or restarting the Computer Use plugin/session.

## Practical takeaway

For stable local vbeam notebook work on this Windows machine:

1. Start Jupyter through `friends/launch_jupyter_vbeam.ps1`.
2. Use the `Python (vbeam_jax_cpu)` kernel.
3. Avoid running notebook cells that install vbeam from PyPI.
4. Keep downloaded UFF data and Jupyter runtime/config directories inside the
   project-local ignored paths.
5. Treat the PyUFF and scan overwrite messages as warnings unless the process
   exits nonzero or the result shape/image is wrong.
