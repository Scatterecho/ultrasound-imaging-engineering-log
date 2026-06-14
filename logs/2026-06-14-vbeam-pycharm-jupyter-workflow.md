# 2026-06-14 vbeam: PyCharm/Jupyter workflow and first code reading pass

## Context

Today I finished turning the local vbeam learning environment into a usable
daily workflow:

- `.py` files can be run directly in PyCharm.
- `.ipynb` files can be run cell by cell in Jupyter Lab.
- Computer Use can now inspect/control PyCharm after fixing a local Codex runtime
  packaging issue.

The practical goal was to stop confusing environment problems with vbeam
problems, so that the following learning can focus on the actual beamforming
pipeline.

## Computer Use / PyCharm control

Computer Use initially failed with:

```text
Package subpath ... computer_use_client_base.js is not defined by "exports"
```

The target JS file existed, but `@oai/sky/package.json` did not export the
internal path required by the Computer Use plugin.

I backed up the local runtime package file and added the missing export entry:

```text
C:\Users\Admin\AppData\Local\OpenAI\Codex\runtimes\cua_node\...\node_modules\@oai\sky\package.json
```

After this, PyCharm was visible to Computer Use:

```text
D:\Pycharm\PyCharm 2025.1.3\bin\pycharm64.exe
window: vbeam - plane_wave_dataset_local_cpu.ipynb
```

This made it possible to inspect PyCharm state and trigger normal GUI actions
when needed.

## Running `run_picmus_plane_wave.py` in PyCharm

The script can now be run directly in PyCharm:

```text
friends/run_picmus_plane_wave.py
```

The important success indicators are:

```text
Result shape: (96, 128)
Result dtype: float32
Saved image: friends/outputs/picmus_plane_wave_das.png
Process finished with exit code 0
```

The red messages in PyCharm are warnings, not fatal errors:

```text
RuntimeWarning: invalid value encountered in multiply
UserWarning: point_position will be overwritten by the scan
```

Interpretation:

- The `RuntimeWarning` comes from `pyuff_ustb` reading PICMUS plane-wave metadata.
- The scan overwrite warning is expected because the imported UFF scan defines
  the vbeam scan grid.

The script also prints `pyuff_ustb.__file__` and `sys.executable` to verify that
the run uses the intended conda environment:

```text
C:\Users\Admin\.conda\envs\vbeam_jax_cpu\python.exe
C:\Users\Admin\.conda\envs\vbeam_jax_cpu\lib\site-packages\pyuff_ustb
```

## Why PyCharm managed Jupyter failed

PyCharm's managed notebook server failed with errors such as:

```text
PermissionError: [Errno 13] Permission denied:
C:\Users\Admin\AppData\Roaming\jupyter\runtime\jpserver-...-open.html
```

The PyCharm log showed that although the interpreter was
`vbeam_jax_cpu`, user-level Jupyter packages were still being loaded from:

```text
C:\Users\Admin\AppData\Roaming\Python\Python310\site-packages
```

This made PyCharm's managed Jupyter startup fragile. The safer workflow is to
start Jupyter Lab manually through a project script and then open the notebook in
the browser.

## Fixing the Jupyter Lab launch script

The launcher lives at:

```text
friends/launch_jupyter_vbeam.ps1
```

Several permission issues appeared and were fixed step by step.

First, `python -m jupyter lab` could route through Anaconda's Jupyter machinery.
The script was changed to call JupyterLab directly:

```powershell
C:\Users\Admin\.conda\envs\vbeam_jax_cpu\python.exe -s -m jupyterlab
```

This confirms the correct JupyterLab source:

```text
JupyterLab extension loaded from
C:\Users\Admin\.conda\envs\vbeam_jax_cpu\lib\site-packages\jupyterlab
```

Second, writing runtime files under the project directory failed:

```text
G:\Projects_Ultrasound\2_magnusdk_vbeam\vbeam\.jupyter_runtime
```

That directory had unusual ACLs and did not allow normal user writes.

Third, reusing a shared temp runtime also failed because a previous test created
`jupyter_cookie_secret` under a Codex sandbox identity.

The final fix was to use a fresh temp base for each PowerShell process:

```powershell
%TEMP%\vbeam_jupyter_$PID
```

and put all Jupyter state there:

```text
config/
data/
runtime/
ipython/
```

The script also accepts an optional port:

```powershell
.\friends\launch_jupyter_vbeam.ps1 -Port 8898
```

This is useful if `8899` is already occupied by an old server.

## Stable notebook workflow

The stable command is:

```powershell
cd G:\Projects_Ultrasound\2_magnusdk_vbeam\vbeam
.\friends\launch_jupyter_vbeam.ps1
```

Success looks like:

```text
Jupyter Server 2.19.0 is running at:
http://localhost:8899/lab?token=...
```

Keep the PowerShell window open; it is the Jupyter server process.

Then open the URL in a browser and run:

```text
friends/notebooks/plane_wave_dataset_local_cpu.ipynb
```

The notebook can now be run cell by cell.

## Recommended daily workflow

Use PyCharm for scripts:

```text
friends/run_picmus_plane_wave.py
```

Use browser Jupyter Lab for notebooks:

```text
friends/notebooks/plane_wave_dataset_local_cpu.ipynb
```

Avoid running the notebook through PyCharm's managed Jupyter server for now,
because that path still has environment/user-site risk.

## First learning pass: what the script and notebook do

`run_picmus_plane_wave.py` and `plane_wave_dataset_local_cpu.ipynb` execute the
same vbeam pipeline:

```text
UFF channel_data + scan
-> import_pyuff(...)
-> SignalForPointSetup
-> get_das_beamformer(setup)
-> jax.jit(...)
-> beamformer(**setup.data)
-> result image
```

The three most important lines are:

```python
setup = import_pyuff(channel_data, scan, frames=0)
beamformer = jax.jit(get_das_beamformer(setup))
result = beamformer(**setup.data).block_until_ready()
```

Interpretation:

- `import_pyuff(...)` converts USTB/PICMUS data into vbeam's semantic setup.
- `get_das_beamformer(setup)` builds a DAS beamformer function from geometry,
  scan, wavefront, apodization, and `Spec`.
- `jax.jit(...)` compiles the function for JAX execution.
- `beamformer(**setup.data)` feeds the structured setup fields into the compiled
  function.
- `.block_until_ready()` is needed for honest timing because JAX execution is
  asynchronous.

The important setup dimensions are:

```text
signal: ['transmits', 'receivers', 'signal_time']
receiver: ['receivers']
point_position: ['points']
wave_data: ['transmits']
```

This is where earlier `Spec / ForAll / Apply / Reduce` learning connects to a
real dataset: vbeam keeps dimension semantics explicit instead of relying on raw
axis positions.

## Next learning step

The next useful step is to inspect `setup` in the notebook:

```python
setup.size()
setup.spec
setup.data.keys()
type(setup.signal), setup.signal.shape
type(setup.receiver), setup.receiver.shape
type(setup.wave_data), setup.wave_data.shape
```

The goal is to connect each concrete array/object to the semantic dimensions
that vbeam later uses for vectorized beamforming.
