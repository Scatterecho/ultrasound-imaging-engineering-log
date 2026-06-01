# 2026-06-01 — vbeam 环境配置与 Spec/ForAll 学习

## What I worked on

- 开始从 PyBF 阶段切换到 `magnusdk/vbeam` 项目。
- 配置 Windows 本机 CPU/JAX 学习环境。
- 阅读并梳理 `signal_for_point()` 的角色。
- 学习 vbeam 如何用 `Spec` 和 `ForAll` 将 pixel-based beamforming 映射到 `vmap`。

## Environment setup

原计划使用：

```text
D:\Anaconda\envs\vbeam_env
```

但该环境继承了 Anaconda 安装目录权限，`Lib\site-packages` 对普通用户不可写，导致 `pip install` 自动退回到用户级 site-packages。为了避免包混入全局用户 Python，改为创建用户目录下的新环境：

```text
C:\Users\Admin\.conda\envs\vbeam_jax_cpu
```

安装方式：

```powershell
conda create --prefix C:\Users\Admin\.conda\envs\vbeam_jax_cpu python=3.10 -y
C:\Users\Admin\.conda\envs\vbeam_jax_cpu\python.exe -s -m pip install --no-user --force-reinstall -e ".[test]" jax matplotlib ipykernel notebook
```

验证结果：

```text
Python: C:\Users\Admin\.conda\envs\vbeam_jax_cpu\python.exe
vbeam: 1.0.10, editable from local repo
jax: 0.6.2
jaxlib: 0.6.2
numpy: 2.2.6
scipy: 1.15.3
spekk: 1.0.9
pyuff_ustb: 3.0.0
JAX devices: [CpuDevice(id=0)]
```

Focused tests passed with:

```powershell
python -m pytest tests\interpolation\test_interp1d.py tests\util\test_vmap.py tests\fastmath\test_backend_implementations.py tests\wave_data\test_wave_data.py --import-mode=importlib -q
```

Result:

```text
26 passed
```

Note: `--import-mode=importlib` avoids pytest import-name collision between the local `tests\fastmath` directory and `spekk`'s optional `import fastmath` logic.

## signal_for_point

`signal_for_point()` is the scalar kernel at the center of vbeam. It computes one contribution for:

```text
one transmit / sender
one receiver
one point
one receiver signal
```

The core logic is:

```text
tx_distance = transmitted_wavefront(...)
rx_distance = reflected_wavefront(...)
delay = (tx_distance + rx_distance) / speed_of_sound - t0
sample = interpolate(delay, signal)
sample = phase_correct(sample, delay, modulation_frequency) if IQ
output = sample * apodization(...)
```

This is not the whole image. The full DAS image is built by repeating this kernel over points, receivers, transmits, and then summing the relevant dimensions.

## Spec

`Spec` gives semantic names to array axes. Instead of relying only on:

```text
signal.shape = (n_transmits, n_receivers, n_samples)
```

vbeam writes:

```python
Spec({
    "signal": ["transmits", "receivers", "signal_time"],
    "receiver": ["receivers"],
    "point_position": ["points"],
    "wave_data": ["transmits"],
})
```

This turns axis programming into dimension-name programming. The code can say `receivers` or `transmits` directly instead of hard-coding axis indices.

## ForAll

My current understanding:

> `Spec` gives signal/data dimensions field-like semantic names. `ForAll` indexes those semantic dimensions and slices the corresponding inputs, so that the backend can apply `vmap` over that dimension.

For example, if:

```python
Spec({
    "signal": ["receivers", "signal_time"],
    "receiver_gain": ["receivers"],
})
```

then:

```python
ForAll("receivers")
```

means:

```text
signal: slice along axis 0
receiver_gain: slice along axis 0
inputs without receivers: do not slice
```

This corresponds to writing `jax.vmap(..., in_axes=...)`, but the `in_axes` are inferred from named dimensions.

## Key takeaway

vbeam's important engineering move is not changing the DAS physics. The DAS kernel is familiar. The key move is:

```text
single-point kernel
plus named dimensions
plus composable transformations
plus JAX backend
```

This is what lets pixel-based beamforming map naturally to `vmap`, `jit`, and eventually GPU execution.

## Next step

Study how `get_das_beamformer(setup)` composes:

```text
signal_for_point
ForAll(points / receivers / transmits)
Apply(sum over receivers)
Reduce.Sum(transmits)
unflatten_points
postprocess
```

The next focus should be `Apply(np.sum, Axis("receivers"))` and `Reduce.Sum("transmits")`, especially the memory/performance tradeoff compared with fully materializing the datacube.
