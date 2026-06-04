# 2026-06-04 - vbeam speed of sound, interpolation, and fastmath

## 1. SpeedOfSound interface

Today I studied vbeam's speed-of-sound abstraction.

The core interface is in:

```text
vbeam/core/speed_of_sound.py
```

It defines:

```python
class SpeedOfSound(ABC):
    def average(
        self,
        sender_position,
        point_position,
        receiver_position,
    ) -> float:
        ...
```

Inside `signal_for_point()`, vbeam checks whether `speed_of_sound` is a scalar or a
`SpeedOfSound` object:

```python
if isinstance(speed_of_sound, SpeedOfSound):
    speed_of_sound = speed_of_sound.average(
        sender.position, point_position, receiver.position
    )

delay = (tx_distance + rx_distance) / speed_of_sound - wave_data.t0
```

So the current interface returns an effective average speed of sound, not the travel
time directly.

## 2. HeterogeneousSpeedOfSound

The main implementation is in:

```text
vbeam/speed_of_sound/__init__.py
```

`HeterogeneousSpeedOfSound.average()` samples the speed-of-sound map along two
straight segments:

```text
sender -> point
point  -> receiver
```

Each segment is sampled using `average_between_two_points()`, which interpolates the
speed map at uniformly spaced samples along the line:

```python
FastInterpLinspace.interp2d(...)
```

The two segment averages are then combined by distance weighting:

```python
return (average1 * distance1 + average2 * distance2) / total_distance
```

This is most natural for STAI, where both transmit and receive paths are explicitly
element-to-point straight paths.

The code itself warns that this class is probably only appropriate for synthetic
transmit aperture imaging, and not for more complex plane-wave, diverging, or focused
setups.

## 3. Speed-of-sound limitation and future TODO

For transcranial imaging and distortion correction, this is a useful extension point
but not yet a rigorous propagation model.

Current behavior:

```text
sample c(x, z) along straight paths
compute an arithmetic-type effective average speed
use delay = total_distance / c_eff
```

A more physically rigorous heterogeneous-medium delay would be closer to:

```text
t = integral ds / c(s)
```

and in a strongly heterogeneous medium may also require:

```text
ray bending
eikonal travel-time fields
CT-derived skull maps
separate transmit and receive travel-time models
```

For future compound plane-wave and transcranial work, the interface may need to evolve
from:

```text
average(sender, point, receiver) -> effective speed
```

to something closer to:

```text
travel_time(transmit_model, point, receiver) -> seconds
```

or a deeper wavefront/delay-layer redesign.

For now, this is recorded as a future improvement. The current goal remains to finish
understanding the project and get the code running.

## 4. Interpolation interface

The interpolation interface is in:

```text
vbeam/core/interpolation.py
```

The abstract interface is:

```python
class InterpolationSpace1D(ABC):
    def __call__(self, x, fp):
        ...
```

In beamforming terms:

```text
x:
    delay time

fp:
    discrete RF/IQ signal for one transmit-receiver pair

interpolate(delay, signal):
    delayed signal sample
```

Inside `signal_for_point()`:

```python
signal = interpolate(delay, signal)
```

This is where geometric delay becomes a sampled RF/IQ value.

## 5. FastInterpLinspace

The main interpolation implementation is:

```text
vbeam/interpolation/fast_linear.py
```

`FastInterpLinspace` assumes a uniformly sampled time axis:

```python
FastInterpLinspace(
    min=initial_time,
    d=1 / sampling_frequency,
    n=N_samples,
)
```

The PyUFF importer constructs it from channel data:

```python
t_axis_interpolate = FastInterpLinspace(
    min=float(channel_data.initial_time),
    d=1 / float(channel_data.sampling_frequency),
    n=int(channel_data.N_samples),
)
```

The core 1D interpolation converts time into a pseudo sample index:

```python
pseudo_index = (x - self.min) / self.d
i_floor = floor(pseudo_index)
di = pseudo_index - i_floor
```

Then it performs linear interpolation:

```python
v = fp[i1] * (1 - di) + fp[i2] * di
```

In conventional beamforming notation:

```text
sample_index = (delay - initial_time) * fs
sample = (1 - frac) * signal[floor_index]
       + frac       * signal[floor_index + 1]
```

Out-of-bounds delays return zero by default:

```python
v = np.where(bounds_flag == -1, left, v)
v = np.where(bounds_flag == 1, right, v)
```

The implementation uses vectorized array operations (`where`, `clip`, indexing), which
makes it suitable for JAX tracing, `jit`, and `vmap`.

## 6. interp2d

`FastInterpLinspace.interp2d()` is also used by:

```text
scan conversion
speed-of-sound map sampling
```

It performs bilinear interpolation over two uniformly spaced axes. It also supports:

```text
edge_handling="Value":
    return a default value outside bounds

edge_handling="Nearest":
    clip to nearest valid value
```

This keeps scan conversion and speed-map lookup consistent with the same backend-aware
array style.

## 7. fastmath backend abstraction

The central backend abstraction is:

```text
vbeam/fastmath/__init__.py
```

vbeam code usually imports:

```python
from vbeam.fastmath import numpy as np
```

This `np` is not a fixed NumPy module. It is a proxy backend:

```python
backend_manager = BackendManager()
numpy = proxy_backend(backend_manager)
```

Calls such as:

```python
np.array
np.sin
np.where
np.vmap
np.jit
np.scan
np.reduce
```

are forwarded to the currently active backend.

The included backend priority is:

```python
backend_priority = ["jax", "numpy"]
```

So if JAX is installed, vbeam prefers the JAX backend; otherwise it falls back to NumPy.

The backend can also be selected manually:

```python
from vbeam.fastmath import backend_manager

backend_manager.active_backend = "jax"
```

or temporarily:

```python
with backend_manager.using_backend("numpy"):
    ...
```

## 8. JAX backend vs NumPy backend

The JAX backend maps vbeam operations to real JAX primitives:

```python
def jit(self, fun, ...):
    return jax.jit(fun, ...)

def vmap(self, fun, in_axes, out_axes=0):
    return jax.vmap(fun, in_axes=in_axes, out_axes=out_axes)

def scan(self, f, init, xs):
    return jax.lax.scan(f, init, xs)
```

The NumPy backend is mostly for debugging and simple execution:

```python
def jit(self, fun, ...):
    return fun

def vmap(self, fun, in_axes, out_axes=0):
    for i in range(v_ax_size):
        ...
```

So:

```text
JAX backend:
    real jit/vmap/scan
    can compile and run on GPU if available

NumPy backend:
    jit is no-op
    vmap is simulated with Python loops
    useful for debugging and conceptual checks
```

This is how vbeam keeps one code path for both CPU debugging and JAX acceleration.

## 9. traceable_dataclass

`traceable_dataclass` is defined in:

```text
vbeam/fastmath/traceable.py
```

Many vbeam domain objects are decorated with it:

```python
@traceable_dataclass(("position", "theta", "phi", ...))
class ElementGeometry:
    ...

@traceable_dataclass(("source", "azimuth", "elevation", "t0"))
class WaveData:
    ...

@traceable_dataclass(data_fields=("min", "d", "n"))
class FastInterpLinspace:
    ...
```

This is necessary because vbeam passes structured objects, not just arrays, into
`signal_for_point()`:

```text
ElementGeometry
WaveData
FastInterpLinspace
PlaneWavefront
TxRxApodization
SpeedOfSound
Scan
```

JAX needs to know how to flatten and reconstruct those objects during `jit` and `vmap`.

The decorator records:

```text
data_fields:
    fields that participate in tracing/vectorization

aux_fields:
    static or auxiliary fields that should persist but not behave like dynamic arrays
```

In the JAX backend, these classes are registered as pytrees:

```python
jax.tree_util.register_pytree_node(cls, flatten_fn, unflatten_fn)
```

This lets JAX understand objects like:

```python
WaveData(
    azimuth=array([...]),
    t0=array([...])
)
```

as structured containers of traceable fields.

## 10. Link to Spec and ForAll

`traceable_dataclass` also gives objects a spekk-style tree definition:

```python
__spekk_treedef_keys__
__spekk_treedef_get__
__spekk_treedef_create__
```

This connects the object layer to `Spec` and `ForAll`.

The combined picture is:

```text
Spec:
    knows semantic dimensions such as points, receivers, transmits

spekk treedef:
    knows how to access and reconstruct structured objects by fields

traceable_dataclass:
    marks those fields for both spekk and backend tracing

JAX pytree:
    lets jit/vmap handle structured objects
```

Without this mechanism, vbeam would likely need to flatten every domain object into a
long argument list of arrays:

```python
sender_position, sender_theta, receiver_position, wave_azimuth, wave_t0, ...
```

Instead, it keeps meaningful domain objects:

```python
sender: ElementGeometry
wave_data: WaveData
interpolate: FastInterpLinspace
apodization: TxRxApodization
```

while still making them compatible with `jit` and `vmap`.

## Current takeaway

This part connects several earlier ideas:

```text
wavefront + speed_of_sound:
    produce delay

FastInterpLinspace:
    samples RF/IQ data at that delay

fastmath:
    routes array operations to JAX or NumPy

traceable_dataclass:
    lets structured vbeam objects enter JAX tracing

Spec / ForAll:
    maps semantic dimensions to vmap axes
```

This is the engineering bridge from a pixel-based scalar kernel to JAX-vectorized
beamforming.

## Next step

Continue with:

```text
vbeam/util/transformations.py
```

The goal is to inspect how `ForAll`, `Apply`, and `Reduce` are implemented and how
they call `np.vmap`, `np.reduce`, and spekk `Spec` machinery.
