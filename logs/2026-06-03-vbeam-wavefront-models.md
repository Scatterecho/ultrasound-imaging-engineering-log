# 2026-06-03 - vbeam wavefront models

## 1. Wavefront abstraction

Today I continued studying how vbeam represents transmit and receive propagation.

The key files are:

```text
vbeam/core/wavefront.py
vbeam/wavefront/plane.py
vbeam/wavefront/focused.py
vbeam/wavefront/unified.py
vbeam/wavefront/stai.py
vbeam/wavefront/refocus.py
```

vbeam separates propagation geometry into two callables:

```text
TransmittedWavefront:
    transmit propagation distance

ReflectedWavefront:
    receive propagation distance from point to receiver
```

Both return distances in meters. The conversion from distance to time remains inside
`signal_for_point()`:

```python
delay = (tx_distance + rx_distance) / speed_of_sound - wave_data.t0
```

This keeps wavefront geometry decoupled from speed of sound, interpolation, phase
correction, apodization, and vectorization.

## 2. PlaneWavefront

`PlaneWavefront` computes the projection of `point_position - sender.position` onto
the plane-wave propagation direction:

```python
tx_distance = (
    x * sin(azimuth) * cos(elevation)
    + y * sin(elevation)
    + z * cos(azimuth) * cos(elevation)
)
```

In 2D, this is essentially:

```text
tx_distance = x * sin(alpha) + z * cos(alpha)
```

For normal incidence, it reduces to `z`. For an angled plane wave, the lateral
coordinate also changes the transmit arrival time.

## 3. FocusedSphericalWavefront

`FocusedSphericalWavefront` uses `wave_data.source` to represent a focus or virtual
source:

```python
sender_source_dist = norm(sender.position - wave_data.source)
source_point_dist = norm(wave_data.source - point_position)

return sender_source_dist * sign(source_z - sender_z) \
     - source_point_dist * sign(source_z - point_z)
```

The important interpretation is that this function returns the transmit arrival
distance field for the whole modeled wavefront, not a physical one-element-to-pixel
path.

For a virtual source behind the array in 2D:

```text
pixel p = (x1, y1)
virtual source S = (x0, -y0)
reference sender r = (xr, 0)
```

the effective transmit distance is:

```text
tx_distance(p)
= |S - p| - |S - r|
= sqrt((x1 - x0)^2 + (y1 + y0)^2)
 - sqrt((xr - x0)^2 + y0^2)
```

If the reference sender is the array center `r = (0, 0)`:

```text
tx_distance(p)
= sqrt((x1 - x0)^2 + (y1 + y0)^2)
 - sqrt(x0^2 + y0^2)
```

If the virtual source is directly behind the center, `x0 = 0`:

```text
tx_distance(p) = sqrt(x1^2 + (y1 + y0)^2) - y0
```

This is the virtual spherical wave arrival distance relative to the chosen transmit
reference point.

## 4. Reference sender and wave_data.t0

`FocusedSphericalWavefront` does not decide which sender is the reference. It uses the
`sender` supplied by `SignalForPointSetup`.

In the PyUFF importer, for ordinary plane-wave or focused/virtual-source data, vbeam
sets:

```python
sender = ElementGeometry(np.array([0.0, 0.0, 0.0], dtype="float32"), 0.0, 0.0)
```

So the default reference sender is the coordinate origin, usually the probe or array
center in the dataset coordinate system.

`WaveData.t0` is the paired time reference. Its docstring says it is:

```text
The time at which the transmitted wave passed through the "sender" element
```

Therefore, transmit timing is defined by:

```text
sender.position:
    spatial reference point

wave_data.t0:
    time when the modeled transmit wave passes that reference point
```

Then `signal_for_point()` uses:

```python
delay = (tx_distance + rx_distance) / speed_of_sound - wave_data.t0
```

For STAI data, vbeam changes the sender model: `sender` becomes the receiver element
geometry and the spec marks sender as varying over `transmits`.

## 5. FocusedHybridWavefront and FocusedBlendedWavefront

`FocusedHybridWavefront` combines spherical and plane wavefront models:

```text
most points:
    FocusedSphericalWavefront

near the focus depth:
    PlaneWavefront
```

The switch is controlled by `pw_margin`.

`FocusedBlendedWavefront` uses a smoother strategy:

```text
distance = spherical_distance * w + plane_distance * (1 - w)
```

where the weight depends on normalized distance from the source.

Both models preserve the same external interface:

```python
transmitted_wavefront(sender, point_position, wave_data) -> tx_distance
```

## 6. UnifiedWavefront

`UnifiedWavefront` is the default model used by `import_pyuff()` for spherical/focused
wavefronts:

```python
transmitted_wavefront = UnifiedWavefront(array_bounds)
```

It extends the base focused spherical model by considering finite array aperture
boundaries.

The geometry is built from two lines:

```python
line_left = Line.passing_through(array_left, wave_data.source)
line_right = Line.passing_through(array_right, wave_data.source)
```

These lines define the transmit region as seen from the virtual source.

For points inside the central transmit region, `UnifiedWavefront` uses the base
wavefront directly:

```python
self.base_wavefront(sender, point_position, wave_data)
```

For points outside or near boundary regions, it constructs an interpolation line,
intersects it with the two aperture boundary lines, and interpolates the base
wavefront distances evaluated at the two boundary intersection points:

```python
A = line_left.intersection(intersection_line)
B = line_right.intersection(intersection_line)

R1 = self.base_wavefront(sender, A, wave_data)
R2 = self.base_wavefront(sender, B, wave_data)

interpolated_distance = R1 * weight_A + R2 * weight_B
```

So the practical interpretation is:

```text
inside transmit region:
    use focused spherical wavefront distance

outside / boundary region:
    use interpolated extension from aperture-boundary distances
```

The kernel still sees only a callable that returns `tx_distance`.

## 7. STAIWavefront and REFoCUSWavefront

`STAIWavefront` models synthetic transmit aperture imaging:

```python
return distance(sender.position, point_position)
```

In this case, `sender` is truly a transmitting element.

`REFoCUSWavefront` models a time-domain REFoCUS interpretation:

```python
return distance(sender.position, point_position) + focusing_compensation
```

Here, a complex focused transmit is reinterpreted as element-level spherical
emissions, with compensation for the original transmit timing.

## 8. MultipleTransmitDistances

Most transmitted wavefront models return a single distance:

```text
tx_distance: float
```

`MultipleTransmitDistances` allows a model to return several candidate transmit
distances for the same transmit/receiver/pixel combination:

```python
MultipleTransmitDistances(
    values=np.array([d1, d2, d3]),
    aggregate_samples=...
)
```

Because it overloads arithmetic operators:

```python
def __add__(self, other):
    return self.values + other

def __truediv__(self, other):
    return self.values / other
```

this expression inside `signal_for_point()` works naturally:

```python
delay = (tx_distance + rx_distance) / speed_of_sound - wave_data.t0
```

The interpolation then returns multiple delayed samples, and vbeam aggregates them:

```python
if isinstance(tx_distance, MultipleTransmitDistances):
    signal = tx_distance.aggregate_samples(signal)
```

A simple use case is local delay smoothing:

```python
distances = np.array([d - 0.1e-3, d, d + 0.1e-3])
weights = np.array([0.2, 0.6, 0.2])

return MultipleTransmitDistances(
    distances,
    lambda samples: np.sum(samples * weights)
)
```

This extension mechanism allows advanced wavefront models to sample multiple delays
without changing `signal_for_point()`.

## Current takeaway

The wavefront section reinforces the central vbeam design:

```text
wavefront object:
    defines transmit/receive propagation distance

signal_for_point:
    converts distance to delay and samples the signal

apodization:
    weights each local contribution

ForAll / Apply / Reduce:
    map the scalar computation over semantic dimensions
```

Changing the transmit geometry does not require rewriting the beamforming kernel.

## Next step

Continue to the speed-of-sound abstraction:

```text
constant speed of sound
SpeedOfSound interface
heterogeneous / average speed-of-sound models
```

This is especially relevant for distortion correction and transcranial imaging.
