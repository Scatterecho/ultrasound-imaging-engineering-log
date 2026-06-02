# 2026-06-02 - vbeam beamformer composition and apodization

## 1. DAS beamformer composition

The main entry point studied today is `vbeam/beamformers/__init__.py`.

`get_das_beamformer(setup)` is not a monolithic DAS loop. It composes a scalar pixel-based kernel with a sequence of transformations:

```text
signal_for_point
vectorize_over_datacube
sum_over_dimensions
Reduce.Sum("transmits") when appropriate
unflatten_points
optional apodization-overlap compensation
optional normalized decibels
optional scan conversion
```

The important point is that `signal_for_point()` remains a scalar contribution kernel, while `ForAll`, `Apply`, and `Reduce` decide how that kernel is expanded over semantic dimensions such as `points`, `receivers`, and `transmits`.

## 2. Coherence beamformer

`get_coherence_beamformer(setup)` is structurally close to DAS, but it must preserve the receiver dimension long enough to compute the coherence factor.

My current understanding:

```text
DAS:
    compute delayed samples
    sum over receivers/transmits

Coherence beamformer:
    compute delayed samples
    keep receivers
    reduce transmits when needed
    unflatten points
    Apply(coherence_factor, Axis("receivers"))
```

So the coherence beamformer still uses summation and the same scalar signal kernel. The difference is that it adds a receiver-axis coherence operation before the final image is produced.

## 3. Apply and Reduce

`Apply(np.sum, Axis(...))` means applying an operation to a named axis after the data over that axis has been materialized.

`Reduce.Sum("transmits")` is different in spirit. It performs accumulation over a named dimension without requiring the full `points x receivers x transmits` datacube to be materialized at once.

This matters for beamforming because the full datacube can become very large:

```text
num_points x num_receivers x num_transmits
```

The default DAS composition keeps enough parallelism for points and receivers, but often reduces over transmits iteratively to control memory. This is an engineering tradeoff:

```text
more vmap/materialization:
    more parallelism
    higher memory pressure

more Reduce:
    lower memory pressure
    less exposed parallelism
```

This is one of the places where vbeam's abstraction is useful: the beamforming math stays the same, while the execution strategy can change.

## 4. Flattened points and unflatten_points

vbeam scan objects usually provide image points in flattened form:

```text
points: [num_points, 3]
```

This makes `ForAll("points")` simple because there is only one point axis to vectorize over.

After beamforming, `unflatten_points(setup)` reshapes the flat point axis back into the scan geometry:

```text
LinearScan:
    points -> width x height

SectorScan:
    points -> azimuth x depth
```

This also explains why `points` is often treated as a computational axis, while `width` and `height` are image axes that appear after reconstruction.

## 5. scan_convert

`scan_convert(setup)` is only active when the scan coordinate system is polar.

For a Cartesian or linear scan, scan conversion is effectively a no-op.

For a sector scan, vbeam first reconstructs the image on the native polar grid, then performs backward mapping from a Cartesian output grid into polar coordinates and interpolates the polar image.

So the pipeline is:

```text
beamform on native scan grid
unflatten points
if polar:
    interpolate polar image onto Cartesian display grid
```

This keeps the beamforming geometry and the display geometry separated.

## 6. Apodization overview

The files under `vbeam/apodization` are not simply classified by transmit type. They are better understood as small composable apodization components:

```text
NoApodization:
    always returns 1

PlaneWaveTransmitApodization:
    transmit aperture / plane-wave coverage

PlaneWaveReceiveApodization:
    receive F-number / receiver field of view

TxRxApodization:
    product of transmit and receive apodization

RTBApodization:
    focused or retrospective transmit beamforming coverage

MLAApodization:
    multiple-line-acquisition spatial restriction

Window:
    Hamming, Hanning, Tukey, Bartlett, Rectangular, etc.

combine_apodizations:
    product/custom composition of arbitrary apodization functions
```

The design pattern is important: apodization is represented as a callable object with the same local arguments used by `signal_for_point()`:

```python
apodization(sender, point_position, receiver, wave_data)
```

That means apodization naturally participates in the same `ForAll`/`vmap` expansion as delays and interpolation.

## 7. PlaneWaveReceiveApodization

`PlaneWaveReceiveApodization` corresponds closely to the receive aperture or receiver field-of-view restriction used in conventional DAS.

The core idea is:

```text
dist = point_position - receiver.position
tan(theta) = x_dist / z_dist
ratio = abs(f_number * tan(theta))
weight = window(ratio)
```

With a rectangular window, the receiver contributes only when:

```text
abs(tan(theta)) <= 0.5 / f_number
```

This is equivalent to the familiar dynamic receive aperture relation:

```text
half aperture width ~= z / (2F)
```

With a Hamming or Hanning window, the aperture is not a hard cutoff; elements near the edge are smoothly tapered.

In the PyUFF importer, the default plane-wave receive apodization is:

```python
PlaneWaveReceiveApodization(Hamming(), 1.7)
```

## 8. PlaneWaveTransmitApodization

`PlaneWaveTransmitApodization` answers a different question:

```text
For this plane-wave transmit, is this pixel covered by the effective transmit aperture?
```

The implementation computes the plane-wave direction from azimuth/elevation, then projects the pixel backward along the plane-wave direction to the aperture plane:

```text
x_projected = x - direction_x * z / direction_z
```

For a 2D aperture, it does the same in elevation:

```text
y_projected = y - direction_y * z / direction_z
```

Interpretation:

```text
normal incidence:
    x_projected = x

angled plane wave:
    x_projected = x - z * tan(alpha)
```

The projected position is then normalized by the array width and passed through a window. With a rectangular window, pixels whose back-projected position lies outside the aperture receive zero transmit weight. With a smooth window, the transmit aperture is tapered.

This is transmit coverage, not receive F-number. In the default plane-wave PyUFF setup, vbeam combines transmit and receive apodization as:

```python
TxRxApodization(
    transmit=PlaneWaveTransmitApodization(array_bounds),
    receive=PlaneWaveReceiveApodization(Hamming(), 1.7),
)
```

## Current takeaway

The recurring theme is that vbeam does not hide DAS inside a single opaque loop. It factors the problem into:

```text
local scalar physics:
    delay, interpolation, phase correction, apodization

semantic dimensions:
    points, receivers, transmits, frames

execution transformations:
    ForAll, Apply, Reduce, unflatten, scan_convert
```

This is the bridge from pixel-based beamforming to JAX/vmap/GPU-style execution.

## Next step

Continue from apodization into the wavefront model:

```text
PlaneWavefront
ReflectedWavefront
UnifiedWavefront
MultipleTransmitDistances
```

The goal is to understand how vbeam represents transmit and receive propagation geometry as interchangeable callable components.
