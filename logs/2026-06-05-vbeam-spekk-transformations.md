# 2026-06-05 - vbeam spekk and transformations

## 1. Why spekk exists

Today I revisited vbeam's transformation layer more slowly.

The key idea:

```text
spekk lets vbeam program by semantic dimension names instead of raw axis indices.
```

Without spekk, we would often need to write code like:

```python
jax.vmap(f, in_axes=(0, None, 1, None))
np.sum(result, axis=2)
```

This is fragile because we must remember where each dimension currently lives.

In ultrasound beamforming, the dimensions already have natural names:

```text
points
receivers
transmits
frames
signal_time
```

spekk lets vbeam write:

```python
ForAll("receivers")
Apply(np.sum, Axis("receivers"))
```

instead of hard-coding `axis=0`, `axis=1`, or `axis=2`.

## 2. Spec as an axis label table

A `Spec` is a table that names the axes of each input.

For example:

```python
Spec({
    "signal": ["transmits", "receivers", "signal_time"],
    "receiver": ["receivers"],
    "point_position": ["points"],
    "wave_data": ["transmits"],
})
```

means:

```text
signal:
    axis 0 = transmits
    axis 1 = receivers
    axis 2 = signal_time

receiver:
    axis 0 = receivers

point_position:
    axis 0 = points

wave_data:
    axis 0 = transmits
```

The data arrays are not changed. `Spec` just attaches semantic names to their axes.

## 3. index_for

The method:

```python
spec.index_for("receivers")
```

asks:

```text
For each input, where is the receivers axis?
```

For the vbeam-style spec above, it returns:

```python
{
    "signal": 1,
    "receiver": 0,
    "point_position": None,
    "wave_data": None,
}
```

Meaning:

```text
If vectorizing over receivers:
    signal should be sliced along axis 1
    receiver should be sliced along axis 0
    point_position should not be sliced
    wave_data should not be sliced
```

This is how spekk can generate `vmap` `in_axes` automatically.

## 4. ForAll

`ForAll("receivers")` means:

```text
Run the wrapped function for all receivers.
```

Conceptually:

```python
for r in receivers:
    output[r] = f(
        signal=signal[:, r, :],
        receiver=receiver[r],
        point_position=point_position,
        wave_data=wave_data,
    )
```

But instead of writing this loop manually, spekk uses `Spec` to decide which inputs
should be sliced and then calls backend `vmap`.

In vbeam, `ForAll` is connected to:

```python
from vbeam.fastmath import numpy as np

self.vmap_impl = np.vmap
```

So:

```text
JAX backend:
    ForAll -> jax.vmap

NumPy backend:
    ForAll -> Python-loop vmap simulation
```

## 5. Apply and Axis

`Apply` means:

```text
After the wrapped function returns a result, apply another function to that result.
```

For example:

```python
Apply(np.sum, Axis("receivers"))
```

means:

```text
Sum over the axis whose semantic name is receivers.
```

This is different from:

```python
np.sum(result, axis=1)
```

because spekk looks up the actual axis index from the output spec.

`Axis` also describes how the output dimensions change:

```python
Axis("receivers")
```

removes the `receivers` dimension, as expected after a sum.

```python
Axis("receivers", keep=True)
```

keeps the dimension.

```python
Axis("points", becomes=["width", "height"])
```

replaces a flattened `points` axis with image axes.

This is how vbeam can express:

```python
Apply(setup.scan.unflatten, Axis("points", becomes=["width", "height"]))
```

without manually tracking which axis currently contains points.

## 6. Reduce

`Reduce.Sum("transmits")` is similar in meaning to:

```python
ForAll("transmits")
Apply(np.sum, Axis("transmits"))
```

but the execution strategy is different.

`ForAll + Apply`:

```text
Compute all transmit results.
Materialize the transmit dimension.
Then sum.
```

`Reduce.Sum`:

```text
Compute one transmit result.
Accumulate into a running sum.
Move to the next transmit.
```

Conceptually:

```python
acc = 0
for t in transmits:
    acc += result_for_this_transmit
```

This can avoid materializing a large array such as:

```text
points x receivers x transmits
```

In vbeam, `Reduce` is connected to:

```python
self.reduce_impl = np.reduce
```

and with the JAX backend this is implemented using `jax.lax.scan`.

## 7. build(spec)

`compose(...)` only describes the transformation pipeline.

`build(spec)` gives that pipeline the axis-label information it needs.

A useful mental model:

```text
compose(...):
    What operations should happen?

build(spec):
    Given these input axis names, which actual axes should those operations use?
```

For example:

```python
compose(
    f,
    ForAll("receivers"),
    Apply(np.sum, Axis("receivers")),
).build(spec)
```

means:

```text
1. Use spec to find where receivers lives in each input.
2. Build a vmap over receivers.
3. Track the output spec after vmap.
4. Use the output spec to find where receivers lives in the result.
5. Sum over that axis.
6. Update the output spec by removing receivers.
```

## 8. Specced

`Specced` wraps a function and tells spekk what dimensions the function returns.

This is important when the function returns an array instead of a scalar.

Example:

```python
f = Specced(
    lambda signal, gain: signal * gain,
    lambda input_spec: input_spec["signal"],
)
```

The second argument says:

```text
The output dimensions of f are the same as the input dimensions of signal.
```

For vbeam's `signal_for_point()`, the returned value is normally a scalar delayed
sample for one point/receiver/transmit combination, so the surrounding transformations
can add dimensions such as points, receivers, and transmits.

## 9. vbeam DAS mental model

The vbeam DAS chain is equivalent to a familiar nested-loop structure:

```python
for p in points:
    for r in receivers:
        for t in transmits:
            out[p, r, t] = signal_for_point(
                point_position=point_position[p],
                receiver=receiver[r],
                signal=signal[t, r, :],
                wave_data=wave_data[t],
                ...
            )

image[p] = sum_over_receivers_and_transmits(out[p, :, :])
```

vbeam expresses this as transformations:

```python
ForAll("points")
ForAll("receivers")
ForAll("transmits")
Apply(np.sum, Axis("receivers"))
Reduce.Sum("transmits")
Apply(scan.unflatten, Axis("points", becomes=["width", "height"]))
```

The important division of labor is:

```text
Spec:
    knows axis names

spekk:
    translates axis names into vmap/sum/reduce behavior

vbeam.fastmath:
    chooses JAX or NumPy execution

signal_for_point:
    remains a scalar beamforming kernel
```

## Current takeaway

spekk is not changing ultrasound beamforming math.

It is an engineering layer that lets vbeam write:

```text
vectorize over receivers
sum over receivers
reduce over transmits
unflatten points
```

using semantic dimension names instead of fragile numeric axes.

This is one of the bridges from a clean pixel-based scalar kernel to JAX/vmap/GPU-style
execution.

## Next step

Return to `get_das_beamformer()` and trace a full transformation pipeline step by step:

```text
input setup.spec
after vectorize_over_datacube
after sum_over_dimensions
after Reduce.Sum("transmits")
after unflatten_points
```

The goal is to see how the output spec changes through the full DAS beamformer.
