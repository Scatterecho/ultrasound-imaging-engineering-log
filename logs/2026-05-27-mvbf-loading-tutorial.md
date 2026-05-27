# 2026-05-27 - MVBF covariance tutorial and diagonal-loading experiment

Today I implemented and ran a small numerical tutorial that mirrors the per-pixel FBSS-MVBF procedure in `pybf`.

## Tutorial script

```text
tests/code/tutorial_mvbf_weights/main.py
```

The script uses a deliberately small synthetic case:

```text
Selected aperture channels M = 8
Subarray length L = 4
Overlapping snapshots K = M - L + 1 = 5
```

The desired target after delay alignment is modeled as:

```text
target = [1, 1, ..., 1]
```

A spatially varying residual clutter component is added, so the script can demonstrate how adaptive MVBF weights differ from fixed DAS weights.

## Verified pixel-level processing chain

```text
delayed channel vector
    -> overlapping spatial-smoothing snapshots
    -> forward/backward covariance matrices
    -> diagonal loading
    -> inverse covariance
    -> MV/Capon adaptive weights
    -> averaged subarray output for one complex pixel
```

## First run: PyBF-style loading

Using the default example and `loading_scale=1.0`, which mirrors the strength used in `beamformer_mvbf_spatial_smooth.py`, the result showed:

```text
sum(MV weights) = 1
Target response:  DAS = 1, MVBF = 1
|clutter leakage|:
    DAS  = 0.011069
    MVBF = 0.006563
```

Interpretation:

```text
The distortionless constraint preserves an aligned target response.
The covariance-derived adaptive weights reduce residual clutter leakage.
```

## Diagonal-loading sweep

For the 70-degree clutter phase-step example:

```text
DAS reference |clutter leakage| = 0.011069

loading    MV leakage     DAS/MV reduction    cond(R_loaded)
0          0.011467       0.97x               4.4e16
0.001      0.000929      11.92x               1804
0.01       0.000983      11.26x                181
0.1        0.001828       6.06x                 19
1.0        0.006563       1.69x                  2.8
10.0       0.010340       1.07x                  1.18
```

Engineering lesson:

```text
Too little loading:
    covariance inversion becomes poorly conditioned and unreliable.

Moderate loading:
    strong adaptive clutter suppression while remaining manageable.

Heavy loading:
    high numerical stability, but MVBF becomes increasingly conservative.
```

## Scene dependency

A different synthetic clutter phase step, 110 degrees, produced a different best loading behavior:

```text
DAS reference leakage = 0.062029
loading_scale = 1.0 -> MV leakage = 0.015891, approximately 3.90x reduction
```

This demonstrates that a loading parameter is not universally optimal across all pixels or clutter configurations.

## Transcranial imaging implication

For transcranial imaging, skull-induced phase aberration can cause the desired target response to deviate from an all-ones steering vector. A strong MVBF optimizer applied with an incorrect steering model may suppress true target signal.

A more relevant future engineering route is therefore:

```text
skull-aware phase/delay correction
    -> corrected steering vector
    -> robust MVBF with tuned or adaptive diagonal loading
```

## Code commit

```text
45deaae Add MVBF covariance and loading tutorial
```
