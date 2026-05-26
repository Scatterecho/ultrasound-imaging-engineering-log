# 2026-05-26 - MVBF covariance and adaptive weights

Today I focused on understanding MVBF / MVDR / Capon beamforming at the pixel level.

## DAS vs MVBF

DAS uses fixed receive weights:

```text
delay alignment -> fixed apodization -> channel sum
```

For a pixel `p`, after delay alignment:

```text
x(p) = [x1, x2, ..., xN]^T
```

DAS computes:

```text
y_DAS(p) = sum_e w_apod(e, p) * x_e(p)
```

The apodization weights are predefined by geometry/windowing, such as expanding aperture and Hanning weighting.

MVBF instead computes adaptive weights from the data:

```text
y_MV(p) = w_MV(p)^H * x(p)
```

The key distinction:

```text
DAS apodization:
    fixed, geometry/window-based, usually real and smooth

MVBF weights:
    data-adaptive, covariance-based, pixel-dependent, often complex
```

## MVBF optimization idea

MVBF solves a constrained minimization problem:

```text
minimize    w^H R w
subject to  w^H a = 1
```

where:

```text
R = covariance matrix of delayed channel data
a = steering vector of the target response
w = adaptive weights
```

After delay alignment, the ideal target response is approximately phase-aligned across channels, so the project uses:

```text
a = [1, 1, ..., 1]^T
```

The solution is:

```text
w = R^-1 a / (a^H R^-1 a)
```

In the pybf code this appears as:

```python
ones_vect = np.ones(L)
numerator = R_inv @ ones_vect
denominator = np.sum(R_inv @ ones_vect)
w_tilda = numerator / denominator
```

## How covariance is constructed

In `scripts/beamformer_mvbf_spatial_smooth.py`, the delayed RF data is first gathered into a pixel-channel array:

```text
mvbf_data.shape = (n_pixels, channel_reduction)
```

For each pixel, spatial smoothing creates overlapping subarray snapshots. Example:

```text
channel_reduction = 64
window_width L = 16

snapshot 0: channels 0  ~ 15
snapshot 1: channels 1  ~ 16
snapshot 2: channels 2  ~ 17
...
```

The covariance matrix is accumulated from outer products:

```text
R ≈ sum_j s_j s_j^H
```

The code uses:

```python
corr_matr_y = np.outer(np.conj(corr_array[y, :]), corr_array[y, :])
corr_matrix = corr_matrix + corr_matr_y
```

It also uses forward-backward smoothing by flipping subarray snapshots and combining two covariance estimates.

## Why covariance helps

The covariance matrix describes how channel signals co-vary after delay alignment.

Intuition:

```text
Energy/correlation aligned with target steering vector:
    should pass

Energy/correlation inconsistent with target steering vector:
    likely clutter, sidelobe, interference, or model mismatch
    should be suppressed
```

MVBF uses `R^-1` to downweight high-energy interference directions while preserving the target steering response.

## Diagonal loading

The code adds diagonal loading before inversion:

```python
R_inv = np.linalg.inv(R + loading)
```

Purpose:

```text
stabilize matrix inversion
avoid exploding weights
improve robustness when snapshots are limited or coherent
```

This is important because MVBF is powerful but sensitive.

## Transcranial imaging note

In homogeneous soft tissue, after correct delay alignment, the target steering vector can often be approximated as:

```text
a = ones
```

But in transcranial ultrasound, skull-induced aberration can make the true target response channel-dependent:

```text
a != ones
```

Therefore, for transcranial MVBF, a more meaningful direction may be:

```text
skull-corrected steering vector
phase-screen-aware covariance model
channel/angle-dependent aberration correction before MVBF
```

This links MVBF directly with aberration correction rather than treating them as separate problems.

## Main takeaway

```text
DAS:
    I predefine which channels should matter.

MVBF:
    I estimate from local channel covariance which signal combinations should pass or be suppressed.
```

The covariance matrix is the local evidence; the MVBF formula converts that evidence into adaptive per-pixel channel weights.
