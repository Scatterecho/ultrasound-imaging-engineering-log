# 2026-05-21 - Plane-wave speed-of-sound correction in pybf

Today I extended the `pybf` realtime Cartesian beamformer with a small but meaningful transmit-delay experiment.

## What changed

- Added a reconstruction speed-of-sound interface to `BFCartesianRealTime`.
- Added a nominal plane-wave transmit speed-of-sound interface.
- Corrected the effective plane-wave angle with:

```text
theta = asin(sin(alpha) * sos / sos_PW)
```

where:

- `alpha` is the nominal transmit steering angle.
- `sos` is the homogeneous speed of sound used for reconstruction.
- `sos_PW` is the nominal speed of sound assumed by the transmit steering law.

## Engineering lesson

The original plane-wave distance model is reasonable when the transmit steering law and reconstruction both assume the same homogeneous sound speed. Once the reconstruction speed differs from the transmit nominal speed, the effective propagation angle should be adjusted before computing transmit distance.

This is still a homogeneous-medium approximation. For transcranial ultrasound imaging, the fuller direction is likely a skull-aware correction model, such as phase-screen correction, ray tracing, eikonal solving, or CT-derived aberration compensation.

## Visual check

I ran the realtime beamformer with different reconstruction speeds:

- 1440 m/s
- 1540 m/s
- 1640 m/s

The reconstructed layers and focal structures shifted visibly, confirming that the delay model is actively affecting the image.

## Code commit

`pybf` fork commit:

```text
dfe001e Add speed-of-sound correction for plane-wave delays
```
