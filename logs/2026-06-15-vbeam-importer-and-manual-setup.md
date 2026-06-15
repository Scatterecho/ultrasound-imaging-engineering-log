# 2026-06-15 vbeam: importer mapping and manual SignalForPointSetup

## Context

After confirming that both workflows are usable:

- `friends/run_picmus_plane_wave.py` runs directly in PyCharm.
- `friends/notebooks/plane_wave_dataset_local_cpu.ipynb` runs cell by cell in
  browser-based Jupyter Lab.

I continued from the previous vbeam learning thread. Because `Spec`, `ForAll`,
`Apply`, `Reduce`, apodization, wavefronts, speed of sound, interpolation,
fastmath, and spekk transformations had already been studied, I did not repeat
the beamformer transformation-chain basics.

The next useful step was to understand the data-import layer:

```text
UFF channel_data + scan
-> import_pyuff(...)
-> SignalForPointSetup
```

This layer is important for future transcranial and custom acquisition data,
because new datasets will often need a custom setup constructor rather than a
new DAS implementation.

## Reading `import_pyuff`

The relevant source is:

```text
vbeam/data_importers/pyuff_importer.py
```

For the PICMUS carotid plane-wave dataset, the original UFF data has:

```text
channel_data.data shape = (1536, 128, 75)
dtype = float32
N_samples = 1536
N_channels = 128
N_waves = 75
sampling_frequency = 20832000.0
initial_time = 0.0
modulation_frequency = 0.0
sound_speed = 1540.0
```

The raw UFF order is:

```text
(samples, channels, waves)
```

vbeam expects:

```text
(transmits, receivers, signal_time)
```

So the importer performs:

```python
receiver_signals = np.transpose(data, (2, 1, 0))
```

Therefore:

```text
(1536, 128, 75)
-> (75, 128, 1536)
```

and the final setup has:

```text
setup.signal.shape = (75, 128, 1536)
setup.spec["signal"] = ["transmits", "receivers", "signal_time"]
```

## Why the signal becomes complex

The original UFF signal is `float32`, but:

```text
channel_data.modulation_frequency = 0.0
```

The importer treats this as RF data and applies a Hilbert transform:

```python
if numpy.abs(modulation_frequency) == 0:
    receiver_signals = np.array(hilbert(receiver_signals), dtype="complex64")
```

So:

```text
float32 RF -> complex64 analytic signal
```

This is an important custom-data lesson:

- If future data is RF, a similar conversion may be needed.
- If future data is already IQ, it should not be blindly Hilbert-transformed
  again.

## Probe, scan, and wave mapping

The probe maps directly to receiver geometry:

```python
receivers = ElementGeometry(
    np.array(channel_data.probe.xyz),
    np.array(channel_data.probe.theta),
    np.array(channel_data.probe.phi),
)
```

For PICMUS:

```text
channel_data.probe.xyz shape = (128, 3)
setup.receiver.position shape = (128, 3)
```

The scan is a PyUFF `LinearScan`:

```text
x_axis shape = (1, 387)
z_axis shape = (1, 609)
```

`parse_pyuff_scan(scan)` converts this into a vbeam `LinearScan`:

```text
shape = (387, 609)
num_points = 235683
coordinate system = CARTESIAN
```

For local CPU experiments, the scan is resized:

```python
setup.scan = setup.scan.resize(x=96, z=128)
```

so:

```text
setup.scan.shape = (96, 128)
setup.point_position.shape = (12288, 3)
```

The UFF sequence has:

```text
len(channel_data.sequence) = 75
```

and maps to:

```python
wave_data = WaveData(
    azimuth=np.array([wave.source.azimuth for wave in sequence]),
    elevation=np.array([wave.source.elevation for wave in sequence]),
    source=np.array([wave.source.xyz for wave in sequence]),
    t0=np.array([wave.delay for wave in sequence]),
)
```

For this dataset:

```text
azimuth first/middle/last ~= -0.279, 0.0, 0.279 rad
setup.wave_data.shape = (75,)
setup.wave_data.source shape = (75, 3)
```

## Plane-wave detection nuance

One subtle point: the UFF sequence reports:

```text
wavefront = Wavefront.spherical
source.xyz contains inf/nan
```

This represents a plane wave focused at infinity. The importer handles this with:

```python
if wavefront == pyuff.Wavefront.plane or numpy.isinf(_wave_xyz).any():
    transmitted_wavefront = PlaneWavefront()
```

So the PyUFF warning around `inf/nan` source coordinates is not fatal; vbeam
recognizes it and chooses `PlaneWavefront`.

## Default apodization from importer

For plane-wave data, the importer builds:

```python
apodization = TxRxApodization(
    transmit=PlaneWaveTransmitApodization(array_bounds),
    receive=PlaneWaveReceiveApodization(Hamming(), 1.7),
)
```

This means the imported DAS setup is not simply "all receivers always on" by
default. It includes a receive apodization rule based on a Hamming window and an
F-number-like angular aperture.

## Manual setup without UFF

To make the setup contract concrete, I created a minimal demo that bypasses UFF:

```text
friends/manual_setup_demo.py
```

The script constructs:

```text
receiver geometry
sender geometry
synthetic complex signal
plane-wave WaveData
linear scan
FastInterpLinspace
PlaneWavefront
ReflectedWavefront
NoApodization
Spec
SignalForPointSetup
```

and then runs:

```python
beamformer = jax.jit(
    get_das_beamformer(
        setup,
        compensate_for_apodization_overlap=False,
        log_compress=False,
        scan_convert=False,
    )
)

result = beamformer(**setup.data).block_until_ready()
```

The validated output was:

```text
setup.size(): {'signal_time': 512, 'receivers': 16, 'points': 3072, 'transmits': 3}
setup.spec: Spec({'signal': ['transmits', 'receivers', 'signal_time'], 'receiver': ['receivers'], 'point_position': ['points'], 'wave_data': ['transmits']})

signal shape = (3, 16, 512)
receiver shape = (16,)
wave_data shape = (3,)
point_position shape = (3072, 3)

result shape = (48, 64)
result dtype = complex64
```

The output image was saved as:

```text
friends/outputs/manual_setup_demo_abs.png
```

The synthetic signal is not intended to be physically realistic. The purpose of
the demo is to verify the minimum data contract needed to feed vbeam without
UFF.

## Minimum setup contract

The practical minimum looks like:

```python
SignalForPointSetup(
    sender=sender,
    point_position=None,
    receiver=receiver,
    signal=signal,
    transmitted_wavefront=PlaneWavefront(),
    reflected_wavefront=ReflectedWavefront(),
    speed_of_sound=speed_of_sound,
    wave_data=wave_data,
    interpolate=FastInterpLinspace(...),
    modulation_frequency=...,
    apodization=...,
    spec=spec,
    scan=scan,
)
```

Custom data should be mapped as:

```text
RF/IQ data
-> signal: (transmits, receivers, signal_time)

element positions
-> receiver = ElementGeometry(position=(receivers, 3))

transmit angles / virtual sources / delays
-> wave_data = WaveData(..., shape=(transmits,))

image grid
-> scan = linear_scan(...) or sector_scan(...)

sampling axis
-> interpolate = FastInterpLinspace(min=t0, d=1/fs, n=n_samples)

speed assumption
-> speed_of_sound

wave model
-> transmitted_wavefront / reflected_wavefront

aperture rule
-> apodization
```

## Current takeaway

vbeam's real input interface is not UFF. UFF is just one importer.

The real contract is:

```text
SignalForPointSetup + Spec
```

For future transcranial, compound plane-wave, or custom MATLAB/Python data, the
main engineering task is to construct a correct `SignalForPointSetup`.

## Next step

A useful next experiment is to make `manual_setup_demo.py` more physical:

1. Replace the broadcast synthetic pulse with a single-scatterer simulated RF/IQ
   signal.
2. Verify that vbeam focuses the scatterer to the expected image location.
3. Then extend the setup to multiple plane-wave transmits, custom receive
   apodization, and eventually heterogeneous speed-of-sound experiments.
