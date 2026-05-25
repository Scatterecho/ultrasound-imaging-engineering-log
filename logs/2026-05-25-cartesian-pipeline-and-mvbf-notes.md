# 2026-05-25 - Cartesian pipeline, visualization fix, and MVBF notes

Today I continued studying the `pybf` project from an engineering perspective.

## Cartesian HDF5 pipeline

I ran the full offline Cartesian beamformer pipeline on:

```text
tests/data/rf_dataset.hdf5
```

This dataset contains:

```text
1 frame
3 acquisitions / shots
TX strategy: PW_3_12
RF shape per shot: (192, 2313)
```

The output image dataset was generated at:

```text
tests/output/cartesian_run/image_dataset.hdf5
```

It contains:

```text
beamformed_data/frame_0/high_res_image     (180, 120), complex128
beamformed_data/frame_0/low_res_image_0    (180, 120), complex128
beamformed_data/frame_0/low_res_image_1    (180, 120), complex128
beamformed_data/frame_0/low_res_image_2    (180, 120), complex128
params/pixels_coords_x_z                   (2, 21600)
params/elements_coords                     (2, 192)
sim_params/scatters_data                   (2, 2748)
```

I also generated a local comparison image showing the three LRIs and their coherent-compounded HRI.

## Visualization CLI bug

I found and fixed a bug in:

```text
scripts/visualize_image_dataset.py
```

The command-line entry point was passing optional arguments positionally, which shifted values into the wrong function parameters. In particular, `db_range` could be passed into `frames_to_plot`, causing:

```text
TypeError: object of type 'float' has no len()
```

The fix was to call `visualize_image_dataset(...)` with keyword arguments.

Code commit:

```text
90f21ac Fix visualize image dataset CLI arguments
```

## MVBF learning thread

I started reading the MVBF scripts:

```text
scripts/beamformer_global_mvbf.py
scripts/beamformer_mvbf_spatial_smooth.py
scripts/beamformer_mvbf_DCR.py
```

Initial takeaway:

```text
DAS:
    delay alignment -> fixed apodization -> channel sum

MVBF / Capon:
    delay alignment -> covariance estimation -> adaptive weights -> weighted sum
```

The spatial smoothing MVBF implementation is particularly useful for learning because it constructs local subarray snapshots for each pixel, estimates a covariance matrix, applies diagonal loading, and computes adaptive weights using an MVDR-style formula.

Next study focus: understand exactly how MVBF weights are computed per pixel and how their role compares with receive apodization in DAS.
