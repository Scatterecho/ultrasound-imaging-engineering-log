# 2026-05-30 MVBF 动态孔径：从固定中心裁剪到 pixel-dependent aperture

## 背景

昨天的 PICMUS DAS vs MVBF 对比中发现：当 `channel_reduction` 不是全孔径时，MVBF 图像肉眼上有时更干净；但图像两侧会变暗或缺失。进一步看源码发现，原始 `BFMVBFspatial` 的 `channel_reduction` 不是动态孔径，而是固定截取中心若干阵元：

```python
start_i = ceil((n_channels - channel_reduction) / 2)
stop_i = start_i + channel_reduction
```

因此，`channel_reduction=32` 实际含义是：所有 pixel 永远只用中心 32 个阵元。这种结果看起来干净，但它不是完整图像意义上的公平 MVBF。

## 核心理解

DAS 中的 `alpha_fov_apod` 可以理解为接收可视角，也就是 pixel-dependent receive aperture：

```text
half_aperture = z * tan(alpha_fov / 2)
```

浅层开小孔径，深层开大孔径；侧向 pixel 的孔径中心也应跟着 pixel 横向位置移动。

因此，MVBF 也应该遵守类似的物理约束：

```text
for each pixel:
    active_channels = channels visible from this pixel
    then run MVBF inside that local aperture
```

## 今日代码改动

在：

`G:\Projects_Ultrasound\1_Sergio5714_pybf\pybf\scripts\beamformer_mvbf_spatial_smooth.py`

为 `BFMVBFspatial` 增加：

```python
dynamic_aperture=False
```

默认不改变原始 fixed-center 行为；当开启 `dynamic_aperture=True` 时：

1. 不在父类 apodization 里提前应用固定中心 channel mask；
2. 每个 pixel 通过 `self._apod[pixel, :] > 0` 找到可见阵元；
3. 若可见阵元数超过 `channel_reduction`，则保留距离该 pixel 横向位置最近的若干阵元；
4. 在局部孔径内执行 FBSS-MVBF。

同时在：

`tests/code/benchmark_picmus_das_vs_mvbfss/main.py`

增加：

```powershell
--dynamic-aperture
```

用于比较 fixed aperture 与 dynamic aperture。

## 测试

快速测试命令：

```powershell
python tests\code\benchmark_picmus_das_vs_mvbfss\main.py --image-res 40 60 --channel-reduction 128 --window-width 8 --dynamic-aperture
```

结果可正常运行并保存图像。动态孔径版本相比 fixed full aperture 更快，因为每个 pixel 只使用局部有效孔径，而不是把无效通道也纳入 MVBF 计算。

## 重要发现

dynamic aperture 并不必然让 MVBF 图像更好。原因包括：

- 当前实现是“硬选最近 N 个阵元”，孔径集合会随横向位置跳变；
- MVBF 对协方差估计很敏感，通道集合跳变可能导致纹理不连续；
- 默认是否把 apodization 乘进协方差会显著改变结果；
- fixed-center 32 阵元图像之所以好看，部分原因是它遮掉了侧边困难区域。

## 工程结论

固定中心 `channel_reduction` 更像 demo 或局部 ROI 工具；动态孔径才更接近完整成像系统。但动态孔径需要更平滑的 local apodization、稳定的协方差估计和合理 loading，否则不会天然优于 DAS。
