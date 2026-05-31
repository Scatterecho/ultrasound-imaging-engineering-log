# 2026-05-31 PyBF 阶段总结与下一阶段 vbeam 学习计划

## PyBF 阶段完成内容

本阶段以 Sergio5714/pybf 为对象，从环境配置、DAS 主链路、性能优化、声速修正、PICMUS 数据、MVBF、动态孔径到显示出口，完成了一轮工程化学习。

已经覆盖的内容包括：

- Python/conda/PyCharm 环境配置；
- Git fork、remote、commit、push 与学习日志仓库；
- `realtime_beamformer/main.py` 跑通；
- `BFCartesianRealTime.__init__()` 与 `beamform()` 的初始化/逐帧执行分工；
- `Transducer`、`ImageSettings`、`tx_strategy`、`rx/tx delays`；
- RF preprocessing：filter、IQ demodulation、interpolation/remodulation；
- DAS NumPy vs Numba benchmark；
- IQ phase rotator 与 LUT 优化；
- plane-wave speed-of-sound correction；
- PICMUS 数据下载、检查与低通道 DAS/MVBF 示例；
- MVBF/Capon 权重、协方差、spatial smoothing、FBSS、diagonal loading；
- MVBF 真实数据诊断、背景变亮原因、apodization 进入协方差；
- fixed aperture vs dynamic aperture；
- 自适应全图 MVBF 初版与负结果分析；
- `make_video.py` 显示/视频出口修复。

## 重要认知

1. PyBF 是非常适合学习超声成像工程链路的项目，但不是现代实时 GPU 框架。
2. DAS 主链路已经基本掌握：数据、探头、成像区域、延时、预处理、加权求和、显示。
3. MVBF 不应被视为“必然优于 DAS”的魔法算法；它高度依赖孔径、相位校正、协方差估计、loading 和数据场景。
4. 对经颅成像来说，下一阶段更重要的是：GPU 快速成像、异质声速/相位畸变校正、可微 beamforming 和工程性能。

## 为什么下一步适合学习 vbeam

vbeam 项目定位为 fast and differentiable beamformer，基于 JAX 等机器学习框架。它的 README 明确强调：

- 用 JAX `vmap` 将 pixel-based beamforming 并行化到 GPU；
- 支持 PICMUS plane-wave imaging 示例；
- 支持 heterogeneous speed-of-sound map 的 time-domain REFoCUS 示例；
- beamforming component 可微，可用于优化 apodization、声速图或其他参数；
- 可以和机器学习算法集成。

这正好承接 PyBF 阶段：

```text
PyBF: 理解传统 DAS/MVBF 工程链路
vbeam: 学习 GPU/JAX/可微/异质介质快速成像
```

对于经颅超声成像方向，vbeam 比继续深挖 PyBF 更贴近下一阶段目标：快速重建、异质声速、可微优化和 GPU 原型验证。

## 下一阶段建议路线

1. 新建/fork `magnusdk/vbeam`；
2. 先在独立 conda 环境中安装 Python >=3.9、JAX、vbeam；
3. 跑通官方 PICMUS PWI example；
4. 对照 PyBF 中的 PICMUS 数据链，理解 vbeam 的 setup/data/spec/scan abstraction；
5. 跑 heterogeneous speed-of-sound / REFoCUS example；
6. 最后尝试把经颅相位/声速校正思想迁移到 vbeam 的可微框架中。
