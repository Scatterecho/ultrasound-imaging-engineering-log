# 2026-05-29 MVBF 真实数据诊断：为什么权重接近 DAS

## 背景

在学习 `beamformer_mvbf_spatial_smooth.py` 时，注意到 MVBF 在一些情况下肉眼并没有明显优于 DAS，甚至背景会偏亮。今天从源码和真实 RF 数据层面做了诊断。

## 核心修改

在 `scripts/beamformer_mvbf_spatial_smooth.py` 中引入了两个更明确的控制参数：

- `diagonal_loading_scale`
  - 控制 diagonal loading 的强度。
  - loading 越强，协方差矩阵越稳定，但 MV 权重越容易退化为接近均匀权重。
- `apply_apodization`
  - 原始代码主要用 apodization 作为 mask 判断像素是否有效。
  - 新接口允许在构造 MVBF 协方差前，真正把接收 apodization 乘到对齐后的通道数据上。

## 诊断脚本

使用：

`tests/code/tutorial_mvbf_real_data/main.py`

新增输出包括：

- `weight_uniform_relative_error`
- `weight_phase_span_deg`
- covariance eigenvalue ratio
- mean/max off-diagonal coherence
- `R * 1` residual

这些指标用来判断 MVBF 权重为什么会接近均匀 DAS 权重。

## 关键理解

MVBF 的权重公式为：

`w = R^-1 a / (a^H R^-1 a)`

对于已经完成延时对齐的通道数据，常用 steering vector 是全 1 向量。如果局部协方差矩阵满足：

`R * 1 ≈ λ * 1`

那么全 1 向量已经近似是协方差矩阵的特征向量，MVBF 求出来的权重就会接近均匀权重。这个时候 MVBF 不会表现出很强的自适应性，本质上接近 DAS。

## 背景变亮的原因

如果 MVBF 构造协方差时没有真正使用接收 apodization，而只是用它判断哪些点有效，那么边缘/低可信阵元的能量仍可能进入协方差估计。这样 MVBF 可能在某些区域抬高背景。

启用 `apply_apodization=True` 后，MVBF 更符合 DAS 中接收加窗的工程直觉：低权重阵元不仅不参与显示，也不应强烈影响协方差估计。

## 工程结论

MVBF 不是简单替代 DAS 的“魔法增强器”。它依赖：

- 延时是否足够准确；
- steering vector 是否仍代表真实目标回波；
- covariance 是否稳定；
- diagonal loading 是否过强；
- apodization 是否进入协方差构建；
- 孔径选择是否合理。

对于经颅超声，这一点尤其重要：如果颅骨导致相位畸变，真实目标未必仍对应全 1 steering vector，MVBF 甚至可能把目标当成干扰压掉。
