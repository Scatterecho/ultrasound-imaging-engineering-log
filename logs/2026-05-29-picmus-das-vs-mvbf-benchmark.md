# 2026-05-29 PICMUS DAS vs FBSS-MVBF benchmark：全孔径、公平对比与工程意义

## 新增脚本

新增 benchmark：

`tests/code/benchmark_picmus_das_vs_mvbfss/main.py`

目标是对 PICMUS contrast/speckle 数据做更公平的 DAS 与 FBSS-MVBF 对比：

- DAS 和 MVBF 使用相同接收孔径；
- 支持 `channel_reduction` 扫描；
- 支持 `window_width` 扫描；
- 支持是否在 MVBF 协方差前应用 receive apodization；
- 输出 runtime、CNR CSV、以及 side-by-side dB 图。

## 默认测试参数

- dataset: PICMUS contrast/speckle RF dataset
- image resolution: `80 x 120`
- plane waves: `[33, 37, 38, 42]`
- channel reduction: 128
- window width: 8
- MVBF apply apodization: True

## 全孔径结果

`channel_reduction=128, L=8`

- DAS runtime: about 0.82 s
- MVBFss runtime: about 35.5-37.9 s
- CNR delta:
  - ROI1: about `+0.21 dB`
  - ROI2: about `-0.36 dB`
  - ROI3: about `+0.35 dB`

结论：全孔径下 MVBF 的图像完整性恢复了，但相对于 DAS 的收益非常有限，并且 ROI 相关。

## 子阵窗口扫描

全孔径 `channel_reduction=128` 下：

| L | DAS time | MVBF time | CNR 趋势 |
|---:|---:|---:|---|
| 8 | ~0.82 s | ~37.9 s | ROI1/3 略升，ROI2 略降 |
| 16 | ~0.81 s | ~36.9 s | 与 L=8 基本一致 |
| 32 | ~0.81 s | ~42.4 s | 更慢，部分 ROI 变差 |

说明当前数据和参数下，`window_width` 不是主导变量。MVBF 的表现更多受 covariance、loading、apodization、孔径策略和数据本身影响。

## 为什么非全孔径时 MVBF 肉眼更好

之前 `channel_reduction=32` 时，MVBF 肉眼看起来比 DAS 更干净。但这个现象要谨慎解释：

- 源码里的 `channel_reduction` 是固定中心孔径裁剪；
- 只用中心阵元会避开部分边缘阵元的大角度相位误差和旁瓣；
- 图像两侧会变暗或不完整；
- 因此它的“更干净”一部分来自更保守的孔径，而不一定来自 MV 权重本身。

## 工程意义

固定中心 `channel_reduction` 更像教学/demo/局部 ROI 工具，不适合作为完整成像策略。

真正工程化的方向应是：

- 动态孔径 / F-number：每个像素按几何位置选择合理阵元；
- 全孔径覆盖 + 子阵平滑：保证 FOV，同时稳定估计 covariance；
- 多孔径或多角度复合：用融合策略换取鲁棒性；
- 在经颅场景中结合声速/相位畸变校正，否则 MVBF 可能因 steering vector 不匹配而抑制真实目标。

## 今日结论

MVBF 的工程价值不在于“默认一定比 DAS 好”，而在于它提供了一套利用通道协方差信息进行自适应抑制的机制。它能否超过 DAS，取决于前面的延时、孔径、相位校正和稳健估计是否足够成熟。
