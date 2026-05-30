# 2026-05-30 自适应全图 MVBF：文献式流程与负结果

## 动机

在 fixed aperture 与 dynamic aperture 的对比中，发现简单打开动态孔径并没有让 MVBF 稳定优于 DAS。为了避免继续无脑扫参数，今天单独写了一个更接近基础文献流程的自适应全图 MVBF 脚本。

## 新增脚本

`G:\Projects_Ultrasound\1_Sergio5714_pybf\pybf\tests\code\adaptive_mvbf_full_image\main.py`

这个脚本不修改原始类，而是作为独立实验入口。它实现：

```text
DAS 先产生 delayed channel data
→ 每个 pixel 选动态接收孔径
→ 最多保留 max_channels 个局部阵元
→ spatial smoothing 子阵估计协方差
→ 使用多个 plane waves 作为额外快拍
→ 使用轴向邻域 pixel 作为额外快拍
→ forward-backward averaging
→ diagonal loading
→ MV/Capon 权重输出
```

## 测试命令

快速 smoke test：

```powershell
python tests\code\adaptive_mvbf_full_image\main.py --image-res 40 60 --max-channels 64 --subarray-length 8 --axial-half-window 1
```

更有参考价值的低分辨率实验：

```powershell
python tests\code\adaptive_mvbf_full_image\main.py --image-res 80 120 --max-channels 64 --subarray-length 8 --axial-half-window 1
```

## 80 x 120 结果

- DAS time: about 0.81 s
- Adaptive MVBF time: about 33.59 s
- active channel count:
  - min: 5
  - median: 53
  - max: 64
- snapshot count:
  - median: about 1080

CNR 结果：

| ROI | DAS | Adaptive MVBF | Delta |
|---:|---:|---:|---:|
| 1 | 8.489 dB | 7.904 dB | -0.585 dB |
| 2 | 8.573 dB | 7.697 dB | -0.876 dB |
| 3 | -0.527 dB | 0.559 dB | +1.086 dB |

## 结论

这个版本更接近基础 MVBF 文献流程，但仍没有全面超过 DAS。这个负结果很重要：MVBF 的价值不在于“公式更高级所以一定更好”，而在于在合适的数据、孔径、相位校正、协方差估计和 loading 设置下，它可能改善分辨率、旁瓣或局部对比。

对于 speckle/cyst phantom，尤其是当前只用 4 个 plane waves、朴素 steering vector、简单 local Hanning 和固定 loading 的情况下，MVBF 不稳定优于 DAS 是合理现象。

## 后续方向

暂时不继续纠结这一块。后续如果再回到 MVBF，应优先考虑：

- 更系统的 diagonal loading sweep；
- local apodization 与硬通道选择的平滑化；
- temporal / angular averaging；
- 点目标 resolution phantom 上的 FWHM 对比；
- 经颅场景中的相位畸变校正后再做 MVBF。

今天先把这段作为阶段性探索封存。
