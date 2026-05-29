# 2026-05-29 PICMUS 数据接入与脚本修复

## 今天完成的事

- 查到 PyBF README 中提到的 PICMUS 数据来源，并将 Polybox 包中的数据解压到本地项目目录。
- 数据位置：
  - `G:\Projects_Ultrasound\1_Sergio5714_pybf\pybf\tests\data\Picmus\contrast_speckle\rf_dataset.hdf5`
  - `G:\Projects_Ultrasound\1_Sergio5714_pybf\pybf\tests\data\Picmus\resolution_distorsion\rf_dataset.hdf5`
- 原始压缩包临时放在 `C:\tmp`，解压后已经删除，避免 C 盘残留大文件。

## 数据检查

两个 PICMUS 数据集均可被 `DataLoader` 正常读取：

- frames: 1
- LRIs / frame: 75
- transmit strategy: `PW_75_16`
- sampling frequency: about 20.832 MHz
- transducer elements: 128
- single LRI RF shape: `(128, 3328)`

## 脚本修复

为了让 PICMUS 示例在当前 Windows + PyCharm/PowerShell 环境中稳定运行，修复了几个工程问题：

- `scripts/picmus_eval.py`
  - 当 `is_plot=False` 时，不再访问未创建的 matplotlib axis。
- `tests/code/low_ch_das_vs_mvbf_spatial/main.py`
  - 使用 `matplotlib.use("Agg")`，避免 GUI backend 干扰。
  - CNR 计算关闭绘图。
  - 默认不弹窗、不保存根目录图片。
- `tests/code/low_ch_mvbf_global/main.py`
  - 使用 `Agg` backend。
  - CNR/FWHM 评估关闭绘图。
  - 默认只保存结果图，不弹窗。
- `scripts/beamformer_global_mvbf.py`
  - 使用 `Agg` backend，使 global MVBF 脚本可在非交互环境下运行。

## 工程经验

这一步的价值不是“下载了数据”这么简单，而是把项目从作者的本地研究环境迁移到自己的工程环境中。真实项目里，很多坑并不在算法公式，而在：数据路径、backend、脚本参数、绘图副作用、以及大文件是否应该进 Git。

本次明确将 `tests/data/Picmus/` 和 benchmark 生成图表排除出 Git，只提交可复现实验脚本。
