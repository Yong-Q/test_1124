# 三维 ξ 自由能训练

一套命令训练八个配置。输入采用预处理好的三维 \(\xi\) 场和探针响应，训练选择过量自由能或总内禀自由能。第 7–11 页给出与代码对应的损失和二阶算法。[训练技术报告 PDF](docs/report.pdf) · [总方法 PDF](docs/methods_revised.pdf)

## 三个模型目录

| 目录 | 配置名称 | 输出与损失 |
|---|---|---|
| [邻域 MLP](models/neighborhood_mlp/) | `fft-excess`、`fft-intrinsic`、`sparse-excess` | 152 项一阶、分层 24×24 二阶、局部 EL、均相元素核。FFT 或稀疏滤波共用同一 MLP |
| [六层 3D CNN](models/cnn3d/) | `cnn-single-excess`、`cnn-single-intrinsic`、`cnn-dual-excess` | 同上四项，单／双尺度改变长程密度支路 |
| [稳定核](models/stable_kernel/) | `stable-base`、`stable-bounded` | 带非负约束的总内禀导数最小二乘加岭项，分别使用正定基底或受限解析修正 |

网络训练使用解析方向导数构造曲率，再自动微分损失。稳定核使用解析基底构造拟合矩阵。配置在 [variants.json](variants.json)，框架公用数值模块在 `backend/`。

## 本地快速运行

推荐 Python 3.10 或 3.11。先安装适合本机 CUDA 的 PyTorch，再安装余下依赖。CPU 也能跑两个小案例。

```bash
python -m pip install -r requirements.txt
bash run.sh list
bash run.sh validate
bash run.sh train fft-excess --case ch4 --device cpu --output runs/fft_demo
```

GPU 把 `--device` 改成 `cuda`。一次验证 8 个配置：

```bash
bash verification/smoke.sh runs/local_cpu cpu
bash verification/smoke.sh runs/local_gpu cuda
```

每次运行用新输出目录。训练从种子 617 初始化，输出 `weights.pt`、`summary.json` 以及 `train_predictions.npz` 和 `test_predictions.npz`。JSON 包含参数更新数、CH₄ 更新前后 RMSE、冻结后 NH₃ RMSE 和解析导数对自动微分的核对。RMSE 数组按一阶、二阶排列。小案例一阶用完整 152 个探针，训练和 RMSE 冒烟核对的二阶用采样的 24×24 块。参考标签复算核对全部 152×152 条目。完整训练在留出评估时计算全部 152² 个条目。

## 两个预处理案例

| 案例 | 用途 | 内容 |
|---|---|---|
| [CH₄](cases/ch4/) | 短训练和生成外势状态 | 400 帧预处理小块、一个重加权状态、过量和总内禀标签 |
| [NH₃](cases/nh3/) | 冻结权重后的独立验证 | 同样的输入输出格式，训练入口固定把它排除在参数更新之外 |

每个案例的 `input.npz` 含聚合后的 `xi_frames[400,125,133]`、三维格距、单格体积、帧权重和探针能量 `probe_energy[400,152]`。133 通道依次是总密度、128 个径向–角向环境分量、按 C/H/N/O 排列的四个元素数密度。前两类来自同一原子分辨 ξ 的格点累加，元素密度的和与总密度一致。公开示例保留数值场和标签，没有原子坐标。

`reference.npz` 的 `c1_excess`、`c2_excess` 是过量自由能的带负号投影，`y1_intrinsic`、`y2_intrinsic` 是对应的总内禀量。程序按配置自动选择。`--case` 也可以传相对路径，例如 `--case cases/ch4`。两例从原预处理的中心取 \(5^3\) 格点，重建小块探针标签，并保留 32 个原盒体相监督模式。小块训练设周期边界，验证目的是复算、训练与求导对齐。`archived_full_box.npz` 另保留原盒均相参考标签作溯源，训练小例从小块的 `reference.npz` 读标签。

## 一键生成 300 条三维外势微扰

```bash
bash run.sh perturb --case ch4 --count 300 --output runs/ch4_perturb_300.npz
bash run.sh validate --case ch4 --states runs/ch4_perturb_300.npz --row 299
bash run.sh train fft-excess --case ch4 --states runs/ch4_perturb_300.npz --row 299 \
  --output runs/ch4_generated_demo
```

命令在一份已有预处理输入上组合空间探针、径向–角向探针及两者的组合。对每个候选算 Boltzmann 权重、检查帧和时间块有效样本数，再保存通过筛选的权重、外势系数和两套标签。固定种子可复算。生成文件保存在用户指定的位置，仓库附带脚本和小例输入。300 条是重加权状态。

## 直接调用完整数据训练

```bash
bash run.sh train cnn-single-excess \
  --manifest path/to/training_split.json --bulk-index path/to/bulk_labels_index.json \
  --device cuda --epochs 1 --batch 4 --microbatch 1 --memory-gib 20 \
  --output runs/cnn_full
```

`training_split.json` 是已有的 152 探针训练划分。各项提供 `case`、`species`、`states`、预处理 `cache` 和 `labels`，可选训练专用的 `additional_labels`。`bulk_labels_index.json` 给出各角色、各轨迹的体相标签路径。文件项的相对路径以清单本身的目录为基准。命令行参数里的相对路径以运行时工作目录为基准。完整训练需要用户已有的预处理数据。加 `--check-only` 检查完整清单与文件位置，实际读数训练还会校验帧与标签形状。稳定核接受同一 `--manifest`，由训练入口重新拟合并保存两套权重，可省略 `--bulk-index`。

`bash run.sh train --help` 查看参数。原报告训练用训练标签 RMS 归一化，所有模型从各自初始化训练。外势迭代和训练各自计算。局部小例验证用的尺度只由 CH₄ 计算。NH₃ 只用来报告验证。

## 验证和源码说明

[参考标签复算](verification/reference_cpu.json) · [外势生成验证](verification/perturb_300_cpu.json) · [本地测试记录](verification/README.md)

报告源和总方法源可在 `docs/` 查阅。 `cases/build_fixtures.py` 记录了如何从项目已有预处理重建这两份小例，需要能访问原项目输入；日常使用直接读已交付的 NPZ 文件。
