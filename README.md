# 三维 ξ 自由能训练

**代码版本 v0.3.0** · [版本记录](CHANGELOG.md)

一套命令训练八个配置。输入采用预处理好的三维 \(\xi\) 场和探针响应，训练选择过量自由能或总内禀自由能。第 7–11 页给出与代码对应的损失和二阶算法。[训练技术报告 PDF](docs/report.pdf) · [总方法 PDF](docs/methods_revised.pdf)

## 新增的过量自由能模型

[原子 ξ 自由能模型](fex_models/README.md)提供 PPT 的 5 Å 方向核 SOG、独立 $c_1/c_2$ 双线、冻结校准和双向互教。用 `python -m fex_models check` 验证两个真实预处理案例。训练模式、输入数据格式及输出单位见[接口说明](fex_models/INTERFACE.md)。

### 双向互教

仓库中的双向互教入口是 `python -m fex_models train --model mutual`。使用 `--model gated` 运行有门控的版本，使用 `--model no-peer` 做无互教对照。

```bash
python -m fex_models train --model mutual --device cpu --output runs/mutual_demo
```

[双向互教运行说明](fex_models/MUTUAL.md) · [发布版训练损失](fex_models/training.py) · [全量互教原始入口](fex_models/provenance/native/train_mutual.py) · [开关对照原始入口](fex_models/provenance/native/ablation.py)

## 三个模型目录

| 目录 | 配置名称 | 输出与损失 |
|---|---|---|
| [邻域 MLP](models/neighborhood_mlp/) | `fft-excess`、`fft-intrinsic`、`sparse-excess` | 152 项一阶、分层 24×24 二阶、局部 EL、均相元素核。FFT 或稀疏滤波共用同一 MLP |
| [六层 3D CNN](models/cnn3d/) | `cnn-single-excess`、`cnn-single-intrinsic`、`cnn-dual-excess` | 同上四项，单／双尺度改变长程密度支路 |
| [稳定核](models/stable_kernel/) | `stable-base`、`stable-bounded` | 带非负约束的总内禀导数最小二乘加岭项，分别使用正定基底或受限解析修正 |

网络训练使用解析方向导数构造曲率，再自动微分损失。稳定核使用解析基底构造拟合矩阵。配置在 [variants.json](variants.json)，框架公用数值模块在 `backend/`。

## 本地快速运行

推荐 Python 3.10。先安装与本机设备匹配的 PyTorch，再安装 [验证环境的固定依赖](requirements-repro.txt)。CPU 可运行小例，完整网格训练需要 CUDA。GPU3 验证用 Python 3.10.4、PyTorch 2.12.0+cu130。见 [全部实测版本](verification/requirements-tested.txt) 和 [常规依赖](requirements.txt)。

```bash
python -m pip install -r requirements-repro.txt
bash run.sh doctor
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

## 选择模型和传入训练参数

训练接口统一为 `bash run.sh train <配置名> [参数]`。只换下面的配置名，就会切换网络、自由能目标和对应损失。

| 配置名 | 实际模型 | 自由能输出 |
|---|---|---|
| `fft-excess` | FFT 邻域 MLP | 过量 |
| `fft-intrinsic` | FFT 邻域 MLP | 总内禀 |
| `sparse-excess` | 稀疏邻域 MLP | 过量 |
| `cnn-single-excess` | 单尺度六层 CNN | 过量 |
| `cnn-single-intrinsic` | 单尺度六层 CNN | 总内禀 |
| `cnn-dual-excess` | 双尺度六层 CNN | 过量 |
| `stable-base` | 正定基底 | 总内禀 |
| `stable-bounded` | 正定基底加受限修正 | 总内禀 |

| 参数 | 用法 |
|---|---|
| `--case cases/ch4` | 从已附带的小例训练；默认 CPU。训练后用 NH₃ 验证 |
| `--manifest <文件>` | 用完整的 152 探针预处理清单训练；默认 CUDA |
| `--bulk-index <文件>` | 完整网格的 MLP 和 CNN 需要，用于体相匹配损失 |
| `--device cpu\|cuda` | 小例可任选；完整网格训练用 CUDA |
| `--epochs N` | MLP 或 CNN 的训练轮数，默认 1。紧凑小例每轮一次更新 |
| `--batch N` 与 `--microbatch N` | MLP 或 CNN 完整网格每次更新的状态数与 GPU 分批数，默认 4 和 1 |
| `--memory-gib N` | MLP 或 CNN CUDA 分配器预算上限，默认 20。小例 CUDA 也读取这个参数 |
| `--train-states N` 与 `--test-states-per-species N` | 完整网格短测的训练总数和每个留出物种的测试数 |
| `--skip-validation` | 完整网格跳过中间验证集，仍保留留出测试 |
| `--output <新目录>` | 保存参数、指标和预测。每次运行使用新目录 |
| `--check-only` | 核对输入路径与清单，小例会复算标签 |

`--case` 与 `--manifest` 二选一。训练初始化、学习率及损失权重按发布版设定，MLP 和 CNN 默认种子 617、AdamW 学习率 `3e-4`，四项损失权重为 1。稳定核使用一次有界最小二乘，完整网格会同时保存 `base.pt` 和 `bounded_mlp.pt`。它的完整网格分批由 `--microbatch` 控制，GPU 预算由实现按空闲显存选择。

```bash
# CNN 小例，CPU 上实际更新一次
bash run.sh train cnn-single-excess --case cases/ch4 --device cpu --epochs 1 --output runs/cnn_demo

# 稳定核小例，一次有界最小二乘
bash run.sh train stable-bounded --case cases/ch4 --device cpu --output runs/stable_demo

# 用 1 个完整网格训练状态短测 FFT，总内禀可换成 fft-intrinsic
bash run.sh train fft-excess --manifest data/training_split.json \
  --bulk-index data/bulk_labels_index.json --device cuda --epochs 1 --batch 4 --microbatch 1 \
  --train-states 1 --test-states-per-species 1 --skip-validation --output runs/fft_full_smoke
```

查看全部可用参数用 `bash run.sh train --help`。两套数据输入格式和生成外势的命令见下文。

### 双尺度 CNN 从命令到输入

配置名是 `cnn-dual-excess`。它启用 5 Å 短程卷积和 12 Å 长程支路，长程支路读四个元素密度。输出过量自由能，训练由此求过量 `c1/c2`。

先用仓库小例确认环境和参数：

```bash
bash run.sh train cnn-dual-excess --case cases/ch4 --device cuda \
  --epochs 1 --memory-gib 8 --output runs/cnn_dual_demo
```

用项目现有预处理数据跑一个完整网格短测。下面两条相对路径适用于交付仓库与原项目并排放置：

```bash
bash run.sh train cnn-dual-excess \
  --manifest ../xi_3d_charge_20261117_v1/experiments/xi_architecture_50/split.json \
  --bulk-index ../xi_3d_charge_20261117_v1/experiments/xi_prl_bulk_coupling/bulk_elements/run_159428/bulk_labels_index.json \
  --device cuda --epochs 1 --batch 4 --microbatch 1 --fft-group 8 --memory-gib 20 \
  --train-states 50 --test-states-per-species 2 --skip-validation \
  --output runs/cnn_dual_50
```

- `--manifest` 传 152 探针预处理清单。程序按清单读加权状态和方向；`--bulk-index` 提供体相损失的参考核。
- `--batch 4` 是累积四个状态更新一次。GPU 每次处理的状态数由 `--microbatch 1` 控制。`--fft-group 8` 是每组进入 FFT 的方向数。训练仍用全部 152 个一阶方向和每次分层抽出的 24 个二阶方向。
- `--memory-gib 20` 给 PyTorch CUDA 分配器设预算，程序还会根据空闲显存预留余量。训练跑完看 `summary.json` 和 `trained.pt`。小例则输出 `summary.json`、`weights.pt` 和预测数组。

跑完整数据时换成全量预处理清单，并去掉 `--train-states`、`--test-states-per-species`、`--skip-validation`，把 `--output` 改为新目录。

## 两个预处理案例

| 案例 | 用途 | 内容 |
|---|---|---|
| [CH₄](cases/ch4/) | 短训练 | 400 帧预处理小块、一个重加权状态、过量和总内禀标签 |
| [NH₃](cases/nh3/) | 冻结权重后的独立验证 | 同样的输入输出格式，训练入口固定把它排除在参数更新之外 |

每个案例的 `input.npz` 含聚合后的 `xi_frames[400,125,133]`、三维格距、单格体积、帧权重和探针能量 `probe_energy[400,152]`。133 通道依次是总密度、128 个径向–角向环境分量、按 C/H/N/O 排列的四个元素数密度。前两类来自同一原子分辨 ξ 的格点累加，元素密度的和与总密度一致。公开示例保留数值场和标签，没有原子坐标。

`reference.npz` 的 `c1_excess`、`c2_excess` 是过量自由能的带负号投影，`y1_intrinsic`、`y2_intrinsic` 是对应的总内禀量。程序按配置自动选择。`--case` 也可以传相对路径，例如 `--case cases/ch4`。两例从原预处理的中心取 \(5^3\) 格点，重建小块探针标签，并保留 32 个原盒体相监督模式。小块训练设周期边界，验证目的是复算、训练与求导对齐。`archived_full_box.npz` 另保留原盒均相参考标签作溯源，训练小例从小块的 `reference.npz` 读标签。

## 一键生成 300 条逐原子三维外势微扰

见 [低浓度 C₂H₄ 测试](cases/low_density_c2h4/README.md)。使用你的既有预处理缓存，在 240³ 原盒里给每个原子施加随机余弦和周期三维势井，再重加权，生成与原 152 个标签方向对齐的新状态。

```bash
WORKERS=12 bash cases/low_density_c2h4/reproduce.sh \
  path/to/training_split.json runs/c2h4_low_density_3d
```

任意一条训练样本可用 `run.sh perturb --manifest ... --case ... --count 300 --output ...` 指定。状态由命令在本地生成。输出目录的 `training_split.json` 可直接接到下面的完整训练入口。

## 直接调用完整数据训练

```bash
bash run.sh train cnn-single-excess \
  --manifest path/to/training_split.json --bulk-index path/to/bulk_labels_index.json \
  --device cuda --epochs 1 --batch 4 --microbatch 1 --memory-gib 20 \
  --output runs/cnn_full
```

`training_split.json` 是已有的 152 探针训练划分。各项提供 `case`、`species`、`states`、预处理 `cache` 和 `labels`，可选训练专用的 `additional_labels`，可给一条或多条标签分片路径。`bulk_labels_index.json` 给出各角色、各轨迹的体相标签路径。文件项的相对路径以清单本身的目录为基准。命令行参数里的相对路径以运行时工作目录为基准。完整训练需要用户已有的预处理数据。加 `--check-only` 检查完整清单与文件位置，实际读数训练还会校验帧与标签形状。稳定核接受同一 `--manifest`，由训练入口重新拟合并保存两套权重，可省略 `--bulk-index`。

`bash run.sh train --help` 查看参数。原报告训练用训练标签 RMS 归一化，所有模型从各自初始化训练。外势迭代和训练各自计算。局部小例验证用的尺度只由 CH₄ 计算。NH₃ 只用来报告验证。

## 验证和源码说明

[参考标签复算](verification/reference_cpu.json) · [逐原子外势验证](cases/low_density_c2h4/README.md) · [本地测试记录](verification/README.md)

报告源和总方法源可在 `docs/` 查阅。 `cases/build_fixtures.py` 记录了如何从项目已有预处理重建这两份小例，需要能访问原项目输入；日常使用直接读已交付的 NPZ 文件。
