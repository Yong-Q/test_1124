# 原子 ξ 过量自由能模型

原子 ξ 使用 8 个连续径向基和 $l=0,1,2,3$ 的 16 个实球谐分量。模型以重加权后的原子环境与周期空间谱为输入，前向输出无量纲过量自由能 $\beta F_{\rm ex}$。训练损失比较由自由能导出的 $c_1$、$c_2$ 或微扰响应。数据包里的外势信息用于形成参考标签和响应，训练网络从 ξ 特征读取输入。

| 入口名 | 网络和训练目标 |
|---|---|
| `ppt` | PPT 的 `directional_shells`。局域 MLP、径向及角向 SOG、5 Å 短程方向核。一个标量自由能，拟合过量 $c_1,c_2$ 的探针投影 |
| `c1` | 独立一阶线。逐格点 Euler–Lagrange 残差，逐状态、逐元素解析拟合隐藏 $\beta\mu$ |
| `c2` | 独立二阶线。拟合纯相元素对的直接关联核，含重加权坐标的二阶链式项 |
| `guided` | 从随机初始化训练一阶线。读取冻结的二阶线权重，用同一对原始态和微扰态的 $\Delta c_1$ 校准一阶线 |
| `mutual` | 两套独立自由能及各自优化器。叠加 MD 响应与按 MD 误差加权的双向互教 |
| `gated` | 双向互教增加零响应门控和辅助梯度投影。较准的一方在同一对状态上生成固定目标 |
| `no-peer` | 保留两线的 MD 数据和直接外场响应目标，用于同预算对照 |

完整架构来源、输入数组和输出单位见 [接口说明](INTERFACE.md)。[原始训练入口](provenance/native/)按发布时所对应的工作区程序归档。它们需要原始预处理清单与特征库的目录结构。

[双向互教专页](MUTUAL.md)说明三种互教配置、梯度去向和各自的使用入口。

## 安装与运行

在仓库根目录运行以下命令。建议 Python 3.10，并先安装与本机 CPU 或 CUDA 匹配的 PyTorch。

```bash
python -m pip install -r requirements-repro.txt
python -m fex_models check --device cpu

python -m fex_models train --model ppt --epochs 1 --device cpu --output runs/ppt_demo
python -m fex_models train --model c2 --epochs 1 --device cpu --output runs/c2_demo
python -m fex_models train --model c1 --epochs 1 --device cpu --output runs/c1_demo
python -m fex_models train --model mutual --epochs 1 --device cpu --output runs/mutual_demo

python -m fex_models train --model guided --teacher runs/c2_demo/weights.pt \
  --device cpu --output runs/guided_demo
python -m fex_models evaluate --model mutual --weights runs/mutual_demo/weights.pt \
  --device cpu --output runs/mutual_validation
```

把 `--device cpu` 改成 `--device cuda` 可运行 GPU 测试。`--memory-gib` 控制 CUDA 分配器预算，默认 8 GiB。`--epochs` 控制轮数，`--lr` 覆盖学习率，默认 PPT 模型为 0.005，格点模型为 0.002。训练输出目录必须是新目录。

[实测记录](provenance/verification.json)包含 CPU 和 RTX 4090 上的算子对照、7 种训练配置的参数更新和检查点重新加载验证。完整复算命令为 `python -m fex_models.verify runs/fex_suite --device cpu --device cuda`。只装 CPU 版 PyTorch 时可只传 `--device cpu`。

## 两个真实预处理样例

| 文件 | 用途 | 来源 |
|---|---|---|
| [ch4_prepared.json](examples/ch4_prepared.json) | 训练及梯度更新 | CH₄，290.5 K，训练轨迹 |
| [c2h6_prepared.json](examples/c2h6_prepared.json) | 冻结权重的留出验证 | C₂H₆，465.4 K，留出轨迹 |

一个案例清单引用若干无损 NPZ 分片，命令只需传入清单。完整的同格式单个 NPZ 也可直接传入。每个样例包含 PPT 投影、一对逐格点原始态与弱微扰态，以及两个有支持的纯相 $c_2$ 波段。格点分支使用完整的 2,001 帧生成加权态和响应，再把固定线性收缩保存成输入张量。样例固定 6 个空间格点和 6 个 PPT 探针，纯相保存两个波段及两次独立方向抽样。代码保留完整特征的四路前向及其导数。

`check` 用两条真实样例复核发布实现与原实现的 (c_1/c_2) 数值。训练命令会从随机种子初始化，并用真实参考标签更新参数。小样每个数据包对应一次参数更新，用于复现接口、梯度与指标。正式训练使用更大的格点集合、波段数和小批量。

每次训练保存 `weights.pt`、`summary.json` 和 `validation_predictions.npz`，其中两条线的权重分别保存。案例由 `--train-case` 和 `--validation-case` 传入，可以重复指定。角色由数据包内的字段校验，训练与验证按源轨迹分开。
