# 自由能模型输入输出契约

## 模型来源

- `ppt` 对应组会汇报第 18–21 页展示的 5 Å 方向核 SOG。原训练配置为 10,000 个状态、50 轮，独立验证在第 42 轮选中 `directional_shells`。
- `c1`、`c2`、`guided` 对应各自更新参数的两条监督线。
- `mutual`、`gated`、`no-peer` 复现相同的主损失和固定状态对上的更新规则。一轮预热、同一检查点、两组等轮数的消融流程见 [原始消融入口](provenance/native/ablation.py)。

架构层实现在 [core](core/)，解析方向导数使用各层的原始显式表达式。样例导出记录在 [provenance](provenance/)。

## 命令

| 命令 | 输入 | 输出 |
|---|---|---|
| `python -m fex_models check` | 预处理数据包及其中冻结的原实现导数 | 每个案例的 5 项算子一致性检查 |
| `python -m fex_models train --model … --output …` | 一个或多个训练数据包，以及独立验证数据包 | 从随机初始化更新参数，保存分支权重和验证指标 |
| `python -m fex_models evaluate --model … --weights … --output …` | 同结构检查点和验证数据包 | 冻结权重计算 RMSE 与 $R^2$ |

所有路径相对调用命令的工作目录解析。默认数据包相对于此模块目录定位。可用 `python -m fex_models train --help` 查询参数。

## 预处理包格式 v1

案例可用单个 NPZ 或 JSON 清单读取。JSON 的 `parts` 指向同目录的 NPZ 分片。分片按字段命名前缀无损拆分。

数组索引采用 NumPy 从零开始的约定。实空间长度以 Å 计。浮点场默认 `float32`，复场默认 `complex64`。

| 字段 | 形状或含义 |
|---|---|
| `format_version`, `case_id`, `role` | 版本 1、源状态标识、`train` 或 `holdout` |
| `feature_scale` | 116 位正值，按训练数据标定的共享不变量尺度 |
| `{prefix}_local` | $[1+D,N_{\rm atom},8,16]$ |
| `{prefix}_radial` | $[1+D,K,8]$ |
| `{prefix}_angular` | $[1+D,J,8,16]$ |
| `{prefix}_short` | $[1+D,Q,8,3]$ |
| `{meta}_wave`, `{meta}_angular_wave`, `{meta}_basis` | 径向谱波数、角向谱波数和 15 个角向基投影，波数单位 Å⁻¹ |
| `{meta}_short_kernels`, `{meta}_mode_weight`, `{meta}_volume` | 三角阶、三个径向壳的有限模态核，角向模态权重，周期体积 Å³ |
| `ppt_y1`, `ppt_y2`, `ppt_probe_indices` | $D$ 个一阶投影、$D\times D$ 个二阶投影及对应探针 |
| `grid_element`, `grid_flat_voxel`, `grid_xyz_A` | 逐元素格点标签索引、格点三维坐标，同一格点可含多个元素 |
| `grid_{base,weak}_label` | 各态的 $\ln(\rho\,\mathrm{Å}^3)+\beta V_{\rm ext}$，维度为有效格点元素对数 |
| `bulk_reference`, `bulk_mode_counts`, `bulk_density_scale` | 支持波段数 $S$ 上的 $E\times E$ 元素关联、波段模态计数、$\sqrt{\rho_E\rho_{E\prime}}$ |
ho_{E'}}) |
| `bulk_draws`, `bulk_shell_indices` | 每波段两次独立抽样，保存的波段下标 |
| `*_native_c1`, `*_native_c2` | 固定种子、初始化模型在原实现计算的导数，供数值一致性测试 |

`prefix` 为 `ppt`、`grid_base`、`grid_weak` 或 `bulk`。`meta` 为 `ppt` 或 `grid`。每个场的首行是加权输入，其余行是对所选变量的线性收缩。纯相块前 $2SE$ 行为一阶响应切向，随后 $SE^2$ 行为外场到密度响应的二阶链式收缩。二阶关联的频域参考单位为 Å³。使用这些预处理输入时可以直接从同一个标量自由能求导。

## 损失及隐藏化学势

格点一阶损失按状态和元素解析消去常数偏移。模型评估中的格点 $R^2$ 使用去掉该常数后的参考波动。PPT 的导数评价采用探针子集。纯相二阶相对 RMSE 按各波段模态数加权。

双向学习使用同一对状态、相同空间点上的 $\Delta c_1$。每条线以各自的参考监督为主。一阶线比较实际格点 EL 残差。二阶线比较纯相直接关联核，并补充来自实际外场的 MD 变化量。每对状态用 MD 残差给两条线分配互教权重，指导方的预测冻结。门控版本还要求指导方优于零响应基线，并投影掉与本线监督冲突的辅助梯度分量。

## 复算与范围

[导出程序](provenance/export_native_examples.py)从研究版已预处理的核和坐标响应库冻结这两个小案例。包内训练入口支持自己的同格式数据包，每个数据包顺序更新一次。此仓库案例用于算子对齐与可执行验证。研究版的 5,000 格点、10,000 状态和迭代统计由原训练与验证入口生成。

## 预测数组

`validation_predictions.npz` 以案例顺序编号，$i$ 与 JSON 中 validation 的顺序相同。字段以 case_i_ 为前缀，后缀包括 ppt_c1、ppt_c2、grid_c1_base、grid_c1_weak、grid_c1_delta、grid_beta_mu_base、grid_beta_mu_weak、bulk_c2 和 bulk_c2_draws。各分支还保存对应状态的 beta_fex 以便诊断。格点 $c_1$ 数组保持未加化学势偏移的预测值，偏移存成独立数组，元素顺序为 grid_element 的去重升序。
