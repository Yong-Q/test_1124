# 方法、报告和输入输出核对

| 报告位置 | 代码路径 | 已核对的行为 |
|---|---|---|
| 第 1–3 页，Ξ 构造与理想项 | `cases/build_fixtures.py`、`prepared.py` | 小例沿用 8 个径向、16 个球谐、smoothstep 及原子环境标记。读入时验证 129 个原始通道和 4 个元素通道，元素密度合计等于总密度 |
| 第 4–6 页，导数与 MD 投影标签 | `prepared.py`、`demo.py` | 两个小例的 152 个一阶、152² 个二阶参考均重新计算。预测取负号，按过量／总内禀配置读取对应的参考 |
| 第 7–10 页，邻域 MLP 和 CNN 损失 | `production.py`、`models/*/model.py`、`backend/experiments/xi_conv3d_152_sampled_c2/projection.py` | 相同的一阶 RMS、分层二阶逆概率、局部 EL 和均相匹配实现；FFT／稀疏和单／双尺度各接对应的体相算子 |
| 第 11 页，稳定核损失 | `models/stable_kernel/model.py`、`demo.py`、`backend/experiments/xi_positive_total_pair/fit.py` | 总内禀一阶、二阶标签分组归一化，非负基底与有界软修正用约束最小二乘；两种配置分别从解析设计矩阵拟合 |
| 第 12 页，外势求解 | `backend/experiments/xi_fft_neighborhood_10k/iterate.py`、`backend/experiments/xi_positive_total_external/evaluate.py` | 保留报告对应的自动微分 HVP 与稳定核解析 HVP 代码。这一轮的本地核验覆盖训练、标签和方向导数 |
| 第 3 页，重加权预处理 | `run.sh perturb`、`preprocessing/atomic.py` | 低浓度 C₂H₄ 逐原子施加随机余弦和周期三维势井，重加权；沿原 152 方向生成标签，输出可供训练读取的清单 |

8 个 GPU 配置均已通过参数更新与两方向解析／自动微分核对，最大绝对差为一阶 (3.68\times10^{-8})、二阶 (1.27\times10^{-7})。两份预测输出另外验证了参考标签切片和二阶矩阵对称性。这里的 RMSE 用于程序核对，完整物理性能见报告。
