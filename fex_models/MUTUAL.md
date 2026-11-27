# 两套过量自由能网络的双向互教

两条线分别保存自由能网络和优化器。第一条线从加权 ξ 求格点 c₁，使用带隐藏化学势的 Euler–Lagrange 残差。第二条线从加权 ξ 求纯相 c₂，保留体相曲率匹配。实际外场的微扰数据为两条线提供可直接比较的响应。

## 运行三个配置

在仓库根目录执行。默认读取 CH₄ 训练包和 C₂H₆ 留出包，每个配置输出独立目录。

```bash
python -m fex_models train --model mutual --device cpu --output runs/mutual_demo
python -m fex_models train --model gated --device cpu --output runs/gated_demo
python -m fex_models train --model no-peer --device cpu --output runs/no_peer_demo
```

| 配置 | 共同的 MD 监督 | 两线之间的监督 |
|---|---|---|
| `mutual` | 逐格点 c₁、纯相 c₂、MD 外场直接响应 | 根据两线在本状态对上的误差分配指导权重 |
| `gated` | 同一组监督量 | 指导方先优于零响应基线。辅助梯度移除与本线监督反向的分量 |
| `no-peer` | 同一组监督量 | 互教权重设为零 |

## 一次互教如何发生

取原始态和弱微扰态，两条线用相同的格点，分别计算各自的 c₁ 变化量。第二条线在两态上的一阶响应差对应沿路径累积的曲率信息。MD 参考来自两态的对数密度和外势差，按元素消去常量偏移。

比较三组变化量后，由更接近 MD 的一方生成指导目标。指导目标使用停止梯度，只有接收指导的一条线更新参数。第二条线还直接用 MD 变化量训练，因此实际外场数据也能补充纯相曲率监督。

`summary.json` 记录损失、两方向的互教权重、网络梯度范数及留出指标。`weights.pt` 分别保存两个网络。`validation_predictions.npz` 保存对应的导数、格点偏移和中间自由能。

- [可运行的小样训练损失](training.py)
- [完整实验的双线训练入口](provenance/native/train_mutual.py)
- [同预算开关对照的实验入口](provenance/native/ablation.py)
- [数据字段及输出接口](INTERFACE.md)
- [CPU 和 4090 的实测记录](provenance/verification.json)
