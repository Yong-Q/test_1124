# 低浓度 C₂H₄ 外势测试

[case.json](case.json) 固定输入指纹、随机种子和程序版本。用的是原 152 探针训练清单中的 `C2H4-source239-C2H4-PRL-T430.3-r00-0.000291`。分子数密度约 0.000291 Å⁻³，原盒约 120 Å，网格为 240³。这是当前训练集中网格最大的一条。原子坐标来自已有预处理缓存。

## 一键生成

从仓库根目录运行，第一项是你已有的 152 探针训练清单。清单里的缓存和标签路径相对清单解析。

```bash
WORKERS=12 bash cases/low_density_c2h4/reproduce.sh \
  ../xi_3d_charge_20261117_v1/experiments/xi_pnas_probes152/run_159379/training_split.json \
  runs/c2h4_low_density_3d
```

每个候选包含沿 (x,y,z) 的随机余弦项和周期三维平滑势井。对每个原子的实际三维坐标算势能，同一帧内求和，除以 (RT)，再做 Boltzmann 重加权。通过帧和连续时间块 ESS 筛选后，计算原 152 个探针上的一阶和二阶标签。对应 [报告第 3–7 页](../../docs/report.pdf#page=3) 的重加权及导数投影。命令在本地写出状态文件、场参数和供训练使用的相对路径清单。

## 复核

```bash
bash cases/low_density_c2h4/verify.sh runs/c2h4_low_density_3d --device cuda
```

复核从首尾状态的外势参数重算逐原子能量，检查周期平移、重加权和标签读回。再从同一低浓度体系取一个 5³ 小块，对 FFT 邻域 MLP、六层 CNN 和稳定核的方向导数作解析／自动微分对照。生成标签使用 240³ 原盒。

完整训练入口直接读输出目录中的 `training_split.json`，体相索引用原来的文件。新增状态仅写进这条训练轨迹，其他训练、验证和测试的分组及标签保留。

[生成验证记录](generation_verified.json) 记录 300 条状态及首尾标签复算。
[CPU 模型核对](model_verified_cpu.json) · [GPU3 模型核对](model_verified_cuda.json) · [实测软件包版本](../../verification/requirements-tested.txt)
