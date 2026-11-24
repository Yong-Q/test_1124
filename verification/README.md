# 验证范围

- `reference_cpu.json` 验证两个小例的输入形状及全部一、二阶标签复算。
- 最新逐原子外势验证见 [低浓度 C₂H₄ 用例](../cases/low_density_c2h4/README.md)。
- `smoke.sh OUTPUT_DIR DEVICE` 逐配置验证真实参数更新、冻结后的 NH₃ 验证和两方向解析／自动微分核对。小例用于数值与接口验证。
- [GPU 本地训练记录](gpu_training.json) 包含全部 8 个配置的参数更新、NH₃ 验证和两方向导数核对，使用本地 GPU3 通过 bash 启动。
- [CPU 训练记录](training_cpu.json) 验证原案例可直接训练。旧的探针组合诊断记录见 [历史小测](legacy_probe_smoke/)。
- [CH₄ 输出](example_predictions/ch4_fft_excess.npz) 和 [NH₃ 输出](example_predictions/nh3_fft_excess.npz) 保存同一次 FFT 过量小测的实际预测值、参考值与二阶探针索引。已逐项核对标签与案例保存的参考值一致。

[方法、报告与代码的逐项核对](method_alignment.md)
- [原盒 GPU 训练入口](full_grid_gpu.json) 使用项目真实预处理的相对路径清单。FFT 过量配置在 CH₄ 的完整网格更新一次，随后对留出的 NH₃、C₂H₆ 各一条算出全部 152×152 二阶条目。该内部原盒核对只发布统计。
- [实测软件包版本](requirements-tested.txt) · [环境检查记录](environment_cpu.json)。Torch 对应 CUDA 13.0，CPU 检查同样使用这一套安装。
