# 验证范围

- `reference_cpu.json` 验证两个小例的输入形状及全部一、二阶标签复算。
- `perturb_300_cpu.json` 记录同一 CH₄ 输入上的 300 条生成统计和首尾两行复算。生成的大文件在本地运行目录。
- `smoke.sh OUTPUT_DIR DEVICE` 逐配置验证真实参数更新、冻结后的 NH₃ 验证和两方向解析／自动微分核对。小例用于数值与接口验证。
- [GPU 本地训练记录](gpu_training.json) 包含全部 8 个配置的参数更新、NH₃ 验证和两方向导数核对，使用本地 GPU3 通过 bash 启动。
- [CPU 训练记录](training_cpu.json) 与 [生成数据读回训练记录](generated_input_training_cpu.json) 分别验证原案例和一键生成的最后一行可直接训练。
