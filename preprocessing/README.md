# 三维外势重加权接口

`bash run.sh perturb --manifest <清单> --case <训练轨迹> --count 300 --output <目录>`。

模型训练的基本变量仍是 ξ。这个入口在已有 152 探针预处理缓存上算额外的逐原子势，外势参数用于重加权；网络使用加权状态及其响应方向。缓存文件由现有清单提供。输出包括 `background_fields.json`、`xi_response_labels.npz`、生成统计和可直接传入训练入口的相对路径清单。输出标签复用原帧顺序、格点以及 152 个方向。

数值实现依据与复算方法见 [低浓度最小用例](../cases/low_density_c2h4/README.md)。
