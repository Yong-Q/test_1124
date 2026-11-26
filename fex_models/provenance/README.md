# 发布来源

- PPT 方向核模型来自 `xi_sognet_atomic/xi_readout_1000/short_spatial`。局域读出、径向、角向与短程解析导数的原实现均在发布包保留。
- 逐格点和纯相模型来自 `grid3d_supervision/full_frames/mixed_regional_grid/c1_bulk_c2`。双路、单向及带门控互教的主要训练入口存于 `native/`。
- 发布权重可用 CLI 从随机初始化训练。样例内的 `*_native_*` 是参考导数数组，用于核对计算路径。
- PPT 参考行按保存的探针清单选择首个均相状态。格点参考读取 2,001 帧处理清单及其参考标签。纯相保留两个可靠波段，第二链式项在双精度下核对两种等价收缩。
- 源轨迹按 CH₄ 训练与 C₂H₆ 留出分开。没有归档原始轨迹。
- `export.slurm` 从原研究工作区运行导出。发布后的 `check` 和 `train` 用已经冻结的样例，无需该工作区。
