# 邻域 MLP

`model.py` 将 FFT 和稀疏邻居滤波配置到同一套球谐读出、方向微分和体相监督。运行例：

```bash
bash run.sh train fft-excess --case ch4 --output runs/fft
bash run.sh train fft-intrinsic --case ch4 --output runs/fft_total
bash run.sh train sparse-excess --case ch4 --output runs/sparse
```
