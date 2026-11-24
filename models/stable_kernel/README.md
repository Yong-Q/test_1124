# 稳定核

解析基底先形成总内禀导数设计矩阵，再解有界最小二乘。受限修正版增加局部曲率修正。

```bash
bash run.sh train stable-base --case ch4 --output runs/stable_base
bash run.sh train stable-bounded --case ch4 --output runs/stable_bounded
```
