# 六层三维 CNN

单尺度按 5 Å 首层球形卷积读入，双尺度另加 12 Å 元素密度支路。各项损失与报告相同。

```bash
bash run.sh train cnn-single-excess --case ch4 --output runs/cnn
bash run.sh train cnn-single-intrinsic --case ch4 --output runs/cnn_total
bash run.sh train cnn-dual-excess --case ch4 --output runs/cnn_dual
```
