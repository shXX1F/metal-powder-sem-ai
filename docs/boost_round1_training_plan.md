# 第一波补强训练数据说明

## 1. 固定评估集

这 20 张图是固定评估集，不参与训练，只用于每次训练完成后对比模型效果。

位置：

```text
data/fixed_eval_20/images
```

清单：

```text
data/fixed_eval_20/manifest.csv
data/fixed_eval_20/exclude_from_training.txt
data/fixed_eval_20/reference_particles.csv
```

注意：`exclude_from_training.txt` 里的图以后不要放进训练集，否则评估结果会虚高。

## 2. 这次补强训练集

这次训练使用 GH3536 和 HX 两个数据集整理后的补强集。

训练图片根目录：

```text
data
```

训练标注文件：

```text
data/boost_round1_gh_hx/annotations.json
```

数据规模：

```text
168 张训练图
45169 个颗粒实例
```

来源处理方式：

```text
GH3536-SA0120724061801-H1：只使用已有切片图，不用整图，避免训练爆内存。
HX-ZA4720724102201：已自动切成 8 列窄图，并裁剪对应 polygon 标注。
```

生成脚本：

```text
tools/build_boost_training_set.py
```

重新生成命令：

```bash
python tools/build_boost_training_set.py --overwrite
```

## 3. 上传到火山训练时需要的内容

最省事做法：上传整个项目文件夹，但至少要包含这些：

```text
requirements.txt
app_streamlit.py
metal_powder_sem_ai/
tools/
data/GH3536-SA0120724061801-H1/
data/boost_round1_gh_hx/
已有训练权重 .pth
```

说明：

```text
data/boost_round1_gh_hx/annotations.json 中的 GH 图片路径指向 data/GH3536-SA0120724061801-H1。
HX 图片已经切到 data/boost_round1_gh_hx/hx_tiles。
所以火山训练必须同时能看到 data/GH3536-SA0120724061801-H1 和 data/boost_round1_gh_hx。
```

如果想在火山上重新构建补强集，再额外上传：

```text
data/HX-ZA4720724102201/
```

## 4. 火山训练命令

把 `你的已有权重.pth` 替换为火山里实际的权重路径。

```bash
REQ=$(find . -maxdepth 5 -name requirements.txt | head -n1) && cd "$(dirname "$REQ")" && python -m pip install -r requirements.txt && python -m metal_powder_sem_ai.train segment --train-images data --train-ann data/boost_round1_gh_hx/annotations.json --resume-from 你的已有权重.pth --epochs 12 --batch-size 1 --num-workers 2 --lr 0.0002 --augment --min-mask-area 4 --min-box-size 2 --output-dir /tos-mlp-zgci/runs/train_maskrcnn_boost_round1
```

## 5. 训练后固定评估

训练完成后，用固定评估集 `data/fixed_eval_20` 跑评估，和上一次结果对比：

```text
总颗粒数误差
小颗粒漏检情况
平均球形度偏差
团聚率变化
可视化图中明显漏检/误检位置
```

固定评估集不要改，补强训练集可以继续扩充。
