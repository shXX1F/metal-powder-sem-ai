# boost_round1_gh_hx 数据检查结论

## 1. 当前补强集是否完整

当前补强集：

```text
data/boost_round1_gh_hx
```

统计结果：

```text
训练图片：168 张
训练实例：45169 个颗粒
图片路径缺失：0
```

结论：

```text
它不是把 GH3536 和 HX 两个原始文件夹全部原样复制进去，
而是整理成了更适合 Mask R-CNN 训练的切片版补强集。
```

## 2. GH3536-SA0120724061801-H1 覆盖情况

原始 GH 数据：

```text
LabelMe 标注文件：90 个
切片标注图：80 张
整图标注图：10 张
原始标注实例总数：76665 个
切片标注实例：42310 个
整图标注实例：34355 个
```

当前 boost_round1_gh_hx 中：

```text
已纳入 GH 切片图：80 张
已纳入 GH 训练实例：42303 个
未纳入 GH 整图：10 张
```

说明：

```text
GH 的 10 张整图与 80 张切片来自同一批图像内容。
当前版本只使用切片图，避免一张整图包含几千个颗粒导致训练显存/内存压力过大。
少量实例被 min-area 过滤掉，是面积太小或无效 polygon。
```

## 3. HX-ZA4720724102201 覆盖情况

原始 HX 数据：

```text
LabelMe 标注文件：11 个
整图：11 张
原始标注实例：1920 个
```

当前 boost_round1_gh_hx 中：

```text
HX 已自动切成：88 张切片
已纳入 HX 训练实例：2866 个
```

说明：

```text
HX 的 11 张整图都已经参与处理。
由于切片之间有 overlap，同一个跨边界颗粒可能会在不同切片中出现，所以训练实例数会大于原始 1920 个。
当前训练不直接使用 HX 原始整图，而是使用 data/boost_round1_gh_hx/hx_tiles 里的切片图。
```

## 4. 火山 H100 训练需要上传哪些

必须上传：

```text
requirements.txt
app_streamlit.py
metal_powder_sem_ai/
tools/
data/GH3536-SA0120724061801-H1/
data/boost_round1_gh_hx/
已有模型权重 .pth
```

原因：

```text
data/boost_round1_gh_hx/annotations.json 里引用了 GH 原始切片图片路径。
HX 切片图已经保存在 data/boost_round1_gh_hx/hx_tiles 中。
```

可选上传：

```text
data/HX-ZA4720724102201/
```

只有当你想在火山上重新生成 boost_round1_gh_hx 时才需要上传 HX 原始目录。

固定评估集：

```text
data/fixed_eval_20/
```

如果训练后要直接在火山上评估模型，也建议上传。它不要参与训练。

## 5. H100 第一波推荐训练命令

把 `你的已有权重.pth` 换成火山里实际的权重路径。

```bash
REQ=$(find . -maxdepth 5 -name requirements.txt | head -n1) && cd "$(dirname "$REQ")" && python -m pip install -r requirements.txt && python -m metal_powder_sem_ai.train segment --train-images data --train-ann data/boost_round1_gh_hx/annotations.json --resume-from 你的已有权重.pth --epochs 16 --batch-size 2 --num-workers 4 --lr 0.0002 --augment --min-mask-area 4 --min-box-size 2 --output-dir /tos-mlp-zgci/runs/train_maskrcnn_boost_round1_h100
```

如果 H100 运行稳定、显存还有很多，可以下一轮尝试：

```text
batch-size 4
epochs 20
```

不建议第一轮直接把 GH 整图也加进去，因为整图和切片内容重复，而且单张整图实例数最高超过 5000，容易让训练变慢或不稳定。
