# SEM 金属粉末识别：标定、标注、训练流程

这份流程按你现在的目标整理：GUI 最后呈现 **总颗粒数、平均球形度、空心粉率、团聚率**。

## 1. 你最终要得到的四个数

- 总颗粒数 `N`：一张图或一批图中被识别出的颗粒实例数。
- 平均球形度 `Qmean`：先对每个颗粒算 `Q = 4 * pi * A / P^2`，再对所有颗粒取算术平均值。
- 空心粉率 `K`：`K = n_hollow / N * 100%`。
- 团聚率 `P_agglom`：优先用数量口径，`P_agglom = N_agglom / N * 100%`。PPT 中还有面积口径 `P_area = A_agglom / A_total * 100%`，当前报告也会一起给出，方便备用。

注意：GB/T 41978 的空心粉率严格要求截面显微图或 CT 判断内部空心。普通粉末表面 SEM 图只能做“可见孔洞/疑似空心”的近似判定。如果老师要求严格按国标，需要补充镶嵌、磨抛后的截面 SEM 图，或 CT 数据。

## 2. 做比例尺标定

先生成标定表：

```powershell
python tools/create_calibration_template.py --image-root "图像SEM数据" --output "data/sem_image_calibration.csv"
```

打开 `data/sem_image_calibration.csv`，每张图填三列：

- `scale_um`：图中标尺的真实长度，例如 `30`。
- `scale_bar_px`：用 ImageJ/Fiji、LabelMe 或 CVAT 量到的标尺像素长度。
- `pixel_size_um`：用 `scale_um / scale_bar_px` 计算。

同一批次、同一放大倍数的图，通常比例尺相同，可以先量一张，再复制到同批其他图片。你的图片底部有 SEM 信息栏，GUI 里建议先用 `crop_bottom_fraction = 0.12` 裁掉。

## 3. 做颗粒实例标注

推荐工具：CVAT 或 LabelMe。你如果不熟，先用 LabelMe 更轻。

标注规则：

- 每一个可见颗粒都画一个 polygon mask，标签写 `particle`。
- 边缘只露出一小部分、无法判断边界的颗粒，可以不标，或标成 `ignore_particle`。
- 粘连颗粒尽量按可见边界拆开，不要把一串颗粒画成一个整体。
- 团聚判断建议先人工记录：通常 3 个及以上颗粒聚集在一起算团聚体。
- 空心粉如果只看表面 SEM，要标成“疑似空心”；如果用截面 SEM/CT，才按国标空心粉来标。

这类图不建议一开始切成很多小图再标。先标整张图，避免把边缘颗粒切断；如果显卡显存不够，后面训练阶段再做切片或缩放。

标注时最容易错的地方：

- 小颗粒贴在大颗粒边缘：如果是独立小球，要单独圈一个 `particle`，不要并进大球。
- 破碎/不规则颗粒：只要是一个独立粉末颗粒，也标 `particle`。
- 阴影和背景亮斑：不要标。
- 图像底部的标尺和文字栏：不要标，训练图片建议裁掉底部信息栏。
- 看不清完整轮廓的边缘颗粒：第一版可以先不标，后续再按统一规则补。

如果觉得从零画太慢，可以先生成“机器初标”，再人工修正：

```powershell
python tools/create_labelme_starter.py --image-root "图像SEM数据" --output-dir "data/labelme_starter" --max-images 60
```

生成后的目录：

```text
data/labelme_starter/
  train/images
  train/labelme
  val/images
  val/labelme
  test/images
  test/labelme
```

这些 JSON 只是草稿，必须人工检查：删掉错圈的，补上漏圈的，修正粘连颗粒边界。

建议第一轮数据量：

- 先标 40-60 张图训练分割模型。
- 每种材料、每个批次都抽一些图，不要只标一个文件夹。
- 验证集按批次拆分，例如 80% 训练、20% 验证，同一批次尽量不要同时放进训练和验证。

## 4. LabelMe 转 COCO

目录建议这样放：

```text
data/
  train/
    images/
    labelme/
    annotations.json
  val/
    images/
    labelme/
    annotations.json
```

转换命令：

```powershell
python tools/labelme_to_coco.py --labelme-dir "data/train/labelme" --image-dir "data/train/images" --output "data/train/annotations.json"
python tools/labelme_to_coco.py --labelme-dir "data/val/labelme" --image-dir "data/val/images" --output "data/val/annotations.json"
```

如果你用 CVAT，直接导出 COCO Instance Segmentation 格式即可，不需要这个转换脚本。

## 5. 训练实例分割模型

项目里的训练集读取已经支持直接解析 COCO polygon 标注，Windows 下不再强依赖 `pycocotools`。如果你从 CVAT 导出，请优先选 polygon 实例分割格式。

训练命令：

```powershell
python -m metal_powder_sem_ai.train segment `
  --train-images "data/train/images" `
  --train-ann "data/train/annotations.json" `
  --val-images "data/val/images" `
  --val-ann "data/val/annotations.json" `
  --epochs 30 `
  --batch-size 2 `
  --output-dir "runs/train_maskrcnn"
```

训练完成后会得到：

```text
runs/train_maskrcnn/maskrcnn_particle_last.pth
```

## 6. 在 GUI 里使用模型

启动：

```powershell
python -m streamlit run app_streamlit.py
```

GUI 左侧选择：

- 分割方式：`maskrcnn`
- 权重路径：`runs/train_maskrcnn/maskrcnn_particle_last.pth`
- `1 pixel = X um`：填标定表里的 `pixel_size_um`
- 裁掉底部信息栏比例：先用 `0.12`

点击“开始识别”后，页面会显示：

- 总颗粒数
- 平均球形度 Q
- 空心粉率 K
- 团聚率 P_agglom

Excel 报告里还会保存每个颗粒的面积、周长、球形度、长短轴、孔洞占比、团聚判断，方便你回查。

## 7. 现在最建议你先做的事

1. 先用 `tools/create_calibration_template.py` 生成标定表。
2. 从每种材料里各挑几张图，先标 40-60 张。
3. 训练第一版 Mask R-CNN。
4. 用 GUI 跑几张图，看标注图哪里错得最多。
5. 回到标注集中补标错得多的情况，例如粘连、边缘颗粒、破碎颗粒、强阴影颗粒。

第一版模型不用追求完美，目标是让“分割颗粒边界”和“总颗粒数”先稳定下来。空心粉和团聚率后面再根据老师认可的判定口径做校准。
