# 金属粉末 SEM 图像 AI 识别系统

本项目按给定国标逻辑实现：球形度 `Q = 4*pi*A/P^2`、球形颗粒轴比 `Lmajor/Lminor <= 1.2`、球形率 `S = n/N*100%`、空心粉孔洞占比 `>=25%`、团聚体“大球附着小球且小球直径 > 大球直径 0.5 倍”。

## 模块划分与数据流

1. `preprocess.py`：读取 SEM 图像，执行高斯滤波、CLAHE 增强、Otsu 阈值，输出二值图。
2. `segment.py`：实例分割。默认提供 `classical` 分割器用于快速跑通；训练后可用 `MaskRCNN` 权重推理。
3. `feature_extract.py`：对每个颗粒 mask/contour 计算面积、周长、Q 值、长短轴、轴比、孔洞占比。
4. `classify_stat.py`：执行规则判定，统计总颗粒数、球形颗粒数、空心粉数、团聚体数、球形率 S。
5. `visualize.py`：绿色=球形、红色=非球形、蓝色=空心粉、黄色=团聚体，叠加 mask、轮廓、标签。
6. `report.py`：导出 Excel，包含统计结果和颗粒级特征。
7. `train.py`：Mask R-CNN 实例分割训练，以及 XGBoost 几何特征分类器训练。
8. `infer.py`：端到端推理入口，输出可视化图和 Excel 报告。

## 安装依赖

```bash
pip install -r requirements.txt
```

Windows 下 `pycocotools` 如安装困难，可使用：

```bash
pip install pycocotools-windows
```

## 快速推理

无训练权重时，先用传统 CV 基线跑通流程：

```bash
python -m metal_powder_sem_ai.infer ^
  --image "D:/桌面/cvv/1.jpg" ^
  --pixel-size-um 0.1 ^
  --segmenter classical ^
  --crop-bottom-fraction 0.12 ^
  --output-dir runs/sample_classical
```

输出：

- `runs/sample_classical/visualized.png`
- `runs/sample_classical/report.xlsx`
- `runs/sample_classical/report.particles.csv`
- `runs/sample_classical/preprocess/*.png`

比例尺必须手动输入：若 `1 pixel = X um`，则传入 `--pixel-size-um X`。样图底部带 SEM 信息栏和 100 um 标尺，建议通过图像软件量取标尺长度后换算 X。

## 简易 GUI

项目已集成 Streamlit 界面，适合实验室人员上传图片、输入比例尺并一键导出报告：

```bash
conda activate sdsd_torch
python -m streamlit run app_streamlit.py
```

也可以直接双击 `run_gui.bat` 启动。

打开浏览器中的本地地址后，在左侧完成：

1. 上传 SEM 图像。
2. 输入比例尺：可直接输入 `1 pixel = X um`，也可输入标尺真实长度和图中像素长度自动换算。
3. 设置底部信息栏裁剪比例，样图可先用 `0.12`。
4. 选择分割方式：无权重时选 `classical`；有训练权重后选 `maskrcnn` 并填写 `.pth` 路径。
5. 点击“开始识别”，右侧会显示标注图、统计卡片、颗粒级表格，并提供 Excel、标注图、CSV 下载。

GUI 的每次运行结果会保存到 `runs/gui/时间戳/`，包含上传图、二值图、标注图和 Excel 报告。

## Mask R-CNN 训练

COCO 标注要求：每个颗粒为一个实例 mask，类别统一为 `particle`。训练命令：

```bash
python -m metal_powder_sem_ai.train segment ^
  --train-images data/train/images ^
  --train-ann data/train/annotations.json ^
  --val-images data/val/images ^
  --val-ann data/val/annotations.json ^
  --epochs 20 ^
  --batch-size 2 ^
  --output-dir runs/train_maskrcnn
```

训练输出：

- `maskrcnn_particle_last.pth`
- `loss_curve.csv`
- `loss_curve.png`（若安装 matplotlib）

使用训练权重推理：

```bash
python -m metal_powder_sem_ai.infer ^
  --image SEM_001.tif ^
  --pixel-size-um 0.1 ^
  --segmenter maskrcnn ^
  --weights runs/train_maskrcnn/maskrcnn_particle_last.pth ^
  --output-dir runs/SEM_001
```

## XGBoost 几何特征分类器训练

当已有颗粒级特征和人工标签 CSV 时，可训练辅助分类器。默认标签列：

- `is_spherical`
- `is_hollow`
- `is_agglomerate`

```bash
python -m metal_powder_sem_ai.train classifier ^
  --feature-csv data/particle_features_labeled.csv ^
  --output-dir runs/train_xgb
```

当前推理主流程严格使用规则判定，保证符合题设阈值；XGBoost 适合在后续扩展中做质量复核或疑似样本排序。

## 核心判定逻辑

- 球形度：`Q = 4*pi*A/P^2`
- 球形颗粒：`Lmajor/Lminor <= 1.2`
- 球形率：`S = 球形颗粒数 / 总颗粒数 * 100%`，用 GB/T 8170 的五留双规则保留两位小数
- 空心粉：`hole_area / particle_area >= 0.25`
- 团聚体：两个颗粒接触，且 `small_diameter > big_diameter * 0.5`
