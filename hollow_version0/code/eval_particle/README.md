# eval_particle

这里放 particle 的评估/推理脚本。当前核心脚本是：

```text
infer_full_plus_tiles.py
```

它的策略是：

1. 对原图整图推理一次，得到 `full_preds`，作为主结果。
2. 将原图切成 3 列 x 2 行，共 6 张重叠小图。
3. 对每张小图单独推理，并把 tile mask 映射回原图坐标。
4. 只保留那些和整图结果、以及已加入结果交集很小的 tile 预测。
5. 用最终融合结果计算 IoU>=0.5 的 precision/recall/F1，并输出预览图。

推荐第一版参数：

```bash
python -m eval_particle.infer_full_plus_tiles \
  --src_dir /root/code/empty_heart/labelme/work/particle_test_2026.2.3GH5188_png \
  --out_dir /root/code/empty_heart/labelme/0_result/0_particle/first6/example-full05-tile02-tiles6 \
  --model /root/code/empty_heart/labelme/model/particle.pt \
  --full_score_threshold 0.5 \
  --tile_score_threshold 0.2 \
  --tile_cols 3 \
  --tile_rows 2 \
  --tile_w 1024 \
  --tile_h 960 \
  --tile_margin 64 \
  --tile_max_intersection_ratio 0.05 \
  --mask_threshold 0.5 \
  --no_export \
  --preview_limit -1
```

预览图颜色：

- 绿色：GT
- 红色：最终预测中的整图来源结果
- 蓝色：最终预测中的 tile 补充结果
