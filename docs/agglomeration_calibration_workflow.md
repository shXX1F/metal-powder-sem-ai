# 团聚率校准流程

## 当前统计定义

- 数量团聚率：P_agglom = N_agglom / N_total * 100%
- 面积团聚率：P_area = A_agglom / A_total * 100%
- 默认团聚组：至少 3 个颗粒通过强接触关系形成同一连通组。

## 当前判定证据

1. 使用 SEM 标尺换算：tolerance_px = ceil(tolerance_um / pixel_size_um)。
2. 邻近颗粒必须满足以下任一强接触条件：
   - 间隙不超过物理容差，且接触弧占比较高；
   - 掩膜具有足够的重叠占比。
3. 仅有点接触或短距离靠近不会直接计入团聚。
4. 每次识别报告都会生成 report.agglomerate_pairs.csv 和 Excel 的“团聚判定证据”工作表。

## 真值校准步骤

1. 从不同倍率、不同粉末类型中选择至少 10 张图，建议包含 100 个以上候选颗粒对。
2. 批量评估时增加 --save-agglomerate-evidence，每张图会在
   agglomerate_evidence 目录生成一个 *.agglomerate_pairs.csv。
3. 当候选对过多时，先生成分层复核表：

    python tools/prepare_agglomeration_review_set.py outputs/fixed20/agglomerate_evidence
      --output-csv outputs/review/agglomeration_review_160.csv
      --accepted-target 80 --rejected-target 80
      --max-per-file-per-class 8 --seed 20260718

4. 在 manual_truth 列填写 1（真实团聚连接）或 0（正常接触/非团聚）。
5. 正负真值样本都要覆盖，建议各不少于 30 个；不确定项先留空。
6. 执行：

    python tools/calibrate_agglomeration_thresholds.py outputs/reviewed_reports
      --output-dir outputs/agglomeration_calibration

7. 将生成的 agglomeration_calibration.json 中推荐参数写入 GUI 默认值。

## 验收指标

- 颗粒对 Precision、Recall、F1；
- 每张图团聚率绝对误差（百分点）；
- 团聚面积率绝对误差（百分点）；
- 不同倍率分组误差，确认不存在随 pixel_size_um 改变的系统偏差。

在没有人工团聚真值前，当前结果属于“尺度统一、证据可追溯”的工程估计值，不能宣称为已完成计量验收。
