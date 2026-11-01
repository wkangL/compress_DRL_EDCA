# 正式训练实时状态

运行：`diagnostic_target_soft001_seed11_20261020`

状态：complete

已保存阶段：6/6；经验：120；更新：500；目标网络同步：500。

独立评估阶段：[1, 2, 3, 4, 5, 6]；评估seed：[301]。

完整续训检查点：`train/checkpoint.pt`；阶段模型快照：`snapshots`（仅评估用）。

阶段评估：`monitoring/stage_*/eval`；实时曲线与对比图：`results/figures/运行名`。

阶段评估用于诊断，最终测试seed保持独立；单个训练seed及预算完成均不构成收敛证明。

更新时间：2026-10-20T14:22:13.885420+08:00
