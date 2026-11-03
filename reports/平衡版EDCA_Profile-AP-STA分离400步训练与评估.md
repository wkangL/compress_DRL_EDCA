# 平衡版 EDCA Profile：AP/STA 分离 400 步训练与评估

## 1. 实验配置

- 场景：单 BSS、40 STA、126 条低负载直接 MAC 业务。
- Profile 表：`configs/profiles_v2_balanced.json`；P0 为 Default，P5–P15 包含不同程度的 BE/BK 保护配置。
- 动作：AP Profile 与 STA Profile 分离，四个自定义 CW 增长参数共享；动作编码为 `(AP Profile, STA Profile, alpha, beta1, beta2, gamma)`。
- 训练：seed 42，随机初始化，20 个 episode × 20 步，共 400 步；Replay warmup 后共 340 次梯度更新；epsilon 在 300 步衰减到 0.05。
- 评价：固定使用 400 步 final checkpoint，在未参与训练的 seed 66 上进行 40 s Default/Final 配对评价。
- Reward/Cost、TCN、聚合器、学习率与 soft Target 更新均未相对上一轮修改。

## 2. 训练稳定性

训练完成且未出现 NaN、Inf、Q 值爆炸或梯度爆炸。最终状态记录为 400 个环境步、340 次更新。实时训练诊断和完整 loss 序列保存在实验目录的 `seed_42/live_training_diagnostics.png` 与 `seed_42/losses.csv`。

该结果只能说明本次单 seed 训练在数值上稳定；不能仅凭 Q-loss 下降判定策略收敛或具有跨 seed 泛化能力。

## 3. seed 66 配对评价

| 指标 | Default | Final | 相对变化 |
|---|---:|---:|---:|
| Reward | 0.1999 | 0.3610 | +80.61% |
| CMDP utility | 0.4956 | 0.6188 | +24.85% |
| VO 及时率 | 0.5380 | 0.6563 | +21.99% |
| VI 及时率 | 0.2740 | 0.5132 | +87.32% |
| VO 需求满足度 | 0.9730 | 0.9852 | +1.26% |
| VI 需求满足度 | 0.7906 | 0.9369 | +18.50% |
| Jain 需求满足公平性 | 0.5268 | 0.5983 | +13.59% |
| VO 成功吞吐量 / Mbps | 0.1963 | 0.1988 | +1.26% |
| VI 成功吞吐量 / Mbps | 51.6139 | 61.1627 | +18.50% |
| BE 成功吞吐量 / Mbps | 0.00035 | 0.00077 | +120.00% |
| BK 成功吞吐量 / Mbps | 0 | 0.00035 | 从零变为非零 |

方向级结果显示：

- VO-UL p95 从 120.94 ms 降至 81.58 ms，最终失败率从 5.95% 降至 2.89%。
- VO-DL p95 从 36.51 ms 升至 46.07 ms，但最终失败率从 0.80% 降至 0.26%。因此 VO-DL 并非所有指标都改善。
- VI-UL p95 从 477.79 ms 降至 424.95 ms，最终失败率从 18.70% 降至 5.38%。
- VI-DL p95 从 124.43 ms 降至 114.28 ms，失败率基本持平（约 5.01%）。
- BE/BK 的绝对成功样本仍极少：BE-UL 仅 11 个成功 MPDU，BK-UL 仅 5 个成功 MPDU；不能把其低样本 p95 当作稳定服务改善。

## 4. BE/BK 保护动作是否有效

平衡 Profile 表使 Final 策略在 seed 66 中实现了非零 BK 成功吞吐，并将 BE 吞吐由 0.00035 Mbps 提高到 0.00077 Mbps。但绝对服务量仍接近零，BE/BK 需求满足度分别只有约 0.0368% 和 0.0090%。因此当前证据是“缓解了完全零服务”，还不能称为“解决了低优先级饥饿”。

AP/STA Profile 的训练选择分布见 `figures/ap_sta_profile_action_distribution.png`。该图可用于检查 P5–P15 的实际选择频率；动作选择频繁并不等价于有效服务，最终结论仍以配对 MAC 结果为准。

## 5. 与上一轮旧 Profile 表的关系

上一轮 100 步分离 Profile 实验在 seed 66 的 Final 结果中，VO/VI 及时率和吞吐提升幅度更大；本轮则使用 400 步训练、平衡 Profile 表和 40 s 评价窗口。由于 Profile 表、训练长度和评价时长同时变化，两轮结果不是只改变 Profile 表的严格消融，不能据此断言旧表一定优于新表。

本轮的明确收益是：在保持高优先级总体改善的同时，BE/BK 出现了更多实际成功服务；明确代价是 VO-DL p95 仍有退化，而且低优先级绝对服务量仍不足。

## 6. 结论与限制

本次 400 步训练、final checkpoint 评价及绘图流程已完整结束，测试通过。Final 在 seed 66 上显著改善总体 Reward、VI 服务、VO-UL、失败率和公平性，并让 BK 从零服务变为少量非零服务；但 VO-DL p95 退化，BE/BK 仍近乎饥饿。

当前只有一个训练 seed 和一个评价 seed，不能证明统计稳健性。若要确认平衡 Profile 表的真实贡献，下一步应保持 40 s 窗口和训练预算不变，对旧表与新表做相同 seed 的多 seed 配对消融。

## 7. 结果位置

- 汇总 CSV：`results/experiments/mac_cmdp_balanced_profiles_split_400step_seed42_eval66_20261102/summary/`
- PNG：`results/experiments/mac_cmdp_balanced_profiles_split_400step_seed42_eval66_20261102/figures/`
- checkpoint 与训练日志：`results/experiments/mac_cmdp_balanced_profiles_split_400step_seed42_eval66_20261102/seed_42/`
- 测试：`python -m pytest tests -q`，112 passed、7 skipped。
