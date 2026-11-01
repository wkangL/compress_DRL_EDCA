# MAC-CMDP seed 42 final checkpoint：180秒配对评估

## 评估设置

- 固定模型：训练seed 42的 `step_400.pt` final checkpoint；没有继续训练或修改模型。
- 独立评估seed：44、55、77。
- 每个seed分别运行180秒Default EDCA和180秒Final DDQN，外生业务需求按seed配对。
- 对照用20秒结果来自相同checkpoint及相同seed。
- 这是旧版CMDP Reward下的历史checkpoint评估；当前代码中新实现的Cost-v2不影响本次已保存策略。

## 180秒三个seed均值

| 指标 | Default | Final DDQN | 相对变化 |
|---|---:|---:|---:|
| CMDP Reward | 0.0357 | 0.3257 | +812.57% |
| Utility | 0.1824 | 0.4115 | +125.65% |
| VO 20 ms及时率 | 42.67% | 59.06% | +38.41% |
| VI 50 ms及时率 | 20.62% | 45.94% | +122.78% |
| VO需求满足度 | 91.80% | 97.65% | +6.37% |
| VI需求满足度 | 59.22% | 93.87% | +58.50% |
| BE需求满足度 | 0.389% | 6.141% | +1479.66% |
| BK需求满足度 | 0% | 1.070% | Default为零，Final仍很低 |
| Jain需求满足公平性 | 0.3419 | 0.4448 | +30.07% |
| VO成功吞吐量 | 0.3670 Mbps | 0.3903 Mbps | +6.37% |
| VI成功吞吐量 | 40.3327 Mbps | 63.9240 Mbps | +58.49% |
| BE成功吞吐量 | 0.0098 Mbps | 0.1579 Mbps | +1511.69% |
| BK成功吞吐量 | 0 Mbps | 0.0190 Mbps | Default为零，Final服务有限 |

Reward仍使用评估环境创建时的乘子，仅作辅助。及时率、需求满足度、吞吐、时延、最终失败率和Jain才是策略效果判断依据。

## 成功MPDU服务时延p95

下表是三个seed各自p95的均值，只包含最终成功ACK的MPDU，必须结合最终失败率与成功样本数解释。

| 方向/AC | Default | Final DDQN | 结论 |
|---|---:|---:|---|
| VO UL | 191.52 ms | 111.38 ms | 改善41.84% |
| VO DL | 50.40 ms | 88.42 ms | **退化75.44%** |
| VI UL | 487.18 ms | 397.16 ms | 改善18.48% |
| VI DL | 244.81 ms | 213.16 ms | 改善12.93% |

Final明显改善VO UL及VI尾时延，但VO DL仍是明确退化项，不能用整体VO及时率掩盖。

## 最终丢包率

| 方向/AC | Default | Final DDQN | 相对变化 |
|---|---:|---:|---:|
| VO UL | 11.286% | 0.090% | -99.20% |
| VO DL | 7.861% | 6.593% | -16.13% |
| VI UL | 37.181% | 4.910% | -86.80% |
| VI DL | 7.338% | 6.879% | -6.25% |

这些是MAC最终Drop，不是碰撞率。重传率仅作为机制诊断，图名也明确使用`retransmission_rate`。

## 20秒与180秒的BE/BK、HTTP/FTP样本

该配置中：BE UL对应FTP、BE DL对应HTTP、BK UL对应FTP、BK DL对应HTTP。因此可用各方向成功MPDU数判断长阅读间隔业务在测量期内是否真正出现。

| 策略/业务 | 20秒成功MPDU（3 seed合计） | 180秒成功MPDU（3 seed合计） |
|---|---:|---:|
| Default BE/FTP UL | 0 | 46 |
| Default BE/HTTP DL | 193 | 447 |
| Default BK/FTP UL | 0 | 0 |
| Default BK/HTTP DL | 0 | 0 |
| Final BE/FTP UL | 177 | 1027 |
| Final BE/HTTP DL | 642 | 6652 |
| Final BK/FTP UL | 0 | 19 |
| Final BK/HTTP DL | 43 | 890 |

180秒确实显著增加Final策略下HTTP/FTP有效样本，证明20秒会低估长间隔业务的可观测性。但Default的BK在180秒仍完全没有成功样本，说明这不只是窗口过短，而是Default下真实严重饥饿。Final的BK满足度从20秒的0.115%增至180秒的1.070%，仍不能称为充分服务。

BE满足度在Final下由20秒1.521%上升至180秒6.141%，表明FTP/HTTP的长状态周期会显著影响短窗口估计。Jain则从20秒0.5518降至180秒0.4448，说明延长窗口暴露了更多低优先级需求，公平性结论比20秒更严格。

## 结论

180秒结果确认seed 42 final策略的主要收益不是20秒偶然现象：VO/VI需求满足度、及时率、VI吞吐、UL尾时延和高优先级最终丢包均优于Default。策略同时让原本几乎不可见的BE/BK业务获得更多服务。

但它仍不能被认定为全面优于Default：

1. VO DL成功MPDU p95由50.40 ms恶化到88.42 ms；
2. BK满足度仅1.07%，仍有明显饥饿；
3. 180秒Jain只有0.4448；
4. 当前只有一个训练seed，尚不能证明训练稳定性。

因此适当结论是：**单训练seed的final策略在高优先级容量、及时交付和最终丢包方面具有稳定收益，但VO DL尾时延与低优先级公平服务仍未解决。**

## 输出位置

汇总CSV：

- `results/experiments/mac_cmdp_custom/summary_seed42_180s/final_checkpoint_vs_default_aggregate.csv`
- `results/experiments/mac_cmdp_custom/summary_seed42_180s/final_checkpoint_vs_default_per_run.csv`

PNG目录：`results/experiments/mac_cmdp_custom/figures_seed42_180s/`

- `throughput_by_ac_direction.png`
- `final_drop_rate_by_ac_direction.png`
- `mac_service_delay_ccdf_by_ac_direction.png`
- `final_mac_service_delay_ccdf_pooled.png`
- `retransmission_rate_collision_diagnostic.png`
- `jain_demand_fairness.png`
- `final_comparison_and_training_diagnostics.png`
