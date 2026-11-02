# MAC-CMDP三Cost 300步训练与独立评估

## 1. 实验范围

- 环境：单BSS，1 AP + 40 STA，802.11ax HE SU，80 MHz，HE MCS 7。
- 业务：VoIP UL 40 / DL 20，Gaming UL 80 / DL 40，BV3 UL 10；其余BE/BK业务沿用实验YAML。
- 数据路径：直接MAC随机到达，`direct_high_watermark=0`，不存在自定义64包准入水位。
- 控制对象：AP及40个STA的EDCA Profile和研究型CW增长参数。
- 训练：seed 42，10个30秒episode，共300个1秒决策步；Replay warmup 64，完成240次梯度更新。
- 评估：只使用300步final checkpoint，在未参与训练的seed 66上进行20秒Default/Final配对评估。
- 冻结配置：`results/experiments/mac_cmdp_three_cost_300step_gaming80_40_seed42_eval66_20261102/generated_config/train.json`。

## 2. Reward与约束

每步优化目标为：

\[
R_t=U_t-\lambda_{\mathrm{avg}}Q_{\mathrm{avg},t}
-\lambda_{\mathrm{max}}Q_{\mathrm{max},t}
-\lambda_{\mathrm{starve}}C_{\mathrm{starve},t}.
\]

吞吐效用先对VO/VI/BE/BK需求满足度应用
\(f(S)=\ln(1+9S)/\ln 10\)，再按0.4/0.4/0.1/0.1加权。成功VO/VI MPDU的QoS Cost为
\(d/(d+D)\)，其中VO的\(D=20\) ms、VI的\(D=50\) ms；最终失败记为1，未完成MPDU跨窗口结转。四个方向为VO UL、VO DL、VI UL、VI DL，`qos_avg`取有效方向均值，`qos_max`取最差有效方向。低优先级饥饿Cost为
\(1-\min(S_{BE},S_{BK})\)。

预算依次为0.40、0.65、0.90。训练结束时乘子为0.11502、0.10937、0.11000。预算只驱动乘子更新，TD Reward使用完整Cost而不是仅使用超预算部分。

## 3. 训练稳定性

- Q-loss初期最高约0.036，随后下降并稳定在约0.002；最后一次为0.00202。
- 最后一次Online Q范围为0.249--0.564，梯度范数为0.0587。
- 未出现NaN、Inf、Q值爆炸或梯度异常。
- 训练本身数值稳定，但数值稳定不等于策略满足全部CMDP约束。

训练诊断图：`results/experiments/mac_cmdp_three_cost_300step_gaming80_40_seed42_eval66_20261102/seed_42/live_training_diagnostics.png`。

## 4. seed 66配对结果

| 指标 | Default | Final | 相对变化 |
|---|---:|---:|---:|
| Reward | 0.1439 | 0.3308 | +129.98% |
| CMDP Utility | 0.3869 | 0.5494 | +42.01% |
| QoS平均Cost | 0.6019 | 0.5425 | -9.87% |
| QoS最大方向Cost | 0.8285 | 0.6435 | -22.33% |
| 饥饿Cost | 1.0000 | 1.0000 | 无改善 |
| VO 20 ms及时率 | 40.90% | 45.31% | +10.79% |
| VI 50 ms及时率 | 19.65% | 34.71% | +76.61% |
| Jain需求满足公平性 | 0.4464 | 0.5603 | +25.53% |
| VO需求满足度 | 88.20% | 94.62% | +7.29% |
| VI需求满足度 | 51.15% | 81.44% | +59.22% |
| BE需求满足度 | 0.056% | 1.286% | 绝对值仍低 |
| BK需求满足度 | 0 | 0 | 完全饥饿 |

Final使`qos_max=0.6435`略低于0.65预算，但`qos_avg=0.5425`仍高于0.40，`starvation=1.0`仍高于0.90。因此不能把该策略称为满足全部约束的可行策略。

## 5. 方向级MAC表现

| 方向 | p95：Default→Final (ms) | 吞吐：Default→Final (Mbps) | 最终失败率：Default→Final |
|---|---:|---:|---:|
| VO UL | 218.40 → 151.73 | 0.2406 → 0.2652 | 11.99% → 0.49% |
| VO DL | 46.95 → 102.21 | 0.1160 → 0.1175 | 12.55% → 11.18% |
| VI UL | 488.93 → 450.92 | 31.650 → 54.387 | 39.72% → 13.11% |
| VI DL | 197.52 → 237.44 | 6.608 → 6.528 | 13.88% → 14.30% |

主要收益来自上行：UL-VO失败率大幅下降，UL-VI吞吐提高71.84%、失败率下降67.00%。代价是DL-VO p95增加117.69%，DL-VI p95增加20.21%，且DL-VI吞吐和失败率略微退化。虽然最大方向Cost下降并达到预算，但该Cost是所有最终完成MPDU的连续均值，不等价于约束p95，因此不能阻止两个下行方向的尾时延恶化。

BE仅获得少量服务：UL-BE只有2个成功样本，DL-BE有277个；BK上下行均无成功样本。低优先级结果仍明显不可行，低样本形成的时延不得解释为改善。

## 6. 动作行为

20个贪心决策步的Profile分布为：P14 12次、P10 7次、P12 1次。`beta1`始终选择2.0，`gamma`始终选择6；alpha多数选择3，beta2多数选择2.4。策略不是固定单一Profile，但明显集中于P14/P10。

P10/P14相对Default显著增大VO/VI的AIFSN和CW，并改变TXOP。这与结果一致：大量上行STA的竞争和失败下降，但AP下行队列获得的尾时延保障不足。当前策略展示了状态相关切换，但尚未证明这种切换在多seed上可稳定泛化。

## 7. 结论

本轮三Cost设计成功降低了平均及最差方向QoS Cost，并显著改善上行吞吐、上行失败率、总体及时率、需求满足度和Jain公平性；训练数值稳定。但Final仍违反平均QoS和饥饿预算，DL-VO/DL-VI尾时延退化，BK完全没有服务。

因此，本轮证明了动作空间存在明显优化潜力，也说明加入`qos_max`能抑制最严重方向的平均QoS Cost；但单个seed66结果不足以宣称算法稳定优于Default，更不能宣称满足CMDP约束。下一步应先处理p95尾时延与BK不可行性，再进行多训练seed和多评估seed复核。

原始汇总：`results/experiments/mac_cmdp_three_cost_300step_gaming80_40_seed42_eval66_20261102/summary/final_checkpoint_vs_default_per_run.csv`。全部PNG位于同一实验目录的`figures/`。
