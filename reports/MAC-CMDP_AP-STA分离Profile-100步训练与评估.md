# MAC-CMDP AP/STA分离Profile：100步训练与评估

## 1. 实验配置

- 场景：单BSS、1 AP、40 STA、126条低负载直接MAC业务流。
- 训练：seed 42，随机初始化，10个episode×10步，共100个环境步、40次梯度更新。
- 评估：未参与训练的seed 66，20 s Default/Final配对评估。
- 动作：`(AP Profile, STA Profile, alpha, beta1, beta2, gamma)`，合法动作15553种。
- AP Profile只作用于AP队列；STA Profile统一作用于40个STA；四个CW增长参数由AP和STA共享。
- Reward、五项方向Cost、预算、TCN、聚合器、学习率与soft Target更新均未改变。

本轮只改变Profile控制粒度，因此可用于判断区分AP/STA接入参数是否具有额外优化能力。评估只有一个训练seed和一个评估seed，结论仍属于机制验证而不是多seed统计结论。

## 2. 训练数值稳定性

训练最终Q-loss均值为0.002220，最后一次Q-loss为0.002209；Q值范围为0.232—0.613，梯度范数为0.0526。训练中没有NaN、Inf、Q值爆炸或梯度异常。最终checkpoint固定为step 100，没有进行中途checkpoint择优。

五个最终训练episode Cost乘子为：VO-DL 0.0875、VO-UL 0.0837、VI-DL 0.0827、VI-UL 0.0806、Starvation 0.1100。高优先级Cost均处于有限范围；Starvation Cost仍接近1，说明低优先级饥饿问题没有由本轮训练解决。

## 3. Default与Final核心结果

| 指标 | Default | Final | 相对变化 |
|---|---:|---:|---:|
| Reward | 0.2703 | 0.5274 | +95.15% |
| Utility | 0.5643 | 0.7408 | +31.27% |
| VO 20 ms及时率 | 53.24% | 80.11% | +50.48% |
| VI 50 ms及时率 | 27.75% | 72.91% | +162.77% |
| VO需求满足度 | 97.13% | 99.22% | +2.15% |
| VI需求满足度 | 77.38% | 98.02% | +26.68% |
| Jain需求满足公平性 | 0.5515 | 0.6367 | +15.45% |
| VO成功吞吐量 | 0.1894 Mbps | 0.1935 Mbps | +2.15% |
| VI成功吞吐量 | 51.9402 Mbps | 65.7960 Mbps | +26.68% |

### 分方向高优先级MAC结果

| 方向 | p95 Default→Final | 最终失败率 Default→Final | 重传率 Default→Final |
|---|---:|---:|---:|
| VO-UL | 123.00→48.15 ms | 5.86%→0.82% | 70.92%→51.58% |
| VO-DL | 36.37→35.64 ms | 0.91%→0.02% | 37.87%→17.44% |
| VI-UL | 478.28→230.51 ms | 19.40%→0.82% | 48.94%→39.73% |
| VI-DL | 109.78→63.91 ms | 5.03%→5.01% | 7.99%→5.25% |

四个高优先级方向的p95均未退化，其中VO-UL、VI-UL和VI-DL下降明显。重传率只是逐MPDU重传次数/传输尝试次数，当前trace不能把重传可靠归因为碰撞。

## 4. AP/STA动作选择

20个Final评估决策窗口中：

- AP Profile：P10选择19次（95%），P4选择1次（5%）。
- STA Profile：P1选择15次（75%），P0选择4次（20%），P12选择1次（5%）。
- 共享CW参数：除第一个窗口外，稳定选择 `alpha=3、beta1=2.0、beta2=2.8、gamma=7`。

因此Final不是把同一套Profile复制到AP和STA，而是形成了“AP主要P10、STA主要P1”的非对称接入策略。`edca_step_runtime.csv`记录了每个决策窗口真实应用到AP及40个STA的EDCA参数；此前独立门禁也已验证AP=P3、STA=P8时164项运行参数全部匹配配置。

## 5. 低优先级限制

BE成功吞吐由0.00028增至0.00070 Mbps，BK由0增至0.00014 Mbps，但绝对服务量仍极低。Final仅有5个BE-UL成功MPDU和1个BK-UL成功MPDU，其他低优先级方向没有成功样本。因此：

- 不能依据低样本p95或百分比变化宣称BE/BK已改善；
- 新CCDF按规则不绘制少于20个成功MPDU的曲线；
- Starvation Cost仍为1，低优先级保护仍是下一阶段必须解决的问题。

## 6. 结论

在当前单seed机制验证中，AP/STA分离Profile策略相对Default同时改善了VO/VI及时率、四方向p95、成功吞吐、最终失败率、重传率和Jain公平性，说明分离上下行竞争参数具有明确的优化潜力。它也避免了此前统一Profile策略中某个下行方向明显退化的现象。

但该结论尚不能证明稳定泛化：目前只有seed 42训练和seed 66评估，而且低优先级业务仍严重饥饿。下一步应保持配置不变增加独立训练/评估seed，重点验证AP=P10、STA=P1这一非对称行为是否可重复，而不是立即继续扩大动作空间。

## 7. 结果位置

- 原始训练与评估：`results/experiments/mac_cmdp_five_cost_split_100step_seed42_eval66_20261102/`
- 汇总CSV：上述目录的 `summary/`
- 论文风格PNG：上述目录的 `figures/`
- AP/STA动作分布：`figures/ap_sta_profile_action_distribution.png`
- 高优先级CCDF：`figures/mac_service_delay_ccdf_high_priority.png`
- 改善百分比汇总：`figures/improvement_percentage_summary.png`
