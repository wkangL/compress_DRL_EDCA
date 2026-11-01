# TGax Reward-v3与TCN-DDQN训练实验

更新时间：2026-10-28  
正式环境：`candidate_full40_multiac_v6`  
训练：seed 41，400个环境步，340次梯度更新  
独立验证：seed 44、45；留出测试：seed 46、47

## 1. 为什么替换旧Reward

旧Reward直接奖励重传率和最终失败下降。v6在200步验证时，策略因此偏向P10一类较保守的EDCA参数：减少发送尝试确实降低了重传和部分最终失败，但VO/VI数据在队列中等待数百毫秒。以seed 44为例，VO及时率从84.1%降至27.5%，VI从98.3%降至33.1%，旧Reward却仍高于Default。这是明确的优化目标错配。

Reward-v3固定为：

\[
R_t=0.40T_{\mathrm{VO},t}+0.40T_{\mathrm{VI},t}+0.10S_{\mathrm{all},t}+0.10J_t.
\]

- `T_VO`：20 ms期限内完成最终MAC ACK/Block ACK的VO MPDU比例。
- `T_VI`：50 ms期限内完成最终MAC ACK/Block ACK的VI MPDU比例。
- `S_all`：有外生应用需求AC的需求满足率均值，需求包含TCP发送缓冲及历史未交付债务。
- `J`：本窗口有需求STA的需求归一化交付比例的Jain公平性。
- 四项均限制在0～1；Reward中不再包含重传率、最终失败率或动作切换项。

截止期限账本跨1秒窗口保存。到期、最终失败或到期仍未完成的MPDU均进入及时率分母；窗口末尚未到期者结转。应用债务同样跨窗口结转，不重复生成。无截止样本或无需求证据时记0而不是满分。

为计算在线Jain，C++ STEP协议新增每个STA的独立生成需求和应用有效载荷交付增量`V`记录。短时实测中40个有需求STA均进入统计，逐STA汇总与原逐AC账本逐字节相等，且未改变业务、TCP/UDP或MAC行为。

## 2. 训练前离线正确性验证

历史v6日志具有精确的逐窗口截止期限和逐AC需求账本，但没有逐窗口逐STA账本。因此离线重算对`T_VO/T_VI/S_all`使用原始窗口记录，对`J`明确使用仿真结束的flow-summary Jain代理；新训练则使用新增的精确在线Jain。

| 旧训练seed | 验证seed | Default Reward-v3 | 旧DDQN@200 Reward-v3 | 判定 |
|---:|---:|---:|---:|---|
| 41 | 44 | 0.758 | 0.262 | Default更高 |
| 41 | 45 | 0.668 | 0.115 | Default更高 |
| 42 | 44 | 0.758 | 0.302 | Default更高 |
| 42 | 45 | 0.668 | 0.200 | Default更高 |
| 43 | 44 | 0.758 | 0.232 | Default更高 |
| 43 | 45 | 0.668 | 0.122 | Default更高 |

六组配对全部通过预定条件：旧DDQN即使降低重传，也无法凭严重恶化的及时率获得更高Reward-v3。Reward单元测试和后端账本冒烟随后通过，才启动新训练。

离线原始表：`results/summary/tgax_reward_v3_offline_20261027/reward_v3_offline_validation.csv`。

## 3. 训练过程与数值稳定性

除Reward外，场景、TCN、STA聚合器、Branching DDQN、865个实际动作、学习率、Replay、探索率和soft Target更新均与v6一致。Q网络和Target网络从随机初始化开始，没有继承旧Q值。

| 项目 | 结果 |
|---|---:|
| 环境步 / 梯度更新 | 400 / 340 |
| Q-loss前20次均值 | 0.07744 |
| Q-loss后20次均值 | 0.01016 |
| 最大Q-loss | 0.15140 |
| Online/Target Q绝对值最大值 | 1.25551 |
| 最大梯度范数 | 0.57407（裁剪阈值10） |
| NaN / Inf / 发散 | 无 |

Q-loss下降只能说明TD拟合数值稳定，不能单独证明MAC策略有效。训练诊断见`results/figures/tgax_ddqn_reward_v3_seed41_400step_20261028/training_diagnostics.png`。

## 4. 独立验证与checkpoint选择

下表为seed 44、45均值。安全门槛同时要求VO/VI及时率、有效吞吐和Jain公平性不能明显低于配对Default。

| checkpoint | Reward-v3 | VO及时率 | VI及时率 | Jain | 两seed均通过门槛 |
|---:|---:|---:|---:|---:|---|
| Default | 0.6997 | 74.40% | 98.09% | 0.1964 | — |
| 100步 | 0.6997 | 74.40% | 98.09% | 0.1964 | 是 |
| 200步 | 0.7089 | 76.31% | 98.46% | 0.1259 | 否 |
| 300步 | 0.6983 | 73.94% | 98.15% | 0.1402 | 否 |
| 400步 | 0.7016 | 74.65% | 98.29% | 0.1282 | 否 |

100步贪心策略在两个验证seed的20个窗口中均选择动作`[16,0,0,0,0]`，即独立的`Default EDCA + 原生BEB`回退动作，因此它不是智能优化，只是安全地退回Default。200、300和400步虽然部分提高VO/VI及时率或Reward，但都使Jain低于Default的80%，并导致BE或BK吞吐低于门槛。按预先声明的规则，最终选择100步checkpoint。

独立验证曲线见`results/figures/tgax_ddqn_reward_v3_seed41_400step_20261028/independent_validation.png`。

## 5. 留出seed 46/47的MAC结果

| 指标（两seed均值） | Default | Selected-100 | Final-400 | Final相对Default |
|---|---:|---:|---:|---:|
| Reward-v3 | 0.70802 | 0.70802 | 0.70807 | +0.01% |
| VO 20 ms及时率 | 73.86% | 73.86% | 74.79% | +1.26% |
| VI 50 ms及时率 | 99.77% | 99.77% | 99.54% | -0.23% |
| VO聚合MAC服务时延p95 | 24.49 ms | 24.49 ms | 34.25 ms | +39.86%（恶化） |
| VI聚合MAC服务时延p95 | 22.26 ms | 22.26 ms | 23.97 ms | +7.64%（恶化） |
| VO成功吞吐 | 0.2712 Mbps | 0.2712 Mbps | 0.2686 Mbps | -0.96% |
| VI成功吞吐 | 0.2074 Mbps | 0.2074 Mbps | 0.2032 Mbps | -2.03% |
| BE成功吞吐 | 0.02860 Mbps | 0.02860 Mbps | 0.00171 Mbps | -94.03% |
| Jain需求满足公平性 | 0.2266 | 0.2266 | 0.1909 | -15.78% |
| 应用有效载荷/生成需求 | 4.70% | 4.70% | 4.21% | -10.45% |
| MAC最终失败率 | 18.44% | 18.44% | 5.95% | -67.73% |
| MAC重传率 | 77.79% | 77.79% | 72.31% | -7.05% |

BK在Default和策略下仍常无成功样本，因此不能用其低时延或p95评价改善；表中不报告无效BK时延。Selected-100与Default完全一致。Final-400在seed 46 Reward下降1.27%，在seed 47提高1.36%，没有稳定方向；均值几乎不变，却显著牺牲BE、总体应用满足和公平性。它降低最终失败与重传，但这不是Reward-v3的直接奖励，说明动作仍通过更保守的接入行为改变竞争结构。

留出明细及改善百分比：`results/summary/tgax_ddqn_reward_v3_seed41_400step_20261028/heldout_core_metrics.csv`。核心图：`results/figures/tgax_ddqn_reward_v3_seed41_400step_20261028/heldout_core_metrics.png`。

## 6. 学到的EDCA/CW行为

- 100步：固定选择原生Default回退，安全但没有学习收益。
- 200步：几乎固定P5，并使用`alpha=1、beta1=2.0、beta2=2.0、gamma=5`。P5把BE/BK的AIFSN、CWmin显著调大，低优先级服务下降。
- 300步：主要在P1/P3间切换，偶尔P2/P11；不同状态开始产生Profile变化，但没有通过公平性门槛。
- 400步：主要选择P9，少量P8/P5/P15，仍使用上述保守CW增长组合。P9的BE/BK分别为`AIFSN=5/9、CWmin=63/127`，明显比Default的`3/7、15/15`保守，解释了BE吞吐骤降。

因此Reward-v3成功阻止了旧P10策略造成的VO/VI及时率崩溃，却没有阻止策略为了小幅提高高优先级及时率而牺牲BE/BK与公平性。原因不是训练数值不稳定，而是当前四项的实际可变尺度不平衡：VO/VI合计权重80%，而在高债务场景中`S_all`和`J`本身数值较小，各10%的权重不足以区分严重的低优先级饥饿。

## 7. 结论

1. **Reward-v3纠正了旧Reward最严重的目标错配。** 旧策略的VO/VI及时率崩溃在离线重算中得到明确惩罚，新训练各checkpoint不再出现同等级退化。
2. **单seed 400步仍未证明DDQN优于Default。** 唯一通过安全门槛的100步checkpoint等价于原生Default；后续checkpoint均因公平性和BE/BK服务退化被拒绝。
3. **不能依据Q-loss下降宣称收敛或算法有效。** 网络训练稳定，但最终策略的留出Reward几乎不变，BE吞吐下降94%，Jain下降15.8%。
4. **本轮按计划停止。** 不启动多seed训练，也不在本轮继续修改Reward、网络或动作空间。下一阶段若继续，应先研究`S_all/J`的尺度与低优先级硬约束，而不是简单增加训练步数。

## 8. 交付位置

- 原始训练、Replay状态和checkpoint：`results/raw/tgax_ddqn_reward_v3_seed41_400step_20261027`
- 汇总CSV：`results/summary/tgax_ddqn_reward_v3_seed41_400step_20261028`
- PNG图：`results/figures/tgax_ddqn_reward_v3_seed41_400step_20261028`
- 本报告：`reports/TGax Reward-v3与TCN-DDQN训练实验.md`
