# TGax高优先级并发场景Default EDCA Baseline分析

日期：2026-10-22  
新配置：`candidate_voip8_gaming4_ftp_bk`  
对照配置：`candidate_ftp_bk`

## 1. 结论

本轮严格只增加了VoIP和Gaming并发数，网络、PHY、传播、队列、Default EDCA及单条业务生成模型均未修改。新配置成功把BSS 0活跃业务STA从9个增加到22个，上行独立竞争STA从4个增加到14个，并使VO/VI的p95时延明显上升。

但该配置**没有通过候选环境门禁，不建议直接作为后续TCN-DDQN唯一固定环境**。20秒内BSS 0的BK上行FTP未完成TCP连接，没有应用生成字节和MAC MPDU；BK下行也只成功5个MPDU，成功MAC吞吐量仅0.000765 Mbit/s。HTTP应用同样没有完成连接并生成页面业务。VO上行最终丢包率达到9.89%，Jain需求满足公平性从0.8876降至0.6396。

因此，这套配置可以保留为“高优先级重压/饥饿压力测试”，但不满足本轮“VO/VI时延上升且BE/BK仍获得一定有效服务”的平衡目标。当前原`candidate_ftp_bk`仍是已通过验证的候选环境。

## 2. 唯一改动与业务分配

新配置是TGax业务模型驱动的自定义并发场景，不是TGax Scenario 3官方业务数量。

| 业务 | 方向 | AC | 原数量/BSS | 新数量/BSS |
|---|---|---|---:|---:|
| VoIP | 双向 | VO | 1对 | 8对 |
| Gaming | DL | VI | 1 | 4 |
| Gaming | UL | VI | 1 | 4 |
| BV6 | DL | VI | 1 | 1 |
| BV3 | DL | VI | 1 | 1 |
| BV3 | UL | VI | 1 | 1 |
| HTTP | DL | BE | 1 | 1 |
| FTP | DL | BK | 1 | 1 |
| FTP | UL | BK | 1 | 1 |

每BSS共有30个单向应用组件。原配置中未变化的业务、各业务第1路STA和随机流全部保持不变；新增7对VoIP及DL/UL各3路Gaming优先分配给未承载业务的STA。BSS 0达到22个不同活跃STA；邻居BSS在保留原分配的前提下，各自也增加13个独立活跃STA。

![BSS 0业务STA分配](../results/figures/tgax_subset7_candidate_voip8_gaming4_ftp_bk_20s_20261022/02_bss0_sta_business_assignment.png)

完整40行分配见[bss0_sta_business_assignment.md](../results/summary/tgax_subset7_candidate_voip8_gaming4_ftp_bk_20s_20261022/bss0_sta_business_assignment.md)。

## 3. 5秒正确性门禁与异常排查

通用协议和统计审计通过：280个STA全部关联，VO/VI/BE/BK均有MAC活动，VO上下行正常发送，30个组件/BSS与输入声明一致，TCP/UDP、ACK/Block ACK、逐MPDU重传和MAC守恒正常。墙钟时间205.57 s。

候选适用性门禁未通过：BSS 0的两路FTP应用生成字节均为0；MAC只观察到2个BK下行TCP连接控制MPDU，二者均因500 ms队列寿命到期，BK上行没有MPDU。邻居BSS已有BK成功交付，说明FTP→BK映射和协议实现有效。问题是中心BSS在同步启动的高优先级竞争下，低优先级TCP连接未能及时建立，而不是统计漏记。

为判断它是否只是5秒启动窗口不足，又执行了规定的20秒测量。20秒仍未恢复BK上行，因而确认是持续性服务饥饿。

## 4. 20秒BSS 0结果

配置为2秒关联预热、20秒测量、1秒排空、Yans HE PHY summary模式。墙钟798.11 s，峰值工作集98.68 MiB。通用账本审计PASS，但业务可用性门禁FAIL。

| AC | 方向 | MAC时延p95 / ms | 成功MAC吞吐 / Mbit/s | 最终丢包率 | 重传率 | 进入/成功/失败/未完成 MPDU |
|---|---|---:|---:|---:|---:|---:|
| VO | DL | 26.786 | 0.104845 | 0.876% | 47.600% | 3,995 / 3,960 / 35 / 0 |
| VO | UL | 36.095 | 0.060415 | 9.885% | 78.279% | 2,529 / 2,279 / 250 / 0 |
| VI | DL | 28.811 | 4.046472 | 0.145% | 16.757% | 9,642 / 9,628 / 14 / 0 |
| VI | UL | 34.832 | 4.702201 | 0.178% | 42.296% | 8,999 / 8,983 / 16 / 0 |
| BE | DL | 309.328 | 0.092114 | 1.640% | 10.554% | 3,902 / 3,838 / 64 / 0 |
| BE | UL | 396.270 | 0.065912 | 6.440% | 30.895% | 2,935 / 2,746 / 189 / 0 |
| BK | DL | 200.981 | 0.000765 | 28.571% | 0% | 7 / 5 / 2 / 0 |
| BK | UL | 无成功样本 | 0 | 无MAC入队样本 | 无发射尝试 | 0 / 0 / 0 / 0 |

BK-DL的p95比旧配置数值更低没有性能意义：新配置只有5个成功样本，而旧配置有2,405个成功样本。BK-UL的丢包率显示为0也不能解释成可靠，因为分母为0；正确结论是业务未进入MAC且没有服务样本。

![五类核心指标与活跃STA](../results/figures/tgax_subset7_candidate_voip8_gaming4_ftp_bk_20s_20261022/03_bss0_default_edca_metrics.png)

## 5. 与原候选20秒结果比较

| AC | 方向 | 原p95 / ms | 新p95 / ms | 变化 | 原丢包率 | 新丢包率 |
|---|---|---:|---:|---:|---:|---:|
| VO | DL | 13.541 | 26.786 | +97.81% | 0.569% | 0.876% |
| VO | UL | 22.886 | 36.095 | +57.71% | 1.978% | 9.885% |
| VI | DL | 12.786 | 28.811 | +125.33% | 0.0065% | 0.145% |
| VI | UL | 26.009 | 34.832 | +33.92% | 0.192% | 0.178% |
| BE | DL | 67.866 | 309.328 | +355.79% | 0% | 1.640% |
| BE | UL | 183.249 | 396.270 | +116.25% | 0.565% | 6.440% |
| BK | DL | 445.898 | 200.981* | 不可直接比较 | 7.464% | 28.571% |
| BK | UL | 451.065 | 无样本 | 服务消失 | 34.483% | 无入队样本 |

VO/VI时延上升目标已经实现。VI上下行仍保持低于0.2%的最终丢包率，VO下行低于1%；VO上行达到9.89%，不能再称为较低。上行独立业务STA从4个增至14个，说明新增上行竞争确实生效。

BE在MAC层仍有成功活动，但HTTP应用在20秒内生成字节为0，BE流量主要来自其他TCP业务的反向ACK和连接控制。BK下行FTP应用生成1,875,615 Byte、接收仅1,608 Byte，满足率0.086%；BK上行FTP生成和接收均为0。它们不满足“BE/BK仍能获得一定有效应用服务”的目标。

Jain需求满足公平性由**0.887595降至0.639550**，下降0.248045（相对下降27.95%）。这里对所有22个配置需求STA计算；未能建立连接、生成字节为0的配置需求STA按0满足度处理。

## 6. 是否适合动态EDCA与CW增长研究

它具有明显的VO/VI竞争压力和时延梯度，适合作为额外的严重压力或饥饿鲁棒性测试。但作为唯一训练和评估环境存在三项问题：

1. BK上行没有状态样本和有效交付，算法难以从连续性能变化中学习BK改善；
2. HTTP和FTP的TCP建连受阻，观测到的BE/BK负载并不稳定代表其应用需求；
3. VO上行丢包接近10%，场景已越过“高优先级时延上升但仍基本可服务”的平衡区间。

因此本轮不把它提升为固定TCN-DDQN环境，不修改Reward，也不启动训练。若后续继续寻找平衡配置，应在单独任务中减少VoIP或Gaming并发数量，再重复相同门禁；本轮按要求只测试这一套，不展开扫描。

## 7. 结果位置

- 新配置声明：[tgax_s3_7bss_candidate_voip8_gaming4_ftp_bk.json](../configs/tgax_s3_7bss_candidate_voip8_gaming4_ftp_bk.json)
- 5秒原始结果：[运行目录](../results/raw/tgax_subset7_candidate_voip8_gaming4_ftp_bk_5s_20261022/candidate_voip8_gaming4_ftp_bk/seed20261021/default/)
- 20秒原始结果：[运行目录](../results/raw/tgax_subset7_candidate_voip8_gaming4_ftp_bk_20s_20261022/candidate_voip8_gaming4_ftp_bk/seed20261021/default/)
- 新配置精确指标：[bss0_ac_direction_metrics.csv](../results/summary/tgax_subset7_candidate_voip8_gaming4_ftp_bk_20s_20261022/bss0_ac_direction_metrics.csv)
- 新旧逐指标对比：[old_vs_new_metrics.csv](../results/summary/tgax_subset7_candidate_voip8_gaming4_ftp_bk_20s_20261022/old_vs_new_metrics.csv)
- 自动门禁：[candidate_validation.json](../results/summary/tgax_subset7_candidate_voip8_gaming4_ftp_bk_20s_20261022/candidate_validation.json)

历史`candidate_ftp_bk`配置和结果未被覆盖或删除。本轮没有进行负载扫描、Reward修改或DDQN训练。
