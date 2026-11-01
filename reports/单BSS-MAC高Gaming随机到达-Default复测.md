# 单BSS MAC高Gaming随机到达 Default复测

## 1. 实验目的与结论

本轮只在既有单BSS、40 STA、MAC直注随机到达环境中**大量增加Gaming并发数**，其余网络、EDCA、传播、包长分布和到达过程保持不变。没有使用TCP/UDP/IP，没有训练DDQN，也没有搜索EDCA/CW参数。

采用的高压扩展配置为：40个STA每个均配置2路Gaming上行和2路Gaming下行，共160路Gaming（80 UL、80 DL）。Gaming仍使用原TGax参考模型的随机包长和随机到达间隔；它不是Full Buffer。该业务数量属于TGax模型驱动的自定义压力扩展，不能称为TGax官方原始业务数量。

三组20 s Default EDCA正式运行全部通过正确性检查。场景已形成显著的高优先级随机到达竞争：

- 瞬时MAC队列非空STA平均31.84个，峰值40个；
- 信道忙比例由原随机到达场景的31.81%升至65.55%；
- VO上行成功MPDU服务时延p95由0.83 ms升至25.23 ms；
- VI上行p95由1.16 ms升至54.43 ms；
- VO/VI上行及时率分别降至70.76%和90.13%；
- BE/BK出现数百毫秒尾时延及严重失败、过期，低优先级已明显饥饿。

因此，“Gaming数量不足导致高优先级时延始终很低”的疑问已得到验证：大幅增加Gaming并发后，VO/VI尾时延和超期现象确实显著出现。不过，这一160路Gaming配置已经属于严重压力场景，不适合作为“各AC均衡服务”的唯一环境，更适合作为后续机制的高压/过载测试点。

## 2. 固定网络与业务配置

### 2.1 网络与MAC

| 项目 | 设置 |
|---|---|
| 网络规模 | 1 AP + 40 STA，复用原BSS0坐标 |
| PHY | 802.11ax HE SU，5 GHz，80 MHz，固定HE MCS7 |
| 接入 | ns-3原生Default EDCA与BEB |
| 保留机制 | AIFS、CW/Backoff、TXOP、A-MPDU、Block ACK、MAC重传 |
| 禁用 | OFDMA、MU-MIMO |
| 数据路径 | MAC直注，不安装IPv4栈，不经过TCP/UDP/IP/Socket |
| MAC准入 | 应用侧积压；每500 us随机化轮询；MAC队列高水位64包 |
| 时间 | 2 s关联预热 + 1 s业务预热 + 20 s测量 + 1 s排空 |
| 随机种子 | 41、42、43 |

数据对象若大于MSDU限制，会在应用侧拆成不超过1460 B的MSDU，再通过`WifiMac::NotifyTx`和`WifiMac::Enqueue`进入正常QoS队列。没有绕过EDCA或直接驱动PHY。

### 2.2 业务数量

| 模型 | 方向 | AC | 数量 | 到达过程是否保留 |
|---|---:|---:|---:|---|
| VoIP | UL/DL | VO | UL 40、DL 1 | 是，通话/静默状态与包长不变 |
| Gaming | UL/DL | VI | **UL 80、DL 80** | 是，随机包长与随机间隔不变 |
| BV3视频 | UL | VI | 1 | 是，帧大小与时间扰动不变 |
| HTTP | DL | BE/BK | BE 20、BK 16 | 是，页面、对象和请求过程不变 |
| FTP | UL | BE/BK | BE 12、BK 8 | 是，文件大小与阅读间隔不变 |

与上一版随机到达基线相比，唯一有意改变的是Gaming并发：从1路DL增至80路DL，并新增80路UL。每个STA都有2路Gaming UL和2路Gaming DL，所有40个STA均参与高优先级业务竞争。

## 3. 正确性与账本检查

seed 41短时冒烟和seed 41/42/43三组正式运行均通过以下门禁：

- 40/40 STA关联成功，BSSID全部正确；
- 未安装IPv4栈，运行时传输类型全部为`NONE`；
- VoIP、Gaming、BV3、FTP和HTTP均实际产生业务；
- VO/VI/BE/BK上下行均有MAC活动；
- `生成字节 = 已注入MAC字节 + 应用侧待注入字节`；
- 无重复准入；
- MAC成功、最终失败、过期和截止未完成账本守恒。

正式三seed平均每次生成108.65 MB业务，其中41.82 MB在有限窗口内进入MAC，66.83 MB仍停留在应用侧待注入。Gaming平均待注入37.50 MB，FTP平均待注入24.18 MB。这说明该压力点不仅存在MAC竞争，也存在MAC准入之前的持续需求积压；不能只依据成功MPDU时延评价整体业务体验。

## 4. 竞争强度与总体性能

| 指标 | 原随机到达 | 高Gaming 160路 | 变化 |
|---|---:|---:|---:|
| 总成功MAC有效载荷吞吐量 | 17.80 Mbps | 10.94 ± 1.84 Mbps | -38.54% |
| 信道忙比例 | 31.81% | 65.55% | +106.05% |
| VO-UL p95 | 0.83 ms | 25.23 ± 12.13 ms | +2939.54% |
| VI-UL p95 | 1.16 ms | 54.43 ± 24.98 ms | +4592.60% |
| VO-UL 20 ms及时率 | 99.94% | 70.76 ± 11.09% | -29.20% |
| VI-UL 50 ms及时率 | 100.00% | 90.13 ± 4.56% | -9.87% |

原场景报告中的“平均4.44”是旧统计口径：它统计瞬时`WifiMacQueue`非空的源节点且包含AP，不等于有业务STA数量，不能与本轮修正后的严格STA数直接比较。本轮修复后分别输出`active_stas`与`active_nodes`；严格STA口径为平均31.84、峰值40。作为参照，真正Full Buffer场景约为39.99个。

信道更忙但成功吞吐反而下降，主要因为大量小Gaming包使竞争和协议开销增多，同时最终失败、低优先级过期以及应用侧积压加重。这不是物理容量下降的单一指标，也不表示增加业务后网络“传得更多”。

![原随机到达与高Gaming对比](../results/figures/mac_arrival40_gaming160_default_20s_3seed_20261029/base_vs_gaming160.png)

![业务需求、MAC准入与积压](../results/figures/mac_arrival40_gaming160_default_20s_3seed_20261029/model_demand_and_mac_admission.png)

## 5. 分AC结果

以下p95只统计最终获得ACK/Block ACK的成功MPDU，因此必须同时阅读最终失败、过期和截止未完成比例。

| 方向/AC | 成功吞吐 Mbps | 成功MPDU p95 ms | 最终失败率 | 过期率 | 截止未完成 | 及时率 |
|---|---:|---:|---:|---:|---:|---:|
| UL VO | 0.200 | 25.23 | 23.86% | 0 | 0 | 70.76% |
| UL VI | 7.645 | 54.43 | 3.77% | 0.002% | 0 | 90.13% |
| UL BE | 1.592 | 431.68 | 47.22% | 35.60% | 3.91% | 未定义 |
| UL BK | 0.636 | 459.76 | 68.95% | 63.78% | 4.91% | 未定义 |
| DL VO | 0.0045 | 13.28* | 28.41% | 22.36% | 10.98% | 59.85% |
| DL VI | 0.575 | 25.65 | 3.83% | 3.28% | 2.50% | 92.87% |
| DL BE | 0.275 | 427.11 | 46.24% | 45.25% | 0 | 未定义 |
| DL BK | 0.011 | 245.99* | 92.14% | 91.67% | 0 | 未定义 |

`*`：DL VO和DL BK的p95仅有2个seed存在成功样本。seed 42的DL VO成功数为0、截止未完成32.93%；seed 42的DL BK成功数也为0。因此这些低p95绝不能解释成业务改善。

VO上行最终失败率已达到23.86%，说明Default EDCA在该压力点下无法保证所有高优先级小包及时完成。VI上行承担大部分成功吞吐，但其p95已跨过50 ms参考期限。BE/BK的p95接近500 ms队列寿命边缘，并伴随高失败/过期比例，属于明显饥饿。

![各AC上下行成功吞吐](../results/figures/mac_arrival40_gaming160_default_20s_3seed_20261029/direction_ac_throughput.png)

![VO和VI最终MAC服务时延CCDF](../results/figures/mac_arrival40_gaming160_default_20s_3seed_20261029/vo_vi_service_delay_ccdf.png)

![失败、过期与及时率](../results/figures/mac_arrival40_gaming160_default_20s_3seed_20261029/failure_expiry_and_timely.png)

![MAC队列非空与竞争强度](../results/figures/mac_arrival40_gaming160_default_20s_3seed_20261029/arrival_competition_diagnostics.png)

## 6. seed差异与解释边界

三seed总吞吐分别为8.88、12.40和11.54 Mbps，信道忙比例为63.10%、66.63%和66.93%。VO-UL p95分别为11.23、32.34和32.12 ms；VI-UL p95分别为26.38、62.61和74.30 ms。随机业务对象大小、到达相位和EDCA竞争共同造成明显方差，不能用单seed宣称固定收益。

seed 41中BV3有15.29 MB停留在应用侧，而seed 42/43基本完成准入，进一步说明单个大视频源和AP内部多AC调度会造成阶段性阻塞。报告同时保留原始逐seed数据，后续算法比较必须使用配对seed并检查应用侧积压，不能只比较成功样本时延。

## 7. 对后续研究的判断

1. **是否形成高优先级竞争：是。** 40个STA全部有VO/VI业务，Gaming 160路，严格队列活跃STA平均31.84、峰值40，VO/VI尾时延明显上升。
2. **Default是否仍表现很好：否。** VO-UL p95已到25.23 ms且最终失败23.86%；VI-UL p95为54.43 ms，均不再是近1 ms的轻载表现。
3. **是否适合固定EDCA/CW搜索：适合作为高压点，但约束必须完整。** 优化时必须联合考察VO/VI及时率、有效吞吐、失败/过期、应用侧未准入债务以及BE/BK最低服务，不能只降低成功样本p95。
4. **是否适合作为唯一正式场景：不建议。** 该点已使BE/BK严重饥饿且应用侧积压约61.5%，宜与原轻载随机到达和Full Buffer上界共同构成不同压力区间。若后续需要寻找拥塞转折点，应另行测试中间Gaming并发数；本轮未擅自开展扫描。

## 8. 结果位置

- 配置：`configs/mac_arrival40_gaming160_default.yaml`
- 输入构建：`scripts/build_mac_arrival40_gaming160_input.py`
- 运行脚本：`scripts/run_mac_arrival40_gaming160_baseline.py`
- 分析脚本：`scripts/analyze_mac_arrival40_gaming160.py`
- 原始结果：`results/raw/mac_arrival40_gaming160_default_20s_3seed_20261029`
- 汇总CSV：`results/summary/mac_arrival40_gaming160_default_20s_3seed_20261029`
- PNG图：`results/figures/mac_arrival40_gaming160_default_20s_3seed_20261029`

