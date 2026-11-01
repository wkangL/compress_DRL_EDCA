# TGax-based 7 BSS仿真环境说明书

更新时间：2026-10-26（补充当前训练拓扑数量审计）  
正式配置：`subset7`，7 AP、每AP 40 STA，共280 STA  
阶段边界：本阶段只验证Default EDCA环境和负载，不接入TCN、DDQN或压缩反馈控制。

## 1. 模拟的环境

正式环境取TGax Scenario 3室内小BSS场景的reuse-3同信道几何：中心BSS 0加六个最近的同信道邻居，共7个BSS。它是“参考TGax官方参数构建的7 BSS研究子集”，不是完整19 BSS场景。

- BSS 0是未来唯一智能控制对象，本阶段仍使用Default EDCA。
- BSS 1-6及全部STA始终使用Default EDCA。
- 每BSS恰好40 STA，7 BSS共280 STA。
- 7个AP都使用5 GHz、80 MHz、信道42，主20 MHz信道36。
- 最近同信道AP距离30 m；坐标来自TGax reuse几何生成器，没有为结果人工移动AP。
- 不启用UL/DL OFDMA和MU-MIMO；HE单用户传输、QoS、A-MPDU、Block ACK、重传和真实TCP/UDP保留。
- `main3`只用于快速协议调试；`full19`保留配置和历史结果，不安排正式实验。

正式拓扑来自`results/raw/tgax_s3_inputs_20261021/subset7/topology.csv`。STA是在61个TGax参考小区内全局投放后，按包含5 dB静态阴影的最强接收功率关联，因此属于某BSS的STA不要求落在以该AP为中心的10 m参考六边形内。六边形不是覆盖边界。运行时STA通过共同SSID接收Beacon并自行关联，随后对账实际BSSID。

下图同时核对了三个不同数量：拓扑CSV固定为每BSS 40个STA；最新seed 45运行时每BSS均为40/40成功关联且BSSID匹配；当前`candidate_voip8_gaming4_ftp_bk`中实际承载至少一个应用的不同STA为19–22个/BSS，因为多类业务可以随机重叠到同一STA。图中的20 m是地图长度比例尺；中心AP到任一邻居AP的真实距离为30 m。

![7 BSS当前拓扑与数量审计](../results/figures/tgax_current_topology_audit_20261026/current_subset7_topology_verified.png)

## 2. 位置、关联和传播

候选STA先在TGax Scenario 3的61个参考六边形内均匀投放。对每条候选链路计算三维距离、Model-D大尺度路径损耗和5 dB链路静态阴影，再为每个入选BSS保留40个以该AP为最强AP的STA。运行时7个AP使用共同ESS SSID `tgax-s3`。

大尺度路径损耗为：

```text
PL = 40.05 + 20 log10(5/2.4)
     + 20 log10(min(d,10))
     + I(d>10) * 35 log10(d/10) + Xsigma
Xsigma ~ N(0, 5^2) dB
```

正式批量评估使用YansWifiChannel执行标量同信道干扰，仍然运行ns-3真实HE PHY/MAC；SpectrumWifiPhy保留用于短时帧交换和频谱交叉核验。ns-3.44不能直接表达TGax Model D的完整频率选择性多径，因此当前实现保留大尺度损耗与静态阴影，但不宣称严格复现全部小尺度信道抽头。

最终版5秒门禁中，280/280 STA均关联成功，280/280实际BSSID均与预计算最强AP一致。

![AP间接收功率与实测竞争](../results/figures/tgax_subset7_calibration_20261022/png/03_subset7_interference_and_competition.png)

## 3. PHY和MAC参数

| 参数 | 正式值 |
|---|---|
| Wi-Fi标准 | 802.11ax HE，单用户传输 |
| 频段/带宽 | 5 GHz / 80 MHz |
| 信道 | 中心信道42，主20 MHz信道36 |
| 数据/控制模式 | HE MCS 7 / OFDM 6 Mbit/s |
| GI | 0.8 us |
| AP天线与功率 | 2天线，1 TX SS/2 RX SS，总发射功率23.0103 dBm |
| STA天线与功率 | 1x1，15 dBm，-2 dBi增益 |
| CCA-SD / CCA-ED | -82 / -62 dBm；Scenario 3未明确，采用ns-3默认 |
| A-MPDU / A-MSDU | 每AC 65535 Byte / 0 |
| Block ACK | immediate，64 MPDU窗口 |
| RTS/CTS | 关闭，门限65535 |
| FrameRetryLimit | 10 |
| MAC队列 | 500包，最大寿命500 ms；官方未规定 |
| Beacon | 100 TU，ns-3独立初始jitter |

运行时Default EDCA导出值：

| AC | AIFSN | CWmin | CWmax | TXOP |
|---|---:|---:|---:|---:|
| VO | 2 | 3 | 7 | 2080 us |
| VI | 2 | 7 | 15 | 4096 us |
| BE | 3 | 15 | 1023 | 0 |
| BK | 7 | 15 | 1023 | 0 |

## 4. 每个BSS的官方100%业务

官方100%配置不再按3 BSS版本缩减。每BSS完整使用40 STA对应的业务数量：

| 行 | 模型 | 方向 | 数量/BSS | 协议 | AC |
|---|---|---|---:|---|---|
| D1 | BV6视频 | AP到STA | 12 | TCP | VI |
| D2 | BV3视频 | AP到STA | 8 | TCP | VI |
| D3 | FTP | AP到STA | 4 | TCP | BE |
| D4 | HTTP | AP到STA | 12 | TCP | BE |
| D5 | Gaming | AP到STA | 16 | UDP | VI |
| D6 | VoIP | 双向 | 12对，即24个方向 | UDP | VO |
| U1 | BV3视频 | STA到AP | 4 | TCP | VI |
| U2 | FTP | STA到AP | 4 | TCP | BE |
| U3 | Gaming | STA到AP | 16 | UDP | VI |

因此每BSS有100个单向配置组件，7 BSS共700个。D7由原生Beacon承担；U4的官方速率为TBD，没有虚构Probe Request应用。

VoIP保留33/7 Byte、20/160 ms的通话/静音模型和平均1.25 s状态持续时间；BV3/BV6保留Weibull帧长与Gamma网络抖动；Gaming、FTP和HTTP保留各自随机模型。视频、FTP、HTTP使用真实TCP；VoIP和Gaming使用真实UDP。TCP ACK由ns-3 TCP栈真实产生，不用手工小包代替。

扩展负载只改变独立业务实例数：`load025/050/075/125/150`按比例调整全部业务数量；`rt125/150`保持BE数量为100%，只增加VO/VI实例。`mix_min1/mix_min2`分别令每个BSS每种业务行为1/2个实例，用于补充较轻并发区间；它们不是官方比例。除了`load100`外均标记为“TGax-based extension scenario”，不称为官方原始配置。

![官方100%业务与STA分配](../results/figures/tgax_subset7_calibration_20261022/png/02_subset7_business_assignment.png)

## 5. 包如何经过Wi-Fi

应用经真实socket进入IPv4，ToS映射到VO、VI或BE，再进入对应QosTxop。EDCA执行AIFS、退避、TXOP、聚合和重传；普通MPDU由MAC ACK确认，A-MPDU中的每个MPDU由Block ACK位图确认。TCP ACK是承载在QoS DATA中的IP/TCP包，与802.11 MAC ACK和Block ACK分别统计。

summary模式关闭逐包文本、FlowMonitor XML和详细PHY时间线，保留聚合统计。full模式可按需启用详细应用、帧交换、退避和PHY状态审计。提速没有删除协议机制或邻居BSS。

## 6. 唯一正式时延口径

正式时延只使用“最终MAC服务时延”：

```text
同一MPDU首次到达WifiMac的仿真时刻
    -> 该MPDU最终收到成功MAC ACK或Block ACK的时刻
```

对成功入队的MPDU，`WifiMac::MacTx`和队列实际Enqueue位于同一个离散事件内，仿真时钟不前进，因此起点等同于队列入队时刻。统计按节点和Packet UID去重；A-MPDU逐MPDU读取`AckedMpdu`，不会把整个聚合帧只算一个样本。

账本分别记录：

- `acked`：最终成功并进入时延CCDF；
- `failed_enqueue`：入队前被拒绝；
- `expired_lifetime`：超过500 ms队列寿命；
- `retry_limit`：达到重试上限；
- `qos_old_packet`及其他最终失败；
- `unfinished`：测量期到达，但排空结束时仍未成功或最终失败；
- `tx_attempts/retransmissions`：每个MPDU实际PHY发送次数和重复发送次数。

每个BSS、方向和AC都必须满足：`arrived = acked + failed + unfinished`。失败或截止未完成包不伪装成低时延，也不进入成功时延CCDF；交付率、失败率和未完成比例必须与CCDF一起解释。

## 7. 最终正确性、CPU与负载标定

最终配对输入使用稳定逻辑流标识：相同业务在不同负载中保持同一STA、方向和应用随机数流。所有正式运行均通过280/280关联、BSSID匹配、应用数、MAC守恒、逐MPDU重传、分位数、队列和运行时EDCA审计。

| 官方100%测量窗口 | 并发 | 墙钟/s | 峰值内存/MiB |
|---:|---:|---:|---:|
| 5 s | 1 | 225.3 | 99.30 |
| 20 s | 1 | 987.8 | 106.92 |

20秒官方100%中心BSS的最终MAC服务结果为：DL VO/VI p99分别32.21/49.90 ms，UL VI为35.63 ms，DL/UL BE为514.26/454.57 ms。DL VO/VI成功率99.75%/99.96%，UL VI为100%。中心BSS的UL VoIP应用有生成字节，但没有形成QoS MPDU；现有summary只能确认阻塞发生在MAC入队前，直接原因仍需详细轨迹确认，所以官方100%被归类为严重过载压力点。

负载扫描补充了每BSS每种业务1路和2路的`mix_min1/mix_min2`。`mix_min1`是当前最低可用锚点，VO/VI双向均能进入MAC；`mix_min2`已出现明显拥塞转折；25%官方计数及以上均属于严重过载。绝对数值、逐方向VO/VI表和区间判断见`reports/TGax_7BSS负载标定报告.md`。

![中心BSS各AC上下行业务负载](../results/figures/tgax_subset7_calibration_20261022/png/04_target_ac_uplink_downlink_load.png)

![中心BSS最终MAC服务时延CCDF](../results/figures/tgax_subset7_calibration_20261022/png/05_target_final_mac_service_delay_ccdf.png)

![中心BSS VO和VI负载响应](../results/figures/tgax_subset7_calibration_20261022/png/06_target_vo_vi_load_scan.png)

## 8. CPU策略与实验边界

正式批量使用Yans标量干扰后端和summary输出。曾测试full19 Spectrum短测，内存和墙钟成本不适合当前正式扫描；该历史测试保留，但不属于7 BSS研究结论。7 BSS最终版5秒峰值内存约99.3 MiB，证明统计结构没有随详细文本轨迹爆炸。

当前结果只有一个环境seed，用于正确性、CPU基准和初始负载标定，不能当作多seed置信区间，也不能说明任何强化学习算法有效。本阶段全部节点均是Default EDCA。

## 9. 已知偏差

| 项目 | 当前处理 | 限制 |
|---|---|---|
| 场景规模 | 7 BSS中心子集 | 不是完整19 BSS或wrap-around场景 |
| Model D | 大尺度损耗+5 dB静态阴影 | 缺少完整频率选择性小尺度抽头 |
| 正式PHY后端 | Yans标量干扰 | Spectrum只作短时交叉核验 |
| 队列 | 500包/500 ms | TGax官方未明确；直接影响BE过期 |
| VoIP头压缩 | 普通IPv4/UDP | 未实现ROHC 3 Byte头 |
| Beacon长度 | ns-3原生管理帧 | 不保证恰为官方280 octets |
| MU功能 | 未安装MU scheduler | 结论仅适用于HE SU竞争环境 |

## 10. 结果位置

- 输入：`results/raw/tgax_s3_inputs_20261021/subset7/`
- 最终单进程5秒：`results/raw/tgax_subset7_benchmark_paired_5s_20261022/`
- 最终20秒：`results/raw/tgax_subset7_benchmark_paired_20s_20261022/`
- 配对负载扫描：`results/raw/tgax_subset7_loadscan_paired_5s_20261022/`
- 汇总：`results/summary/tgax_subset7_calibration_20261022/`
- 图表：`results/figures/tgax_subset7_calibration_20261022/`
- 进展：`reports/TGax实验进展.md`
- 官方逐项依据：`reports/TGax官方参数核对表.md`
- ns-3差异：`reports/TGax与ns3差异清单.md`

本阶段完成环境验证与Default EDCA标定后暂停，不启动DDQN训练。

## 11. 下一阶段唯一候选配置

2026-10-22新增`candidate_ftp_bk`作为待审核的唯一候选。它保持`subset7`拓扑和`mix_min1`业务数量不变，每BSS各保留一路BV6-DL、BV3-DL/UL、Gaming-DL/UL、FTP-DL/UL、HTTP-DL以及一对双向VoIP。应用AC固定为VoIP→VO、BV3/BV6/Gaming→VI、HTTP→BE、FTP→BK；仅FTP的应用优先级发生变化，业务随机模型和TCP行为不变。

BSS 0的40个STA全部关联，其中9个承载应用业务。5秒正确性和20秒Default EDCA均通过记账审计；20秒VO/VI p95为12.79至26.01 ms，BK上下行约446至451 ms。TCP纯ACK的实际QoS TID审计显示99.98%走BE。完整参数、定义、分配表和结果见`reports/TGax_7BSS唯一候选配置验证报告.md`。当前仍未启动DDQN。


