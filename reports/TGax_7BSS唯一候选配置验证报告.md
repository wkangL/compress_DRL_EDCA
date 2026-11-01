# TGax-based 7 BSS唯一候选配置验证报告

日期：2026-10-22  
候选配置：`subset7 / candidate_ftp_bk / seed 20261021 / Default EDCA`

## 1. 本轮结论

候选环境已按要求固定并通过5秒正确性门禁和20秒Baseline审计。网络仍是7个同信道AP、每BSS 40个STA，共280个STA；BSS 0是未来唯一控制对象，本轮所有AP和STA均使用ns-3 Default EDCA，没有启动DDQN。

`candidate_ftp_bk`严格以`mix_min1`为基础，每BSS保留D1-D6、U1-U3各一路，其中双向VoIP拆成DL/UL两个单向组件，共10个单向应用组件。唯一业务优先级变化是FTP D3/U2从BE改到BK；FTP文件大小、阅读时间、TCP状态和随机流均未改变。该配置是TGax-based候选研究配置，不是TGax官方100%业务数量。

5秒和20秒运行均满足：280/280 STA关联正确、MAC账本守恒、四个AC均有真实活动、BK上下行均成功传输、VoIP上行进入MAC、TCP纯ACK实际AC可审计。20秒结果显示VO/VI得到较低时延，而BK承担了显著的低优先级服务代价，因此该环境同时具备实时业务保护和可优化的拥塞差异，可作为下一阶段唯一候选；是否正式锁定仍由本报告审核决定。

## 2. 网络与业务配置

| 项目 | 固定值 |
|---|---|
| 拓扑 | `subset7`；中心BSS 0及六个最近同信道邻居 |
| 节点 | 7 AP；40 STA/BSS；共280 STA |
| 信道 | 5 GHz、80 MHz、信道42；最近同信道AP间距30 m |
| PHY | 802.11ax HE SU、HeMcs7；禁用UL/DL OFDMA与MU-MIMO |
| MAC | ns-3 Default EDCA、A-MPDU、Immediate Block ACK、重传上限10 |
| 队列 | 每AC 500包、最大排队寿命500 ms |
| 协议 | VoIP/Gaming为UDP；BV3/BV6、FTP、HTTP为真实TCP |
| 控制对象 | 当前无智能控制；未来只控制AP 0，六个邻居AP保持Default EDCA |
| 仿真窗口 | 2 s关联预热 + 5或20 s测量 + 1 s排空 |

每BSS的应用配置如下。这里“一路”是一个随机业务生成器；双向VoIP包含DL和UL两个单向组件。

| 服务 | 方向 | 协议 | AC | 每BSS数量 |
|---|---|---|---|---:|
| D1 BV6 | DL | TCP | VI | 1 |
| D2 BV3 | DL | TCP | VI | 1 |
| D3 FTP | DL | TCP | BK | 1 |
| D4 HTTP | DL | TCP | BE | 1 |
| D5 Gaming | DL | UDP | VI | 1 |
| D6 VoIP | DL+UL | UDP | VO | 1对 |
| U1 BV3 | UL | TCP | VI | 1 |
| U2 FTP | UL | TCP | BK | 1 |
| U3 Gaming | UL | UDP | VI | 1 |

BSS 0共有40个关联STA，其中9个实际承载应用业务。STA 12同时承载双向VoIP，因此10个单向组件对应9个活跃STA。完整40行分配见[业务分配表](../results/summary/tgax_subset7_candidate_ftp_bk_20s_20261022/bss0_sta_business_assignment.md)及[CSV](../results/summary/tgax_subset7_candidate_ftp_bk_20s_20261022/bss0_sta_business_assignment.csv)。

![BSS 0业务STA分配](../results/figures/tgax_subset7_candidate_ftp_bk_20s_20261022/02_bss0_sta_business_assignment.png)

拓扑图依据实际输入坐标和关联关系生成。半透明六边形只表示TGax几何参考小区，不代表传播模型计算出的覆盖边界。

![候选配置实际拓扑](../results/figures/tgax_subset7_candidate_ftp_bk_20s_20261022/01_candidate_actual_topology.png)

## 3. 指标精确定义

- **MAC最终服务时延p95**：MPDU第一次触发发送节点`WifiMac::MacTx`至该MPDU获得成功ACK或Block ACK的时间；只对最终成功MPDU取p95。A-MPDU中的MPDU逐个统计，重传不重新开始计时。
- **成功交付吞吐量**：测量窗口内成功ACK的MAC入队字节数乘8，再除以测量时长。它是MAC成功服务速率，包含TCP数据及反向纯ACK，不等同于应用有效载荷吞吐量。
- **MAC最终丢包率**：`最终失败MPDU / 进入MAC的MPDU`。最终失败包括入队失败、队列寿命到期、达到重试上限、QoS旧包及其他最终失败；仿真截止未完成另列，不混入丢包。
- **重传率**：`首次发送之后的重复PHY发送次数 / 全部PHY发送尝试次数`。当前trace不能把每次失败可靠且唯一地归因为碰撞、信道误码或ACK超时，因此不报告伪精确的碰撞率，按要求使用可验证的逐MPDU重传率。
- **Jain需求满足公平性**：对9个有配置业务需求的STA，先计算`min(应用接收字节/应用生成字节, 1)`，再计算Jain指数。它衡量各需求STA满足比例是否均衡，不是各STA绝对吞吐量是否相等。

## 4. 5秒正确性门禁

5秒运行墙钟123.19 s、峰值工作集83.47 MiB，全部自动检查通过。BSS 0的40个STA全部正确关联；四个应用AC均进入MAC；VO上行有68个MPDU进入MAC；BK上下行都有成功ACK的MPDU；所有AC方向均满足`进入=成功+最终失败+截止未完成`。Jain需求满足公平性为0.819182。

原始运行及检查结果见[5秒运行目录](../results/raw/tgax_subset7_candidate_ftp_bk_5s_20261022/candidate_ftp_bk/seed20261021/default/)和[5秒验证报告](../results/summary/tgax_subset7_candidate_ftp_bk_5s_20261022/candidate_validation.md)。

## 5. 20秒Default EDCA Baseline

20秒运行墙钟440.11 s、峰值工作集103.66 MiB，全部自动检查通过。BSS 0结果如下；百分比按未四舍五入原始计数计算。

| AC | 方向 | MAC时延p95 / ms | 成功MAC吞吐 / Mbit/s | 最终丢包率 | 重传率 | 进入/成功/失败/未完成 MPDU |
|---|---|---:|---:|---:|---:|---:|
| VO | DL | 13.541 | 0.0137 | 0.569% | 33.794% | 527 / 524 / 3 / 0 |
| VO | UL | 22.886 | 0.0115 | 1.978% | 61.473% | 455 / 446 / 9 / 0 |
| VI | DL | 12.786 | 18.0146 | 0.0065% | 16.921% | 30,652 / 30,650 / 2 / 0 |
| VI | UL | 26.009 | 5.4930 | 0.192% | 42.145% | 9,921 / 9,902 / 19 / 0 |
| BE | DL | 67.866 | 0.1268 | 0% | 13.704% | 5,019 / 5,019 / 0 / 0 |
| BE | UL | 183.249 | 0.4708 | 0.565% | 32.694% | 19,650 / 19,539 / 111 / 0 |
| BK | DL | 445.898 | 0.5729 | 7.464% | 19.492% | 2,599 / 2,405 / 194 / 0 |
| BK | UL | 451.065 | 0.0104 | 34.483% | 38.636% | 29 / 19 / 10 / 0 |

![20秒Baseline指标](../results/figures/tgax_subset7_candidate_ftp_bk_20s_20261022/03_bss0_default_edca_metrics.png)

9个需求STA的Jain需求满足公平性为**0.887595**。其中DL FTP需求满足率75.62%，UL FTP仅2.05%；后者是公平性未接近1的主因，也说明BK上行在七BSS竞争下存在明确优化空间。VO、BV3/BV6与Gaming的应用字节满足率约98.16%至100%。

## 6. TCP反向ACK、BK和VoIP核实

PHY发射侧对纯TCP ACK按`节点+Packet UID`去重，再读取真实QoS TID确定AC。BSS 0在20秒窗口内观察到24,517个唯一纯TCP ACK MPDU，其中24,512个属于BE，占99.98%；其余仅5个分散在VI/BK。由此可确认批量TCP反向ACK实际主要使用BE。表中的BE-UL活动主要是TCP反向ACK及HTTP请求，不能解释成额外配置的BE上行业务。

FTP数据已实际映射至BK：20秒内BK-DL成功2,405个MPDU，BK-UL成功19个MPDU。VoIP上行也正常进入MAC：455个MPDU进入、446个成功、9个最终失败，没有截止未完成包。

## 7. 交付位置与阶段边界

- 精确指标：[bss0_ac_direction_metrics.csv](../results/summary/tgax_subset7_candidate_ftp_bk_20s_20261022/bss0_ac_direction_metrics.csv)
- 逐STA需求满足率：[bss0_sta_demand_satisfaction.csv](../results/summary/tgax_subset7_candidate_ftp_bk_20s_20261022/bss0_sta_demand_satisfaction.csv)
- 20秒自动验证：[candidate_validation.json](../results/summary/tgax_subset7_candidate_ftp_bk_20s_20261022/candidate_validation.json)
- 20秒原始结果：[Default EDCA运行目录](../results/raw/tgax_subset7_candidate_ftp_bk_20s_20261022/candidate_ftp_bk/seed20261021/default/)
- 候选配置声明：[tgax_s3_7bss_candidate_ftp_bk.json](../configs/tgax_s3_7bss_candidate_ftp_bk.json)

本轮没有启动DDQN、没有调整业务数量、没有做新负载扫描，也没有删除或覆盖旧实验结果。当前证据验证的是候选环境和Default EDCA基线，不能用于宣称智能算法有效。
