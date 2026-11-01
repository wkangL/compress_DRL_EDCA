# 单BSS、40 STA、MAC Full Buffer Default EDCA基线实验

> 实验日期：2026-10-28  
> 阶段边界：本报告只验证独立的MAC直注Full Buffer环境和原生Default EDCA基线；未启动DDQN、未修改Reward，也未搜索EDCA/CW参数。

## 1. 实验目的与结论边界

本轮从现有TGax工程中单独建立`1 AP + 40 STA`环境。它复用原7 BSS场景中BSS 0的真实AP/STA坐标、TGax室内传播模型和802.11ax无线配置，但移除其余6个BSS。业务不经过TCP、UDP、IP或Socket，而是在正常Wi-Fi MAC入口持续补充合法QoS MSDU；后续EDCA排队、AIFS、随机退避、CW/BEB、TXOP、A-MPDU、Block ACK、MAC重传和PHY传输仍由ns-3.44原生协议栈执行。

这个环境用于隔离研究“单BSS内40个上行竞争STA的EDCA/CW机制”。其结果不能直接代表原TGax应用业务的端到端体验，也不能与7 BSS TCP/UDP实验混为一谈。

## 2. 网络与Default EDCA

| 项目 | 本轮设置 |
|---|---|
| 拓扑 | 1 AP、40个关联STA；坐标取原`subset7`的BSS 0 |
| PHY | 802.11ax HE SU，5 GHz，80 MHz，HE MCS 7 |
| 多用户能力 | OFDMA关闭，MU-MIMO关闭 |
| MAC | QoS EDCA、A-MPDU、Block ACK、重传全部保留 |
| 控制 | 全部设备使用ns-3原生Default EDCA与BEB；运行期不改参数 |
| 关联预热 | 2 s |
| Full Buffer预热 | 1 s，预热后才开始统计 |
| 测量/排空 | 20 s测量 + 1 s排空 |
| 正式seed | 41、42、43 |

Default EDCA的运行时导出值将在正式结果中与ns-3配置一起保存于每个seed的`edca_runtime.csv`，避免仅凭配置文件推断实际值。

## 3. MAC直注位置与协议真实性

### 3.1 注入路径

直注源为每个STA的指定AC构造合法MSDU，添加实验用LLC/SNAP类型`0x88b5`和仅供统计的Full Buffer标签，然后调用：

1. `WifiMac::NotifyTx`，保留与正常`WifiNetDevice::DoSend`一致的MAC入队观测；
2. `WifiMac::Enqueue(packet, AP地址, STA地址, QoS TID)`，进入`StaWifiMac`的普通QoS发送路径；
3. 由`QosTxop`和`WifiMacQueue`执行EDCA竞争、聚合、ACK/Block ACK和重传。

代码没有直接调用PHY发送，也没有绕过`QosTxop`。QoS TID分别映射至VO、VI、BE和BK；802.11序列号由正常MAC头构造路径产生，A-MPDU中的MPDU按UID逐个结算。

### 3.2 明确取消的路径

在`mac-saturation`模式下不安装`InternetStackHelper`，因此没有IPv4、TCP、UDP、Socket或FlowMonitor应用数据流。`fullbuffer_audit.csv`记录`ipv4_stack_installed=0`，`ip_flow_summary.csv`必须只有表头、无数据行。

## 4. Full Buffer业务与受控补充

| AC | STA覆盖 | 数量 | MSDU有效载荷模型 | 到达过程 |
|---|---|---:|---|---|
| VO | STA 0–19 | 20 | 固定33 B，保留TGax活跃VoIP小包特征 | Full Buffer水位补充，不再使用通话/静默过程 |
| VI | STA 20–39 | 20 | Gaming上行Largest-Extreme-Value，均值158 B、尺度26.2 B，裁剪64–1500 B | Full Buffer水位补充，不再使用随机到达间隔 |
| BE | STA 0–9、20–29 | 20 | 800–1500 B均匀分布 | Full Buffer水位补充，不使用HTTP页面到达 |
| BK | STA 10–15、30–35 | 12 | 1200–1500 B均匀分布 | Full Buffer水位补充，不使用FTP文件/阅读时间 |

40个STA全部具有且仅具有一种高优先级主业务：20个VO、20个VI；其中32个STA另有BE或BK，因此共有72个持续活跃的`STA×AC`源。FTP文件大小和HTTP页面大小没有被当作单个MAC帧，所有MSDU均不超过1500 B。

每个源使用低/高水位`32/64 packet`和`100 us`轮询：队列降至32包及以下时补到64包。高水位64与Block ACK窗口/A-MPDU最大MPDU数相容，避免把队列简单灌到500包上限；它仍会形成明确的饱和排队，因此本报告的MAC服务时延是“进入MAC队列到成功确认”的饱和服务时延，而非应用生成到交付的端到端时延。

## 5. 统计口径

- **成功MAC有效载荷吞吐量**：测量窗内逐MPDU成功ACK/Block ACK的实验有效载荷字节，不含LLC和MAC开销。
- **MAC最终服务时延**：MPDU第一次进入MAC至最终成功确认；重传不重新计时，仅统计成功样本。
- **最终失败率**：最终失败MPDU / MAC到达MPDU；另列队列寿命到期比例。
- **及时率**：VO在20 ms、VI在50 ms内成功确认的比例。分母包括在期限内成功、最终失败，以及到达期限仍未完成的数据；未到期限的数据跨时间继续结算。
- **Jain公平性**：同一AC内各饱和源成功有效载荷吞吐的Jain指数。
- **竞争强度**：各源队列非空时间比例、同时有任一上行待发AC的STA数，以及同时非空的源队列数。

成功样本的低时延不能单独视为性能好；所有时延结果必须和最终失败、过期及测量截止未完成比例一起解释。

## 6. 短时正确性门禁

2秒测量的seed 41冒烟测试通过全部门禁：

- 40/40 STA关联且BSSID匹配；
- 72个源覆盖VO/VI/BE/BK四类AC；
- MAC账本满足`到达=成功+最终失败+未完成`；
- 所有活跃源队列非空比例不低于95%；
- `ipv4_stack_installed=0`，无TCP/UDP/IP流量记录；
- 重复MAC admission为0；
- VO/VI均观察到A-MPDU，最大均达到64 MPDU/PSDU。

冒烟结果仅用于正确性门禁，不用于最终性能结论。正式三seed结果见下一节。

## 7. 20秒Default EDCA结果

三组正式运行全部通过自动门禁。单个seed墙钟时间为1486–1575 s；正式运行使用summary模式，没有逐包文本I/O，但保留全部Wi-Fi协议事件。

### 7.1 整体占用与竞争强度

| 指标 | 三seed结果 |
|---|---:|
| 总成功MAC有效载荷吞吐量 | **40.702 ± 3.061 Mbps** |
| 信道忙碌比例 | **89.19%**（三seed均值） |
| 同时有上行待发数据的STA | 平均**39.987/40**；全采样最小39 |
| 72个源队列非空比例 | 各源最小**98.51%**；分AC均值99.83%–100% |
| 活跃源平均队列深度 | 63.46–64.75 packet |

全采样只有测量边界附近的瞬时样本降至39个待发STA；平均值接近40，所有72个`STA×AC`源的队列非空比例均超过98.5%。因此本环境确实形成了40个STA上行高优先级饱和竞争，而不是TCP发送窗或应用到达间隔造成的“偶尔有包”。

### 7.2 分AC核心指标

下表为三seed的“逐seed指标均值 ± 样本标准差”。失败率不包含仿真截止未完成，故未完成比例单列。

| AC | 吞吐量/Mbps | 成功时延p95/ms | 成功时延p99/ms | 20/50 ms及时率 | 最终失败率 | 其中过期率 | 截止未完成 | STA吞吐Jain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VO | 7.444 ± 1.023 | **178.96 ± 18.28** | 312.62 ± 21.92 | **49.92% ± 7.28%** | 0.187% ± 0.092% | 0.005% ± 0.002% | 0 | 0.636 ± 0.177 |
| VI | 32.639 ± 4.017 | **221.20 ± 18.98** | 389.13 ± 4.83 | **73.68% ± 5.49%** | 1.442% ± 0.352% | 0.254% ± 0.221% | 0 | 0.435 ± 0.097 |
| BE | 0.582 ± 0.216 | 566.58 ± 139.67 | 1009.82 ± 364.39 | 不适用 | **79.07% ± 3.84%** | **66.38% ± 5.12%** | 6.20% ± 2.61% | 0.128 ± 0.066 |
| BK | 0.0367 ± 0.0435 | 样本不足 | 样本不足 | 不适用 | **84.19% ± 3.41%** | **82.35% ± 3.09%** | 13.78% ± 4.83% | 0.122 ± 0.055（仅2个有成功样本seed） |

BK的低p95不能解释成性能好：seed 41成功MPDU为0，seed 42为157，seed 43仅47；其吞吐、失败/过期和样本数共同表明BK几乎被饿死。BE也只有0.58 Mbps，约79%的已入队MPDU最终失败，公平性很差。

### 7.3 逐seed高优先级结果

| seed | 总吞吐/Mbps | 忙碌比例 | VO p95/ms | VI p95/ms | VO及时率 | VI及时率 |
|---:|---:|---:|---:|---:|---:|---:|
| 41 | 44.136 | 89.75% | 199.75 | 200.77 | 48.03% | 79.72% |
| 42 | 39.709 | 89.03% | 171.74 | 224.57 | 43.77% | 72.36% |
| 43 | 38.261 | 88.78% | 165.41 | 238.28 | 57.96% | 68.97% |

三seed均呈现百毫秒级VO/VI长尾，而不是此前应用受TCP节流后只对少量进入MAC的数据得到的低服务时延。成功样本数量仍很充足：每seed约51–65万VO MPDU和42–53万VI MPDU，因此p95不是少量样本假象。

### 7.4 原生BEB与聚合证据

运行时Trace观察到的CW集合为：VO `{3,7}`、VI `{7,15}`、BE/BK `{15,31,63,127,255,511,1023}`，与Default EDCA下原生BEB及各AC CWmax截断一致。

VO/VI每个seed的最大A-MPDU均达到64 MPDU/PSDU；三seed平均每个PSDU约59.76个VO MPDU、58.19个VI MPDU。BE最大57–58个MPDU，BK因几乎没有发送机会而不能稳定评估。由此可见32/64水位没有人为压低高优先级A-MPDU能力；它维持饱和队列，同时避免把500包队列上限长期灌满。

### 7.5 图表

![单BSS拓扑与业务覆盖](../results/figures/mac_fullbuffer40_default_20s_3seed_20261028/topology_and_source_coverage.png)

![各AC成功吞吐量](../results/figures/mac_fullbuffer40_default_20s_3seed_20261028/ac_throughput.png)

![VO和VI最终MAC服务时延CCDF](../results/figures/mac_fullbuffer40_default_20s_3seed_20261028/vo_vi_service_delay_ccdf.png)

![失败率、过期率与Jain公平性](../results/figures/mac_fullbuffer40_default_20s_3seed_20261028/failure_expiry_and_jain.png)

![队列非空与竞争强度](../results/figures/mac_fullbuffer40_default_20s_3seed_20261028/queue_and_contention_diagnostics.png)

## 8. 结论

1. **是否真正形成40 STA高优先级饱和竞争：是。** 20个VO STA和另外20个VI STA覆盖全部40个STA；平均39.987个STA同时待发，所有72个源队列至少98.5%的时间非空。
2. **Default吞吐：** 三seed总有效载荷吞吐为40.70 ± 3.06 Mbps，其中VI约32.64 Mbps、VO约7.44 Mbps；BE与BK合计不足0.62 Mbps。
3. **VO/VI尾时延：明显。** VO/VI成功样本p95约179/221 ms，p99约313/389 ms；只有约49.9%的VO和73.7%的VI满足20/50 ms参考期限。
4. **BE/BK是否饥饿：严重。** BE最终失败约79%，BK约84%；BK平均吞吐仅0.037 Mbps且一个seed完全没有成功样本。
5. **水位机制影响：** 32/64水位刻意让每个源保持饱和，因此会产生真实的MAC排队时延；它不是应用到达时延。高水位等于BA窗口且VO/VI实际达到64 MPDU聚合，故没有人为限制聚合；同时避免500包队列长期满载造成更强的人为缓存膨胀。
6. **是否适合下一阶段固定EDCA/CW搜索：适合，但必须带约束。** 该环境对AIFS、CW/BEB、TXOP、聚合和重传很敏感，且Default下已有清晰的时延/公平性/饥饿优化空间。下一阶段搜索必须联合约束VO/VI及时率、BE/BK最低服务、失败/过期和Jain公平性，不能只最大化总吞吐，也不能用成功样本时延掩盖失败包。

自动门禁、三seed原始数据、PNG和汇总CSV均已保存。本结论只针对单BSS MAC饱和机制环境，不宣称替代7 BSS真实TCP/UDP应用场景。

## 9. 复现入口

- 配置：[`configs/mac_fullbuffer40_default.yaml`](../configs/mac_fullbuffer40_default.yaml)
- 输入构建：[`scripts/build_mac_fullbuffer40_input.py`](../scripts/build_mac_fullbuffer40_input.py)
- 运行：[`scripts/run_mac_fullbuffer40_baseline.py`](../scripts/run_mac_fullbuffer40_baseline.py)
- 分析绘图：[`scripts/analyze_mac_fullbuffer40_default.py`](../scripts/analyze_mac_fullbuffer40_default.py)
- ns-3场景：[`ns3_ext/tgax-s3.cc`](../ns3_ext/tgax-s3.cc)
- 三seed汇总：[`results/summary/mac_fullbuffer40_default_20s_3seed_20261028/ac_mean_std.csv`](../results/summary/mac_fullbuffer40_default_20s_3seed_20261028/ac_mean_std.csv)
- 自动门禁：[`results/summary/mac_fullbuffer40_default_20s_3seed_20261028/validation.json`](../results/summary/mac_fullbuffer40_default_20s_3seed_20261028/validation.json)

```powershell
.\.venv\Scripts\python.exe scripts\run_mac_fullbuffer40_baseline.py --phase smoke
.\.venv\Scripts\python.exe scripts\run_mac_fullbuffer40_baseline.py --phase formal --workers 3
.\.venv\Scripts\python.exe scripts\analyze_mac_fullbuffer40_default.py
```
