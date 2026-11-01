# 单BSS、40 STA、MAC直注随机到达Default EDCA复测

> 实验日期：2026-10-29  
> 本轮保留TGax业务到达状态和包长模型，但明确取消TCP、UDP、IP与Socket；未训练DDQN、未修改Reward或EDCA/CW。

## 1. 为什么增加这一组实验

上一轮Full Buffer实验刻意让72个`STA×AC`队列持续非空，用于观察纯EDCA饱和竞争。它不能回答“保留VoIP通话/静默、Gaming随机间隔、视频帧、FTP文件/阅读间隔和HTTP页面过程以后，MAC性能如何”。本轮因此建立独立的`mac-arrival`场景：网络、位置、PHY/MAC和Default EDCA不变，只把业务源改成随机到达模型。

它仍不是端到端应用实验：没有TCP拥塞控制、Socket缓存、IP头或TCP ACK。它回答的是“给定TGax形状的外生业务需求，数据直接进入Wi-Fi MAC时的排队与接入行为”。

## 2. 固定网络与业务分配

网络仍为原BSS0的1 AP、40 STA真实坐标，802.11ax HE SU、5 GHz、80 MHz、HE MCS7，不启用OFDMA/MU-MIMO；所有节点使用ns-3原生Default EDCA和BEB。

业务分配复用已冻结的`candidate_full40_multiac_v6`中BSS0，避免重新人为挑选STA：

| 模型 | 方向/AC | 数量 | 主要作用 |
|---|---|---:|---|
| VoIP | UL/VO | 40 | 40个STA均有上行实时业务 |
| VoIP | DL/VO | 1 | 与其中1个STA组成双向VoIP |
| Gaming | DL/VI | 1 | 随机包长和随机到达间隔 |
| BV3视频 | UL/VI | 1 | Weibull帧长、Gamma网络时延扰动 |
| HTTP | DL/BE | 20 | 页面、对象和阅读过程 |
| FTP | UL/BE | 12 | 文件大小和阅读间隔 |
| HTTP | DL/BK | 16 | 同一HTTP模型映射BK |
| FTP | UL/BK | 8 | 同一FTP模型映射BK |

36个HTTP实例各自增加350 B反向请求模型，因此99条配置流在运行时形成135个直接MAC业务生成器。

## 3. MAC直注和需求缓冲

所有配置流的运行时transport均为`NONE`。原TCP/UDP类型仅保存在输入CSV的`reference_transport`列用于追溯，仿真不安装`InternetStackHelper`。

每个生成器先按原随机模型产生业务对象，再按合法MSDU分段：

- VoIP的7/33 B和Gaming小包保持单包；
- BV3视频帧按不超过1460 B分段；
- FTP文件和HTTP页面绝不作为单个MAC帧，同样分成不超过1460 B的MSDU；
- FTP保留76%使用1460 B、24%使用536 B分段上限的选择；HTTP请求固定350 B。

未提交需求保存在应用侧队列；每个源以随机化的500 us周期检查对应`QosTxop`队列，共享队列低于64包时每次最多提交一个MSDU。这样既保留外生需求债务，又避免把ns-3的500包MAC队列瞬间灌满。数据仍经`WifiMac::NotifyTx`与`WifiMac::Enqueue`进入正常QoS队列，随后由EDCA、退避、BEB、TXOP、A-MPDU、Block ACK和重传处理。

## 4. 短时正确性门禁

seed 41、5秒全Trace冒烟通过：

- 40/40 STA关联且BSSID匹配；
- 五类主业务及HTTP请求全部实际生成；
- 41个VoIP流观察到195次状态切换，7 B静默包和33 B活跃包均存在；
- Gaming、BV3、FTP、HTTP的随机包/帧/文件/页面均存在；
- 最大业务对象达到MB级，但实际MAC有效载荷最大1460 B；
- `生成字节=已提交MAC字节+应用侧待提交字节`；
- 四AC上下行均有真实MAC活动，MAC账本守恒；
- `ipv4_stack_installed=0`，IP Flow结果为空，重复MAC admission为0；
- BE/BK均观察到最大64 MPDU/PSDU的A-MPDU。

冒烟生成器在1秒业务预热和5秒测量内共生成47.14 MB，全部在停止前提交MAC；该数值只用于验证机制，不作为正式性能结果。

## 5. 三seed 20秒Default结果

正式seed 41、42、43均通过以下门禁：40/40关联、无IPv4栈和IP流、全部运行时transport为`NONE`、六种运行时模型均生成需求、应用需求字节守恒、零重复admission、上下行四AC均有活动、MAC账本守恒。

### 5.1 业务需求、MAC提交与竞争强度

生成器统计覆盖1秒业务预热和20秒测量，MAC性能统计只覆盖20秒测量：

| 指标 | 三seed均值 |
|---|---:|
| 外生生成需求 | 64.641 MB |
| 已提交MAC | 59.665 MB |
| 应用侧待提交 | 4.977 MB |
| 20秒成功MAC有效载荷吞吐 | **17.798 ± 1.999 Mbps** |
| 信道忙碌比例 | **31.81%** |
| 瞬时MAC队列非空源节点（旧统计，含AP） | 平均**4.44**，最大26 |

应用侧待提交基本全部来自FTP：FTP平均生成44.177 MB，提交39.218 MB，仍有4.959 MB债务。BV3生成16.006 MB，提交15.989 MB；VoIP、Gaming和HTTP基本能及时提交MAC。由于生成统计比MAC性能多含1秒预热，而且测量末尾还有MAC队列与排空，两者不能直接相除当作交付率。

![需求生成与MAC提交](../results/figures/mac_arrival40_default_20s_3seed_20261029/model_demand_and_mac_admission.png)

### 5.2 各AC与方向的MAC性能

下表为逐seed指标的均值±样本标准差。时延只统计成功确认MPDU，同时列出失败、过期和截止未完成。

| 方向/AC | 吞吐/Mbps | p95/ms | p99/ms | 最终失败率 | 过期率 | 未完成率 | 20/50 ms及时率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| UL-VO | 0.262 ± 0.008 | **0.830 ± 0.163** | 2.391 ± 0.685 | 0.053% | 0 | 0 | **99.943%** |
| DL-VO | 0.008 ± 0.001 | **0.717 ± 0.072** | 2.107 ± 0.806 | 0 | 0 | 0 | **100%** |
| UL-VI | 6.109 ± 0.244 | **1.160 ± 0.408** | 3.684 ± 1.401 | 0 | 0 | 0 | **100%** |
| DL-VI | 0.187 ± 0.001 | **1.076 ± 0.250** | 2.824 ± 1.492 | 0 | 0 | 0 | **100%** |
| UL-BE | 5.911 ± 0.988 | 67.76 ± 17.12 | 220.93 ± 47.85 | 0.719% | 0.374% | 0 | 不适用 |
| DL-BE | 0.366 ± 0.151 | 44.88 ± 68.69 | 52.10 ± 77.79 | 0 | 0 | 0 | 不适用 |
| UL-BK | 4.513 ± 1.046 | 149.51 ± 95.26 | 264.90 ± 121.56 | 0.929% | 0.771% | 0.873% | 不适用 |
| DL-BK | 0.444 ± 0.111 | **321.92 ± 173.49** | 471.08 ± 25.36 | **8.175%** | **7.491%** | 0 | 不适用 |

UL-VO的40个STA吞吐Jain为0.950±0.008。UL-BE和UL-BK只有0.180±0.019和0.320±0.145，但这里每个FTP实例的随机文件大小、启动和阅读状态不同，原始吞吐Jain同时包含需求差异，不能单独解释成EDCA不公平；需要按每STA生成需求归一化后再评价满足公平性。UL-VI只有一个BV3源，其Jain=1没有多用户公平性含义。

![各AC上下行吞吐](../results/figures/mac_arrival40_default_20s_3seed_20261029/direction_ac_throughput.png)

![VO/VI MAC服务时延CCDF](../results/figures/mac_arrival40_default_20s_3seed_20261029/vo_vi_service_delay_ccdf.png)

![失败过期与及时率](../results/figures/mac_arrival40_default_20s_3seed_20261029/failure_expiry_and_timely.png)

![随机到达竞争强度](../results/figures/mac_arrival40_default_20s_3seed_20261029/arrival_competition_diagnostics.png)

### 5.3 为什么VO/VI时延再次很低

这里的4.44不是“有业务STA数”，而是旧版每100 ms采样时MAC队列非空的源节点数，并且错误地把AP下行队列也计入了STA名称。它不包含应用侧待提交、已从队列取出正在退避/重传的MPDU。40个STA始终都有VoIP业务。后续高Gaming复测已把统计修正为严格只计STA，并另列含AP的`active_nodes`。

这次低时延是可解释且账本一致的，不是TCP节流：

- 业务需求时钟在MAC直注生成器中独立运行，没有TCP拥塞窗或Socket阻塞；
- 40路VoIP总吞吐需求本来很小，通话/静默模型使单STA队列多数时间为空；
- VI只有1路6 Mbps BV3上行和1路低码率Gaming下行，不等于Full Buffer中的20个持续VI上行STA；
- 旧队列快照平均只有4.44个非空源节点（含AP），信道忙31.8%，远低于Full Buffer的39.99个上行STA和89.2%。

因此本轮证明的是：**保留当前业务数量和官方随机到达过程后，单BSS中的VO/VI并未进入饱和竞争区。** 低p95并不表示高负载下Default仍能保持该时延。

### 5.4 与Full Buffer基线的关系

| 指标 | 随机到达 | Full Buffer |
|---|---:|---:|
| 总吞吐 | 17.80 Mbps | 40.70 Mbps |
| 信道忙 | 31.81% | 89.19% |
| 100 ms队列快照 | 4.44个非空源节点（含AP） | 39.99个非空上行STA |
| VO-UL p95 | 0.83 ms | 178.96 ms |
| VI-UL p95 | 1.16 ms | 221.20 ms |
| VO 20 ms及时率 | 99.94% | 49.92% |
| VI 50 ms及时率 | 100% | 73.68% |

这不是同一负载下两种实现的性能差，而是两个不同的研究工作点：随机到达保留业务真实性，Full Buffer提供MAC饱和压力上界。二者都应保留，不能互相替代。

## 6. 结论

1. 不经过TCP/UDP/IP也可以完整运行VoIP状态、Gaming随机间隔、BV3视频帧、FTP文件/阅读和HTTP页面过程，本轮已用真实Trace验证。
2. 大业务对象在应用侧保留为需求并分段，MAC有效载荷不超过1460 B；64包提交水位没有把500包队列瞬间灌满。
3. 当前40 STA业务分配下，高优先级到达强度并不足以形成40 STA持续竞争，因此VO/VI时延低是合理结果。
4. 低优先级仍可出现明显积压和长尾，尤其DL-BK p95约322 ms、失败约8.2%，FTP还有约4.98 MB未提交债务。
5. 若下一步要研究“真实到达但更高VO/VI压力”，应只增加独立VoIP/Gaming/视频实例数量或建立倍率变体，并继续保留本轮生成—提交—MAC结果三层账本；不能重新改成Full Buffer后仍称为原到达模型。

## 7. 复现入口

- 配置：[`configs/mac_arrival40_default.yaml`](../configs/mac_arrival40_default.yaml)
- 输入构建：[`scripts/build_mac_arrival40_input.py`](../scripts/build_mac_arrival40_input.py)
- 运行：[`scripts/run_mac_arrival40_baseline.py`](../scripts/run_mac_arrival40_baseline.py)
- 分析：[`scripts/analyze_mac_arrival40_default.py`](../scripts/analyze_mac_arrival40_default.py)
- ns-3后端：[`ns3_ext/tgax-s3.cc`](../ns3_ext/tgax-s3.cc)
- 三seed汇总：[`results/summary/mac_arrival40_default_20s_3seed_20261029/direction_ac_mean_std.csv`](../results/summary/mac_arrival40_default_20s_3seed_20261029/direction_ac_mean_std.csv)
- 自动门禁：[`results/summary/mac_arrival40_default_20s_3seed_20261029/validation.json`](../results/summary/mac_arrival40_default_20s_3seed_20261029/validation.json)
