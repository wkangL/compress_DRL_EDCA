# TGax Scenario 3 官方参数核对表

更新时间：2026-10-22  
状态：原文核对与ns-3.44映射完成；7 BSS、280 STA正式研究子集正在进行Default EDCA标定

## 1. 文档与适用范围

| 文档 | 本项目用途 | 本地原文 |
|---|---|---|
| IEEE 802.11-14/0980r16, *TGax Simulation Scenarios* | Scenario 3 布局、节点数、PHY/MAC候选参数及每BSS业务数量 | `reports/references/11-14-0980-16-00ax-simulation-scenarios.docx` |
| IEEE 802.11-14/0571r12, *TGax Evaluation Methodology* | Appendix 2 应用业务生成器；集成系统仿真所需PHY/MAC功能 | `reports/references/11-14-0571-12-00ax-evaluation-methodology.docx` |
| IEEE 802.11-14/0882r4, *TGax Channel Model Document* | 室内TGn/TGac模型、Scenario 3路径损耗和阴影衰落 | `reports/references/11-14-0882-04-00ax-tgax-channel-model-document.docx` |

这些是TGax工作组评估文档，不是最终IEEE 802.11ax标准正文。文档保留了候选值、方括号值和TBD。本表不会把TBD补成“官方值”。

正式研究配置不是完整19 BSS场景，而是**参考下列TGax官方参数构建的7 BSS研究子集**：中心同信道BSS加reuse-3几何中的六个最近同信道邻居，每BSS完整保留40 STA和官方业务数量。`full19`保留配置但不安排正式实验，`main3`仅作协议调试。

原文SHA-256用于复核版本：0571r12=`074C31B08B773A7E6FF386AFBB15DC43327C4E84673843858B374B3D04C23C2F`；0980r16=`0D29D13B572751BDCAB86151BF6FAA721335557BFB56FBC336A4EF7241F7F461`；0882r4=`EBFCAD7845CC2CF2C3ED475F45D2378F439A42EB2CB0A0C10B83389907B39CA9`。相邻的`.audit.txt`是由`scripts/extract_docx_text.py`从原文XML提取的审计文本，不替代DOCX原件。

## 2. 场景规模与布局

| 参数 | 官方原文位置 | 官方值 | 本次选择 | 状态与说明 |
|---|---|---:|---:|---|
| 场景 | 0980r16，“3 - Indoor Small BSSs Scenario” | Indoor Small BSSs | Scenario 3 | 一致 |
| 小区形状 | 0980r16，Scenario 3 Figure 6/7及参数表“Environment description” | 正六边形规则对称网格 | 正六边形 | 一致 |
| 六边形半径 | 同上 | `R=10 m` | 10 m | 一致 |
| 基础相邻BSS距离 | 同上 | `ICD=2h`, `h=sqrt(R²-R²/4)` | 17.320508 m | 由官方公式计算 |
| 默认复用 | 0980r16，Scenario 3正文及“Primary channels” | reuse 3为默认；reuse 1也应考察 | reuse 3 | 采用官方首选 |
| 同信道最近站点距离 | Figure 7复用3几何 | 图示/几何导出 | 30 m | 已由半径10 m、复用因子3的六边形几何及生成坐标测试核验 |
| 完整性能布局 | 0980r16，Scenario 3 STA placement（reuse 3） | 61-cell投放区域，保留图7中的19个同信道AP，每个装满N1个STA | 保留为`full19`可选配置 | 官方完整共信道性能配置，本阶段不作为主运行 |
| 当前正式研究子集 | 官方未定义 | 无 | 中心+第一圈，共7个同信道BSS、40 STA/BSS | 明确标为“TGax-based 7 BSS研究子集”，不称完整TGax场景 |
| 协议调试场景 | 官方未定义 | 无 | `main3`，3 BSS、10 STA/BSS | 只用于快速协议调试，不作正式性能结论 |
| AP位置 | 0980r16，Scenario 3 “APs location” | 六边形中心，高3 m | 中心，高3 m | 一致 |
| STA位置 | 0980r16，Scenario 3参数表“STAs location” | 61-cell区域均匀随机投放，按含阴影的接收功率关联 | 按同一规则固定种子生成 | seed 20261021的`subset7`已验证7×40 STA；280/280运行时BSSID与最强AP匹配 |
| 每AP STA数 | 0980r16，Scenario 3 “Number of STA and STAs type” | `N=[30] or 40`, `N1=[N]` | 正式配置40 | 采用官方候选值40；不缩减业务数量 |
| 设备类型 | 同上 | HEW；legacy比例TBD | 全部HE | `N1=N`候选；legacy比例官方未定 |
| 关联比例 | 0980r16，Scenario 3 “Association” | `[X=100,Y=0,Z=0]`，按RSSI含路径损耗/阴影/穿透损耗 | 100%最强AP | 一致 |
| wrap-around | 0980r16，Scenario 3 “Wrap-around” | Used / Not used | 初始实现关闭 | 官方未选定；完整运行需报告边缘效应 |
| 建筑墙体/楼层 | 0980r16 Scenario 3；0882r4 “Indoor floor and wall penetration loss” | Scenario 3未定义具体墙/楼层布局；0882r4只为Scenario 1/2给墙损参数 | 无墙/楼层穿透项 | 标记“官方未明确规定”，不虚构建筑平面 |

## 3. 信道与传播

| 参数 | 官方原文位置 | 官方值 | 本次选择 | ns-3.44表达与偏差 |
|---|---|---|---|---|
| 室内空间信道 | 0882r4 “Indoor spatial channel models”, Table I；0980r16 Scenario 3 “Channel Model” | TGn/TGac Model D，NLOS；50 ns RMS delay spread，3 clusters，16/7/4 taps | Model D | ns-3.44没有可直接调用的TGn/TGac Model D频率选择性MIMO信道；本轮先精确实现其大尺度路径损耗和链路静态阴影，频率选择性多径列为明确偏差 |
| 路径损耗 | 0980r16 Scenario 3 “Pathloss model”；0882r4 “Indoor path loss…”, Table III | `40.05 + 20log10(fc/2.4) + 20log10(min(d,10)) + I(d>10)35log10(d/10)`；`d=max(3D距离,1m)` | 原式 | 新增只负责传播的ns-3模型，不改MAC/EDCA |
| 阴影衰落 | 0980r16 Scenario 3；0882r4 Table III | 对所有链路独立、链路静态，NLOS标准差5 dB | 5 dB | 固定seed、TX-RX对称、每链路一个高斯样本 |
| 穿透损耗 | 0882r4 “Indoor floor and wall penetration loss” | Scenario 1单墙5 dB，Scenario 2单墙7 dB；未为Scenario 3给墙体布局/单墙值 | 不添加 | 官方未明确规定 |
| 多普勒 | 0882r4 “Doppler” | 该段明确讨论室外UMi/UMa；Scenario 3无移动速度值 | 静止节点 | 官方未明确规定 |

## 4. PHY选择

| 参数 | 官方原文位置 | 官方候选值 | 本次选择 | 选择理由或偏差 |
|---|---|---|---|---|
| 频段/带宽 | 0980r16公共PHY表及Scenario 3 “Primary channels” | 2.4GHz/20MHz或5GHz/80MHz | 5GHz/80MHz | Scenario 3明确允许，且保留HE 80MHz能力 |
| 实际80MHz信道 | 官方只规定复用和带宽，未给具体IEEE信道号 | 未明确 | 中心信道42，主20MHz信道36 | 合法的5GHz 80MHz块；属于实现选择 |
| MCS | 0980r16 Scenario 3 “MCS” | MCS0或MCS7 | 固定HE MCS7；另保留MCS0验证模式 | MCS7用于承载官方高码率混合业务；不使用自适应率来引入额外变量 |
| GI | 0980r16 Scenario 3 “GI” | Short | HE 0.8 us | ns-3 HE最短GI为0.8 us；早期TGax文档未用最终HE GI枚举 |
| AP天线 | 0980r16公共PHY表及Scenario 3 | 2或4 TX，2或4 RX | 2 TX/2 RX | 官方候选中计算量较低者 |
| STA天线 | 同上 | 1或2 TX/RX | 1 TX/1 RX | 官方候选中计算量较低者 |
| 空间流 | 官方场景未单独给NSS | 未明确 | 单用户1空间流 | 本轮不做MU-MIMO；天线数仍按上述设置 |
| AP发射功率 | 0980r16公共PHY表 | 20 dBm/antenna | ns-3总功率23.0103 dBm（2根等功率天线） | ns-3 PHY功率属性为设备总功率，进行线性功率合并换算并公开记录 |
| STA发射功率 | 同上 | 15 dBm/antenna | 15 dBm | STA为单天线 |
| AP/STA天线增益 | 同上 | AP 0 dBi，STA -2 dBi | 0/-2 dBi | 一致 |
| 接收机噪声系数 | 0980r16公共PHY表 | 7 dB | 7 dB | 0571r12的10 dB位于PHY PER impairment表；Scenario公共系统参数采用7 dB |
| Rx sensitivity | Scenario 3未给具体值；0571r12 Appendix 4只描述处理流程 | 官方未明确规定 | ns-3默认 -101 dBm/20MHz | 明确为ns-3实现值 |
| CCA-SD / CCA-ED | Scenario 3未给具体值；0571r12 Box 2/Box 4出现校准用候选 | 官方未明确规定 | ns-3 CCA-SD -82 dBm，CCA-ED -62 dBm | 不把校准Box中的-70 dBm冒充Scenario 3值 |

## 5. MAC选择

| 参数 | 官方原文位置 | 官方值 | 本次选择 | 状态与说明 |
|---|---|---|---|---|
| 接入协议 | 0980r16公共MAC表；Scenario 3 “Access protocol parameters” | Default EDCA | ns-3标准默认EDCA | 本轮不启用研究CW扩展和DDQN |
| SSID/BSSID选择 | 0980r16 Scenario 3参数表“Association” | 按最强接收功率关联 | 所有AP共同SSID，STA由ns-3选择BSSID | 运行后逐STA与预计算最强AP核对；不使用不同SSID强制归属 |
| A-MPDU | 0980r16公共MAC表；Scenario 3 “Aggregation” | A-MPDU，max size/BA window；无A-MSDU；immediate BA | 四AC 65535 Byte；A-MSDU=0 | 官方未给max size数值，65535为显式ns-3选择，不能称官方值 |
| BA窗口 | 同上 | 未给数值 | 64 MPDU | ns-3显式选择；将通过真实Block ACK帧验证 |
| 重试 | 0980r16公共MAC表；Scenario 3 | 10 | `FrameRetryLimit=10` | 一致；ns-3属性表示最大传输尝试次数，报告中保留语义差异 |
| RTS/CTS | 同上 | no RTS/CTS | 关闭 | 一致 |
| Beacon | 0980r16 Scenario 3 traffic table D7 | 280 octets，5GHz 6Mbps，每100 TU | 100 TU；各AP使用独立初始jitter；实际ns-3 Beacon长度实测 | 文档未要求多AP同步起始；ns-3管理帧字段长度由实现生成，不能强行伪造280 Byte |
| 队列上限/寿命 | 官方Scenario 3未明确规定 | 未明确 | ns-3默认500包/500 ms | 与旧项目10 s队列完全分离 |
| OFDMA/MU-MIMO | 本轮研究约束 | 暂不考虑 | UL/DL OFDMA关闭，MU-MIMO关闭 | 保留SU HE、QoS、聚合、BA和重传 |

## 6. Scenario 3 每BSS官方业务

来源：0980r16 Scenario 3 “Traffic model for each BSS”表。一个STA可以被不同业务类型重复抽中；每一行内部随机分配给不同STA。

| ID | 方向 | 模型 | 官方数量/40 STA | `subset7`数量/40 STA | AC | 本轮传输协议 |
|---|---|---|---:|---:|---|---|
| D1 | AP→STA | BV6 15.6 Mbps buffered video | 12 | 12 | VI | 真实TCP |
| D2 | AP→STA | BV3 6 Mbps buffered video | 8 | 8 | VI | 真实TCP |
| D3 | AP→STA | FTP | 4 | 4 | BE | 真实TCP |
| D4 | AP→STA | HTTP | 12 | 12 | BE | 真实TCP |
| D5 | AP→STA | Gaming GMG | 16 | 16 | VI | UDP |
| D6 | AP↔STA | VoIP | 12对 | 12对 | VO | UDP双向 |
| D7 | AP广播 | Beacon 280 octets/100 TU/6 Mbps | 1 | 1 | 文档列VI | ns-3原生管理帧 |
| U1 | STA→AP | BV3 6 Mbps buffered video | 4 | 4 | VI | 真实TCP |
| U2 | STA→AP | FTP | 4 | 4 | BE | 真实TCP |
| U3 | STA→AP | Gaming GMG | 16 | 16 | VI | UDP |
| U4 | 未关联STA广播 | Probe Request | 全部未关联STA | 0 | 文档列VI | 基础场景X=100且没有未关联STA；官方发送参数TBD |

## 7. Appendix 2生成器参数

| 模型 | 0571r12原文位置 | 关键参数 | 尚需说明 |
|---|---|---|---|
| VoIP | Appendix 2 “Voice-over-IP”, Table 1 | active 33 B/20 ms，silence 7 B/160 ms；IPv4压缩头3 B；状态时长指数均值1.25 s；a=b=0.016；DL Laplacian jitter β=5.11 ms，UL jitter=0 | 使用UDP；状态、包长、间隔和方向均需事件统计验证 |
| Buffered Video | Appendix 2 “Buffered Video Streaming”, Table 5 | BV3 λ=20850，BV6 λ=54210，k=0.8099；TCP segment 1500 B；Gamma latency k=0.2463, θ=60.227，均值14.834 ms | 文档说固定帧间隔但未在该节给数值；实现将用期望帧长与名义码率导出帧周期并标为推导值 |
| FTP | Appendix 2 “Local File Transfer”, Table 1 | 文件截断lognormal：μ=14.45、σ=0.35，均值2 MB、SD 0.722 MB、max 5 MB；阅读时间指数均值180 s；76% MTU1500、24% MTU576；新TCP连接初始窗口1 | 文档没有给文件最小值，按正值且不超过5 MB接受；真实TCP ACK由ns-3产生 |
| HTTP | Appendix 2 “Web Browsing”, Table 2 | 主对象lognormal μ=8.37/σ=1.37/min100/max2MB；嵌入对象μ=6.17/σ=2.36/min50/max2MB；对象数截断Pareto α=1.1/k=2/m=55，随后减k，最大53；阅读30 s；解析0.13 s；请求350 B；HTTP1.0/1.1各50% | 公式已从DOCX Office Math原文提取；实现需验证截断后样本统计 |
| Gaming | Appendix 2 “Gaming”, Table 6 | UDP；DL IAT largest-extreme-value a=15,b=7 ms、size a=390,b=89 B；UL IAT a=23.5,b=10.5 ms、size a=158,b=26.2 B；初始0–20 ms | 表前正文写0–40 ms但表写0–20 ms，属于官方文档内部不一致；本次优先采用表值并记录差异 |

## 8. 当前已知实现边界

1. ns-3.44可以直接实现HE SU、真实IPv4/TCP/UDP、QoS映射、A-MPDU、Block ACK、标准重传和多AP竞争。
2. ns-3.44没有直接可用的TGn/TGac Model D簇/抽头信道。本轮不会用普通Nakagami模型冒充；先实现官方大尺度公式与5 dB链路阴影，并在差异清单中保留多径缺口。
3. 0980r16形成于最终802.11ax标准之前，使用“HEW”、11ac preamble等早期表述。本轮使用最终ns-3 HE SU PHY，同时记录这种年代差异。
4. 当前正式场景为7 BSS、40 STA/BSS；这是以官方reuse-3几何为基础的研究子集，不能称为完整19 BSS TGax Scenario 3。完整19 BSS配置保留，main3只作协议调试。
5. 原S1–S5、27.30225 Mbps容量、旧队列参数和DDQN训练结果均不进入本场景。
