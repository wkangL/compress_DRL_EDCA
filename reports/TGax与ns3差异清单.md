# TGax Scenario 3 与 ns-3.44 实现差异清单

更新时间：2026-10-22  
用途：防止把“官方参数驱动的环境”误写成“逐项严格复现”。详细来源见`TGax官方参数核对表.md`。

| ID | 项目 | 官方文档状态 | 当前实现 | 类型 | 对结论的限制 |
|---|---|---|---|---|---|
| D00 | 主场景规模与业务数 | Scenario 3性能布局为19活动同信道BSS、40 STA/BSS | 正式配置为中心加第一圈7 BSS、40 STA/BSS；业务数量完整保留 | 研究范围子集 | 只能称TGax-based 7 BSS研究子集，不能称完整Scenario 3 |
| D01 | Model D频率选择性多径 | 0882r4给出50 ns、3簇、16/7/4 taps | ns-3.44无直接模型；实现官方路径损耗和5 dB链路静态阴影 | ns-3能力缺口 | PER、MIMO相关性和频率选择性不能称严格Model D |
| D02 | wrap-around | 0980r16列Used/Not used，未定 | 关闭 | 官方未决项选择 | 7 BSS边缘节点缺少外围BSS，边缘干扰可能偏低 |
| D03 | 具体5 GHz信道号 | 只给80 MHz和reuse方案 | ch42，主20 MHz ch36 | 实现选择 | 不影响复用几何，但不是官方唯一值 |
| D04 | CCA/Rx sensitivity | Scenario 3未明确 | ns-3默认-82/-62/-101 dBm | 实现选择 | CCA图和隐藏节点关系依赖这些值 |
| D05 | A-MPDU/BA窗口数值 | 只要求A-MPDU、max size/BA window、immediate BA | 65535 Byte、64 MPDU | 实现选择 | 聚合效率依赖该值 |
| D06 | MAC队列 | Scenario 3未规定容量或寿命 | 500包、500 ms | ns-3默认 | 严重过载下交付和尾部受队列约束 |
| D07 | VoIP压缩头 | IPv4压缩头3 Byte | 普通IPv4/UDP头，无ROHC | 协议栈能力缺口 | 实际空口包略大 |
| D08 | Beacon长度 | 280 octets/100 TU/6 Mbps | 周期和控制速率；帧字段由ns-3生成 | 表达偏差 | 不保证恰好280 octets |
| D09 | HTTP版本语义 | HTTP1.0/1.1各50%，350 Byte请求、解析/阅读时间 | 真实350 Byte反向TCP、对象分布、解析/阅读时间；完整重连/复用语义未完全实现 | 应用模型待完善 | 长时HTTP会话数和握手开销存在偏差 |
| D10 | FTP下一文件连接时刻 | 每文件新TCP连接，传完后读180 s | 首文件是真实TCP、MSS比例及InitialCwnd=1；短测中无第二文件，长时重连完成时刻待完善 | 应用模型待完善 | 不影响短测首文件；长时FTP会话级行为有限制 |
| D11 | U4 Probe Request | 官方发送参数TBD | 不额外生成；X=100时基础输入无未关联STA业务 | 官方TBD | 不能评估探测请求额外占用 |
| D12 | HEW早期候选与最终HE | 2014年文档含早期候选表述 | ns-3.44最终802.11ax HE SU | 年代差异 | 管理/能力字段不能逐字段等同早期草案 |
| D13 | MU功能 | TGax方法论覆盖MU；本轮明确暂不考虑 | 未安装MU scheduler，禁用UL/DL OFDMA和MU-MIMO | 本轮研究边界 | 结果只适用于SU HE竞争环境 |
| D14 | 正式批量PHY后端 | 官方方法没有规定ns-3后端 | 正式扫描用Yans标量同信道干扰；Spectrum用于短时交叉核验 | 计算实现选择 | 保留真实HE PHY/MAC，但不表达频谱选择性干扰细节 |

## 不属于差异的项目

- TCP ACK由真实ns-3 TCP栈产生；没有用手写40 Byte包代替。
- 802.11 MAC ACK与Block ACK由Wi-Fi MAC产生，并与TCP ACK分开统计。
- 旧项目研究CW增长默认值为`false`；TGax程序没有调用启用接口。现有队列补丁只补发原本漏掉的Drop trace，不改变丢弃判定。
- 所有AP当前均为Default EDCA。BSS 0只做醒目标记，尚未接入RL。
- `subset7`始终标为正式7 BSS研究子集；`main3`只作协议调试，`full19`只保留配置，均不冒充已完成的完整19 BSS正式实验。

## 处理优先级

1. 环境审核前必须保留D01、D06、D08、D09、D10的显式说明。
2. 如果后续论文结论依赖PHY PER或MIMO，应先解决D01或改写研究边界。
3. 如果后续采用长时HTTP/FTP负载，应先补齐D09、D10并重新做业务统计验证。
4. 只有在引入MU调度研究时才处理D13；本轮不应为此删除或简化SU HE功能。
