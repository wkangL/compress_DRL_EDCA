# TGax TCN-DDQN第一版训练报告

日期：2026-10-23  
最终有效运行：`results/raw/tgax_ddqn_v1_small_seed41_r4_20261023`  
场景：`candidate_voip8_gaming4_ftp_bk`，7 BSS × 40 STA，仅 BSS 0 可控

## 1. 结论

本轮完成了 BSS 0 的持续在线观测、Async+Age 压缩反馈、动态 EDCA/CW 动作、Reward、Replay 和 Branching Double DQN 的闭环迁移。Default 等价验证、动作真实生效验证、邻居隔离和 MAC 账本检查均通过；单 seed 小规模训练没有 NaN、Inf、Q 值爆炸或梯度爆炸。

第一版策略**尚不能证明算法有效或已收敛**。在相同评估 seed、相同外生业务需求下，学习策略的平均 Reward 从 Default 的 1.2669 降为 1.2380（下降 2.28%）。它改善了 VO 上行 p95、VI 下行吞吐和 BE 上行服务，但恶化了 VO 下行尾时延/丢包、VI 上行服务、VI 下行尾时延/重传以及 BE 下行丢包。10 个环境步只足以验证闭环和数值稳定性，不足以完成策略学习。

## 2. 训练前发现并修复的正确性问题

| 问题 | 影响 | 修复与验证 |
|---|---|---|
| TCP 业务模型在三次握手完成后才开始生成 FTP/HTTP 需求 | 控制器可通过拖延 TCP 建连让 BE/BK 看起来“没有需求” | 应用需求时钟改为独立运行；连接前生成的字节保存在应用待发送区，连接后再交给 TCP。包长、间隔、文件大小和阅读时间分布未改。Default/策略逐步生成需求完全一致，见 `evaluation_demand_audit.json`。 |
| 7 BSS 使用共同 SSID，20/280 个 STA 实际漫游到非配置 AP | 两个邻居 STA 接收了 AP0 广播的 EDCA 参数，违反“仅 BSS0 可控” | 改为每 BSS 独立 SSID，并在 STEP 启动前强制核对 280 个 BSSID；现为 280/280 匹配。跨 BSS 仍处于同一 80 MHz 信道并产生 PHY 竞争/干扰。 |
| 交互后端排空阶段误把绝对停止时刻当相对时长 | 仿真结束后多运行无关时段，污染流级汇总 | 改为从当前时刻仅继续 `drain=1 s`。 |
| 无 MAC 到达/尝试的 BK 上行在汇总中显示 0 丢包率/重传率 | 容易把无样本解释成良好表现 | 最终汇总改为 `null/—`；原始账本保留。 |

相关实现位于 `ns3_ext/tgax-control.cc`、`ns3_ext/tgax-s3.cc` 和 `scripts/run_tgax_ddqn_v1.py`。

## 3. 固定环境与 Default

- 7 个同信道 BSS，每 BSS 1 AP + 40 STA，共 7 AP、280 STA。
- BSS 0 的 AP 和 40 个关联 STA统一接受动作；BSS 1–6 始终保持 Default。
- 802.11ax HE SU，5 GHz，80 MHz，固定 HE MCS 7；真实 IPv4/TCP/UDP、A-MPDU、Block ACK 和重传保留；未启用 OFDMA/MU-MIMO。
- 业务保持 `candidate_voip8_gaming4_ftp_bk`：VoIP→VO，视频/Gaming→VI，HTTP→BE，FTP→BK。
- 每次评估先关联预热 2 s，再用 Default 原生 BEB 做共同业务预热 1 s，随后评估 5 个 1 s 窗口，最后排空 1 s。

Default 的 EDCA 为：

| AC | AIFSN | CWmin | CWmax | TXOP |
|---|---:|---:|---:|---:|
| VO | 2 | 3 | 7 | 2.080 ms |
| VI | 2 | 7 | 15 | 4.096 ms |
| BE | 3 | 15 | 1023 | 0 |
| BK | 7 | 15 | 1023 | 0 |

Profile 0 已更新为上表，但 Default 基线还会明确关闭研究型 CW 增长，使 ns-3 原生 BEB 生效。

## 4. 观测、反馈与网络

每 0.1 s 为每个 STA、每个 AC 采样 6 个区间特征：队列字节数、成功 MPDU 的 MAC 最终服务时延均值、重传次数、最终失败次数、丢弃次数和时延有效标记。历史张量为 `40 × 4 × 10 × 6`。

共享 TCN 对每个 STA/AC 的 10 点历史编码：3 个因果卷积残差块，通道数 32、卷积核 3、膨胀率 1/2/4，时间池化后线性输出 8 维。STA 聚合器按 AC 使用共享 MLP、attention pooling、max pooling，并加入报告 age 均值、p95 和覆盖率，每个 AC 67 维，共 268 维。

AP 本地状态为 10 维：信道忙比例、四类 AC 上行成功 goodput、AP 四队列字节和关联 STA 比例。当前动作使用 16 维 Profile one-hot 加 4 个归一化连续候选索引，共 20 维。因此 DDQN 输入为 `268 + 10 + 20 = 298` 维。

Async+Age 报告仍使用“STA 成功上行数据 MPDU 获得确认后，抽象地产生/到达报告”的门控。报告缓存保存最新历史、age 和 valid；TCN 只在报告时间变化时重编码。该机制尚未承载真实无线反馈比特，不能称为标准信令部署。

## 5. 动作与实际控制

五个动作分支为 `(Profile, α, β1, β2, γ)`：

- Profile：16 个候选；
- α：`[1,2,3]`；
- β1：`[1.0,1.2,1.4,1.6,1.8,2.0]`；
- β2：`[1.0,1.2,...,3.0]`；
- γ：`[5,6,7]`；
- 约束：`β2 ≥ β1`，共 `16 × 3 × 51 × 3 = 7344` 个合法组合。

失败重试计数为 `r` 时，若 `r≤γ`，研究型更新为：

`CW' = min(CWmax, ceil(β(CW+1)) - 1)`，其中 `r≤α` 使用 β1，否则使用 β2。超过 γ 后 CW 保持；关闭研究规则时使用 ns-3 原生 BEB。

动作每 1 s 更新一次，同时作用于 AP0 和其固定关联的 40 STA。最终隔离审计检查了学习评估中的 5,904 条邻居 EDCA 记录，非 Default 行为 0；BSS0 有 820 条非 Default 行。动作生效验证的单步 P5 检查中，BSS0 的 `41设备 × 4AC = 164` 行全部改变，邻居 1,968 行全部保持 Default。

最终贪心策略首个窗口选择 P8、`α=2, β1=1.6, β2=2.2, γ=6`，后四个窗口选择 P8、`α=2, β1=1.6, β2=2.8, γ=6`。P8 将 VO/VI TXOP 均设为 6.016 ms，并将 BE/BK CWmin 提高到 31/63；这解释了下行 VI 吞吐上升，同时也造成方向与业务之间的明显取舍。

## 6. Reward

最终公式仅使用 BSS0 数据：

`R = S + 0.8 O_VO + 1.0 O_VI - 0.3 R_retry - 0.3 R_fail - 0.5 D_low - 0.02 C_switch`

- `S`：按 VO/VI/BE/BK 权重 `1.2/1.1/1.0/0.5` 对活动需求的应用交付满足率加权平均；未交付需求结转为下一窗口债务。
- `O_VO`：满足 20 ms 的最终 MAC 服务及时交付比例。
- `O_VI`：满足 50 ms 的最终 MAC 服务及时交付比例。
- `R_retry`：逐 MPDU 重传次数/发送尝试次数，明确不是碰撞率。
- `R_fail`：最终失败 MPDU/风险 MPDU。
- `D_low`：活动 BE/BK 的满足率低于 0.2 时的平均缺口。
- `C_switch`：Profile 是否变化和四个标量动作归一化变化的平均量。

Default 五个评估窗口中四类 AC 均被标记为有需求；BK 无成功服务，因此低优先级缺口均值为 0.2。学习策略的平均服务项略升（0.0537→0.0558），重传项略改善，但 VO/VI 及时率下降且存在动作切换代价，最终 Reward 下降。

## 7. DDQN 与训练参数

TCN/聚合后的 298 维状态经过 `256→128` ReLU 主干。Dueling Branching 头包含共享 V 和五个中心化 advantage 分支，输出尺寸为 `16/3/6/11/3`。β1/β2 使用联合合法掩码选择，其余分支独立取最大值。

Double DQN 由 online 网络选择下一动作、独立 target 网络评估该动作；五分支 target Q 取均值后 Bootstrap。episode 截断仍允许 Bootstrap，真正终止才置零。Target 是 online 的深拷贝，采用每次更新 `τ=0.001` 的 soft update。

| 参数 | 数值 |
|---|---:|
| seed | 41 |
| 环境步 | 10（2 阶段 × 5） |
| 梯度更新 | 50（每阶段 25） |
| Replay 容量 / batch | 1000 / 4 |
| warmup | 4 transitions |
| 折扣因子 | 0.98 |
| Q 学习率 | 3e-4 |
| TCN/聚合学习率 | 1e-4 |
| 优化器 | AdamW |
| ε | 1.0→0.1，10 步线性；本轮最后训练步为 0.19 |
| 梯度裁剪 | 10 |
| Target | soft，τ=0.001 |

## 8. 正确性与稳定性验证

### Default 等价

固定关联后，以同一 seed 分别运行一次性 Default 后端和持续 STEP 后端。63 个非空 BSS/方向/AC 行的 `arrived/acked/failed/retransmissions/tx_attempts` 完全一致，差异数为 0；280 个 STA 的关联错误数为 0。

### 训练稳定性

| 诊断 | 前 10 次更新均值 | 后 10 次更新均值 | 全程范围 |
|---|---:|---:|---:|
| TD loss | 0.4627 | 0.1553 | 0.0642–0.8264 |
| online Q mean | 0.3819 | 1.2334 | 0.1312–1.5441 |
| target Q mean | 0.2136 | 0.1635 | 0.1271–0.2319 |
| 梯度范数 | 2.1683 | 0.9824 | 0.2894–3.0615 |

全部值有限，最大 Q 为 2.371，未触发中止阈值。Q-loss 的下降说明这 50 个小批量更新没有数值发散；online Q 上升主要是在拟合约 1.0–1.6 的 TD target，不能据此宣称策略收敛。Target Q 变化较慢与 `τ=0.001`、仅 50 次更新一致，目前没有证据支持修改同步方式。

## 9. 同 seed 独立评估

下表的“改善率”统一为正数表示更好；时延/丢包/重传越低越好，吞吐越高越好。BK 上行没有 MAC 到达或成功样本，相关比例为未定义；BK 下行存在最终失败但无成功服务。

| 指标 | Default | 学习策略 | 改善率 |
|---|---:|---:|---:|
| VO UL p95 | 72.17 ms | 53.76 ms | +25.50% |
| VO DL p95 | 28.06 ms | 34.25 ms | -22.09% |
| VI UL p95 | 58.60 ms | 63.11 ms | -7.69% |
| VI DL p95 | 21.08 ms | 23.10 ms | -9.60% |
| VO UL 吞吐 | 0.0415 Mbps | 0.0392 Mbps | -5.45% |
| VO DL 吞吐 | 0.0383 Mbps | 0.0382 Mbps | -0.18% |
| VI UL 吞吐 | 0.1911 Mbps | 0.1759 Mbps | -7.95% |
| VI DL 吞吐 | 1.0333 Mbps | 1.4981 Mbps | +44.97% |
| BE UL 吞吐 | 0.0061 Mbps | 0.0226 Mbps | +267.14%（绝对值仍低） |
| BE DL 吞吐 | 0.0018 Mbps | 0.0034 Mbps | +85.96%（绝对值仍低） |
| VO UL 最终丢包率 | 13.57% | 18.38% | -35.48% |
| VO DL 最终丢包率 | 2.07% | 2.31% | -11.92% |
| VI UL 最终丢包率 | 3.94% | 3.79% | +3.86% |
| VI DL 最终丢包率 | 0 | 0 | 持平 |
| BE UL 最终丢包率 | 80.41% | 19.21% | +76.11% |
| BE DL 最终丢包率 | 64.71% | 91.67% | -41.67% |
| Reward | 1.2669 | 1.2380 | -2.28% |
| Jain 需求满足公平性 | 0.4867 | 0.4930 | +1.29% |

Jain 指标按实际有应用需求的 STA 的 `min(received/offered,1)` 计算，包含双方相同的 1 s Default 业务预热和 5 s 评估。短窗口内低速/长阅读间隔业务样本较少，因此该值只作本次同 seed 辅助比较。

## 10. 交付物与下一步判断

- 完整改善率：`results/summary/tgax_ddqn_v1_fixed_assoc_20261023/evaluation_comparison.csv`
- 修正后的结果 JSON：`results/summary/tgax_ddqn_v1_fixed_assoc_20261023/evaluation_result_corrected.json`
- Default 等价：`results/summary/tgax_ddqn_v1_fixed_assoc_20261023/default_equivalence.json`
- 动作隔离：`results/summary/tgax_ddqn_v1_fixed_assoc_20261023/action_validation.json`
- 需求一致性：`results/summary/tgax_ddqn_v1_fixed_assoc_20261023/evaluation_demand_audit.json`
- 数值诊断：`results/summary/tgax_ddqn_v1_fixed_assoc_20261023/training_diagnostics.json`
- 训练图：`results/figures/tgax_ddqn_v1_small_seed41_r4_20261023/training_diagnostics.png/.pdf`
- VO/VI 时延 CCDF：`results/figures/tgax_ddqn_v1_small_seed41_r4_20261023/vo_vi_service_delay_ccdf.png/.pdf`
- 核心指标图：`results/figures/tgax_ddqn_v1_small_seed41_r4_20261023/bss0_core_metrics.png/.pdf`

进入正式训练前，最小的下一步是延长单 seed 环境交互和独立评估窗口，确认 Reward 与动作分布进入稳定区间，再决定是否扩展多 seed。本轮没有证据要求修改 Reward、Target 同步或网络结构，也没有启动多 seed 训练。
