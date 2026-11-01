# TCN-DDQN训练前配置确认

审查日期：2026-10-23  
固定候选环境：`candidate_voip8_gaming4_ftp_bk`  
审查范围：现有源码、已解析配置和已有Default EDCA结果；本次未启动训练，也未修改算法、Reward、动作空间或TGax配置。

## 1. 固定TGax仿真环境

### 1.1 本轮确认的研究对象

| 项目 | 当前实际配置 | 代码或数据依据 |
|---|---|---|
| 网络规模 | 7个同信道BSS，每BSS 40个STA，共7 AP、280 STA | `configs/tgax_s3_7bss.json:2-12` |
| 控制对象 | 未来仅BSS 0；当前Baseline全部Default EDCA | `configs/tgax_s3_7bss_candidate_voip8_gaming4_ftp_bk.json:3-8` |
| PHY | 802.11ax HE SU，HeMcs7，80 MHz，5 GHz，800 ns GI | `ns3_ext/tgax-s3.cc:1207-1213,1228-1279` |
| 多用户调度 | 未配置UL/DL OFDMA或MU-MIMO调度器；单空间流发送 | `ns3_ext/tgax-s3.cc:1207-1214,1235-1279` |
| 协议 | 真实IPv4、TCP/UDP；A-MPDU上限65535 Byte，A-MSDU关闭，Block Ack缓冲64 MPDU | `ns3_ext/tgax-s3.cc:1003-1015,1184,1363-1402` |
| 业务数量/BSS | BV6-DL 1、BV3-DL/UL各1、Gaming-DL/UL各4、HTTP-DL 1、FTP-DL/UL各1、双向VoIP 8对，共30个单向组件 | `configs/tgax_s3_7bss_candidate_voip8_gaming4_ftp_bk.json:10-32` |
| AC映射 | VoIP→VO；BV3/BV6/Gaming→VI；HTTP→BE；FTP→BK | `configs/tgax_s3_7bss_candidate_voip8_gaming4_ftp_bk.json:24-29` |
| 已有20 s运行 | 2 s预热、20 s测量、1 s排空、seed 20261021、Yans、summary、Default EDCA | `results/raw/tgax_subset7_candidate_voip8_gaming4_ftp_bk_20s_20261022/.../run_manifest.json` |

该配置是“TGax业务模型驱动的自定义并发场景”，不是完整TGax Scenario 3官方业务数量。配置文件当前明确写有`rl_enabled: false`，所以它尚不是可训练环境。

还需注意两项实际运行参数：TGax后端当前把`FrameRetryLimit`设为10，MAC队列为每AC 500 packet、最大排队时间500 ms；A-MPDU与Block Ack保持启用。它们来自实际运行导出的`edca_runtime.csv`和`ns3_ext/tgax-s3.cc:1003-1015`，不能用旧单BSS配置中的重试7、队列寿命10 s替代。

20 s Default结果表明BSS 0有40个关联STA、22个实际业务STA，信道忙比例0.791885。VO/VI已经处于明显竞争状态，但该seed中BK上行没有MPDU进入MAC；BK下行也只有7个到达、5个成功。该事实不改变本轮固定场景，但会直接影响Reward中低优先级项的统计可信度。

## 2. 观测空间与TCN

### 2.1 旧控制框架的STA观测

旧`wifi-control`后端每0.1 s为每个STA、每个AC输出一次区间样本。名称来自`configs/default.yaml:6-9`，真实生成逻辑位于`ns3_ext/wifi-control.cc:308-331`。

| 特征 | 实际定义 | 单位 | AP部署可获得性 | TGax状态 |
|---|---|---:|---|---|
| `queue_bytes` | 采样时该STA对应AC队列的`GetNBytes()` | Byte | STA本地可测，需反馈 | TGax现有队列采样CSV含队列信息，但尚未接入在线报告 |
| `access_delay_ms` | 本采样周期内成功MPDU从旧后端`Send()`打tag到成功ACK的平均时延；只有成功样本 | ms | STA MAC可测，需反馈 | 名称有误导性；不是“首次接入时延”。TGax已有严格MAC入队至ACK服务时延账本，但未形成0.1 s STA状态 |
| `retry` | 本周期带Retry标志的发送次数 | 次/0.1 s | STA本地可测，需反馈 | TGax能逐MPDU累计重传，但未输出在线STA状态 |
| `failed` | 本周期达到重试上限的最终失败数 | 次/0.1 s | STA本地可测，需反馈 | TGax账本已实现，未迁移到在线状态 |
| `discarded` | 本周期非重试上限MAC丢弃数，包括入队失败、过期及其他丢弃 | 次/0.1 s | STA本地可测，需反馈 | TGax账本已区分原因，未迁移到在线状态 |
| `delay_valid` | 本周期是否至少有一个成功MPDU；0时强制把时延置0 | 0/1 | STA本地可测，需反馈 | 未接入 |

历史张量由`reporting/sta_state.py:6-29`维护。现有配置`history_s=1.0`、`sta_sample_s=0.1`，因此：

\[
X_{STA} \in \mathbb{R}^{4\times10\times6},\qquad
X_{BSS} \in \mathbb{R}^{N\times4\times10\times6}.
\]

旧配置中`N=10`，迁移到BSS 0后应为`N=40`，即`40×4×10×6`。Python容器、TCN和聚合器都从`config["network"]["num_sta"]`读取N，没有在这些模块中写死10；写死10的是现有旧训练配置本身（`configs/default*.yaml`）。必须新建TGax训练配置才能得到40，不能直接沿用旧配置。

`StateScaler`对前5个特征做`log1p`，时延先裁剪到10000 ms，再用仅由训练集Default策略rollout拟合的均值/标准差标准化；无效时延最后恢复为0（`reporting/sta_state.py:32-57`）。旧S1–S5拟合出的scaler不能直接用于TGax，应在迁移后用固定候选环境重新拟合；这属于迁移必需步骤，不是本轮参数调整。

### 2.2 TCN结构与训练方式

| 部分 | 当前真实实现 |
|---|---|
| 共享方式 | 所有STA、所有AC共用同一个TCN |
| 输入 | 每个STA-AC的`10×6`历史 |
| 因果卷积 | 3个残差`Conv1d+ReLU`块，channels=32，kernel=3，dilation=`[1,2,4]` |
| 感受野 | 15个采样点，大于当前10点窗口，因此覆盖整个1 s历史 |
| 时间汇聚 | 对时间维求均值 |
| 输出 | 线性层输出8维latent；整体为`N×4×8` |
| 训练 | 与Q网络通过TD loss端到端训练；每个episode内encoder冻结，stage末集中更新，版本加1 |
| 辅助任务 | 配置中`aux_enabled=false`，当前没有辅助损失 |

依据：`models/sta_encoder.py:6-32`、`models/branching_ddqn.py:23-41`、`rl/trainer.py:226-270`。

### 2.3 Fresh、Async、Age与有效标记

- `compressed_fresh`直接读取所有增强STA的最新历史，`age=0`。这是理想信息上界，不是当前可部署反馈。
- `compressed_async_age`只使用AP缓存中最后一次成功报告；`age=当前决策时刻-报告生成时刻`，按1 s缩放、30 s裁剪；`valid=1`表示至少收到过一个匹配encoder版本的报告。
- `compressed_async_no_age`使用同一缓存，但将age清零。
- 新报告只有生成时间严格更新且encoder版本一致才替换缓存；版本升级会清空缓存。

旧后端在STA一次实际上行数据TXOP时发出`G`，对应MPDU成功ACK后发出`R`，失败则发出`X`；Python在`G`时复制历史，在`R`时才把它放入AP缓存（`reporting/report_generator.py:8-31`、`reporting/report_cache.py:16-43`、`ns3_ext/wifi-control.cc:132-149,218-223`）。但这不是一个真实承载latent或历史字节的802.11反馈帧：没有序列化、空口字节和发送开销，只是用数据MPDU的ACK事件门控模拟报告到达。TGax后端目前连这套门控事件也没有。

### 2.4 AP本地观测与40 STA聚合

AP全局向量固定10维（`sim/statistics/ap_state.py:5-38`）：

| 变量 | 来源与归一化 | 是否全局越权 | TGax状态 |
|---|---|---|---|
| `busy_fraction` | AP 0 PHY非IDLE时间区间并集/1 s | 否，AP本地CCA/PHY状态 | TGax已离线统计，未接入STEP |
| `ul_{vo,vi,be,bk}_goodput` | 本周期AP收到的各AC上行成功Byte换算Mbps，再除旧`C_ref` | 否 | TGax有成功账本；旧`C_ref`不适用，在线归一化未迁移 |
| `ap_{vo,vi,be,bk}_queue` | AP四个AC队列Byte，经`log1p(q)/log1p(10^6)` | 否 | TGax可采集，未接入STEP |
| `associated_sta_count` | 关联STA数/64并裁剪到1 | 否 | TGax已知为40，未接入STEP |

聚合器对每个AC把8维latent、STA age和valid拼成10维输入，经`10→32→32`共享MLP；对STA集合分别做attention pooling和masked max，再追加age mean、age p95、coverage三个统计量。每AC输出`32+32+3=67`维，四个AC共268维（`models/sta_aggregator.py:6-38`）。再拼接AP 10维和当前动作20维，DDQN最终输入为：

\[
10+268+20=298\text{维}.
\]

聚合对STA排列不敏感，N从10变40不会改变298维DDQN输入。不存在网络层面的10-STA固定维度；存在的是旧配置、旧后端拓扑和旧反馈事件只为单BSS编写。

旧观测没有直接读取邻居BSS队列、邻居动作或未来信息。Fresh方法直接访问全部STA历史，是仿真oracle；Async+Age的变量在概念上均可由AP 0本地观测和STA反馈获得，但TGax尚未实现真实或等效的报告传输。

## 3. 动作空间与CW增长规则

动作定义为

\[
a_t=(P_t,\alpha_t,\beta_{1,t},\beta_{2,t},\gamma_t).
\]

### 3.1 16种EDCA Profile

单元格顺序均为`AIFSN/CWmin/CWmax/TXOP_ms`，数据逐项来自`configs/profiles.json`，合法性检查见`sim/edca/profile_table.py:11-27`。

| P | VO | VI | BE | BK |
|---:|---|---|---|---|
| 0 | 2/3/7/1.504 | 2/7/15/3.008 | 3/15/1023/0 | 7/15/1023/0 |
| 1 | 2/3/15/1.504 | 2/7/31/3.008 | 3/31/1023/0 | 7/31/1023/0 |
| 2 | 2/7/15/1.504 | 2/15/31/2.528 | 4/31/511/0 | 9/63/1023/0 |
| 3 | 2/3/7/3.008 | 2/7/15/4.096 | 3/15/511/0 | 7/31/1023/0 |
| 4 | 2/7/31/2.528 | 3/15/63/3.008 | 4/31/1023/0 | 7/63/1023/0 |
| 5 | 2/3/15/4.096 | 2/7/31/6.016 | 5/63/1023/0 | 9/127/1023/0 |
| 6 | 3/7/31/1.504 | 3/15/63/2.528 | 3/15/255/0 | 7/31/511/0 |
| 7 | 2/15/63/1.504 | 3/15/127/2.528 | 4/31/255/0 | 7/63/511/0 |
| 8 | 2/3/7/6.016 | 2/7/15/6.016 | 3/31/1023/0 | 7/63/1023/0 |
| 9 | 2/7/15/3.008 | 2/7/31/4.096 | 5/63/1023/0 | 9/127/1023/0 |
| 10 | 3/15/63/1.504 | 3/31/127/2.528 | 3/31/511/0 | 7/63/1023/0 |
| 11 | 2/3/31/2.528 | 2/15/63/3.008 | 4/63/1023/0 | 9/127/1023/0 |
| 12 | 2/7/31/4.096 | 2/7/63/6.016 | 3/127/1023/0 | 7/255/1023/0 |
| 13 | 2/31/127/1.504 | 3/31/255/2.528 | 4/63/511/0 | 7/127/1023/0 |
| 14 | 3/15/127/3.008 | 3/31/255/4.096 | 5/127/1023/0 | 9/255/1023/0 |
| 15 | 2/3/7/1.504 | 2/15/31/3.008 | 3/255/1023/0 | 7/255/1023/0 |

TGax当前Default实测为VO `2/3/7/2.080`、VI `2/7/15/4.096`、BE `3/15/1023/0`、BK `7/15/1023/0`。因此旧P0并不等于当前TGax Default，尤其VO/VI TXOP不同。16个Profile源自旧802.11a实验，尚未接入TGax，也尚未完成HE环境固定动作合法性与可优化性复核。

### 3.2 标量候选及公式

| 分支 | 候选值 | 物理含义 |
|---|---|---|
| `alpha` | 1, 2, 3 | 前段/后段CW增长倍率的重试次数分界 |
| `beta1` | 1.0,1.2,1.4,1.6,1.8,2.0 | 第1至alpha次失败后的CW乘数 |
| `beta2` | 1.0至3.0，步长0.2 | alpha之后至gamma次失败后的CW乘数 |
| `gamma` | 5,6,7 | 使用自定义增长的最后一个重试计数；之后保持当前CW |

实际C++以十分位整数计算。第r次失败后：

\[
\beta(r)=\begin{cases}\beta_1,&r\le\alpha\\\beta_2,&\alpha<r\le\gamma\end{cases},\quad
CW_r=\min\{CW_{max},\lceil\beta(r)(CW_{r-1}+1)\rceil-1\}.
\]

当`r>gamma`时自定义分支不再增大CW；关闭自定义增长时回到ns-3标准公式`min(CWmax, 2^r(CWmin+1)-1)`。依据：`ns3_ext/cw_growth.patch:31-54`及已应用的`vendor/ns-3.44/src/wifi/model/txop.cc:387-405`。

非法组合`beta2<beta1`由DDQN联合mask排除，探索也只从合法pair采样；C++再次拒绝该组合。C++还拒绝`alpha>gamma`和越界Profile，而当前候选集合天然满足`alpha≤3<5≤gamma`。合法笛卡尔动作共`16×3×51×3=7344`种，网络不显式输出7344个Q值，而输出五个分支共`16+3+6+11+3=39`个值。

### 3.3 动作何时生效、控制了谁

旧后端通过持久ns-3进程的stdin接收每个1 s决策的`STEP`，在继续推进仿真时间前执行`Apply()`，因此能够在一次持续仿真中动态生效（`sim/backend/ns3_backend.py:112-129`、`ns3_ext/wifi-control.cc:535-556`）。但是旧`Apply()`会：

1. 把选定Profile写到单BSS的AP和全部STA四个AC；
2. 把自定义CW增长写到AP，以及`enhanced_fraction`覆盖的STA；旧配置为1.0，即AP和全部10 STA。

依据：`ns3_ext/wifi-control.cc:437-449`。这不是“只控制AP 0队列”。

TGax后端当前只有启动参数`--targetProfile`，只接受`default/rt-moderate/rt-aggressive/rt-protect`四个固定值，并且只在设备创建时对AP 0调用一次；STA、其他6个AP不变（`ns3_ext/tgax-s3.cc:1019-1058,1112-1130,1287-1298`）。它没有stdin STEP循环，没有16 Profile入口，也没有调用`SetResearchCwGrowth`。

所以当前五维动作在TGax中**尚未实现**；其最终范围究竟是“仅AP 0”还是“AP 0及其40个STA”也尚未由代码确定。训练前必须先固定这个研究定义并实现相同范围，且加入断言确保BSS 1–6从未被修改。

## 4. Reward公式及各项尺度

### 4.1 当前能确认的版本

TGax候选配置没有Reward段且`rl_enabled=false`，因此不存在“当前正在执行的TGax Reward”。旧入口`run.py`不指定配置时默认`configs/quick.yaml`，它执行Reward-v1；最近完成的三seed中等预算正式实验保存的不可变配置是`results/raw/reward_v2_medium_seed11_20261021/train/config.json`，执行Reward-v2。若目标是迁移“之前研究的TCN-DDQN”，能审查的最新实际版本是Reward-v2，但不能据此宣称它已被选为TGax第一版Reward。

Reward-v2实际公式（`rl/reward.py:5-12,30-53`）为：

\[
S=\frac{\sum_{a\in active}w_a\,clip(G_a/O_a,0,1)}{\sum_{a\in active}w_a},
\]

\[
r=1.0S+0.8O_{VO}+1.0O_{VI}-0.3C-0.3L-0.2(1-J)-0.02A-0.0D_{low}.
\]

| 项 | 实际统计与范围 | 权重 |
|---|---|---:|
| `S`吞吐需求满足度 | 每AC成功goodput/进入旧MAC的offered Mbps，裁剪[0,1]；AC权重VO/VI/BE/BK=`1.2/1.1/1.0/0.5` | +1.0 |
| `O_VO` | 20 ms内最终ACK的VO包/有资格判定的VO包，裁剪[0,1] | +0.8 |
| `O_VI` | 50 ms内最终ACK的VI包/有资格判定的VI包，裁剪[0,1] | +1.0 |
| `C` | 旧后端`collision_rate`，空口重叠归因数/air attempts，裁剪[0,1] | -0.3 |
| `L` | 所有AC最终失败/at-risk包；若无精确账本才回退到旧drop rate，裁剪[0,1] | -0.3 |
| `J` | 按全部STA成功Byte计算的Jain公平性，不是“需求满足公平性” | -0.2×(1-J) |
| `A` | Profile变化指示加四个归一化标量变化绝对值之和，再除5，范围[0,1] | -0.02 |
| `D_low` | BE/BK满足度低于服务下限的平均缺口 | 当前权重0、floor 0，实际不起作用 |

Reward按1 s决策周期计算，最终总Reward没有额外裁剪。虽然继承配置仍包含`vo_delay/vi_delay`字段，Reward-v2分支不使用它们。Reward-v1才使用p95超阈值惩罚：`clip(max(0,p95/T-1),0,2)`，阈值同为20/50 ms（`rl/reward.py:15-27`）。

旧Reward的时延tag在`WifiNetDevice::Send`前创建，到`AckedMpdu`时结算（`ns3_ext/wifi-control.cc:132-143,413-428`）；它接近MAC最终服务时延，但不是TGax现在严格采用的“`MacTx`首次进入MAC至成功ACK/Block ACK”。TGax账本已正确按每个MPDU记录后一口径（`ns3_ext/tgax-s3.cc:749-818,889-951`），但尚未实现20/50 ms在线deadline ledger。因此Reward不能直接复用旧统计对象。

### 4.2 用当前Baseline估计数值尺度

以下只是把TGax 20 s BSS 0账本代入同类定义的尺度检查，不是已经执行过的TGax Reward：

| 量 | Baseline尺度 |
|---|---:|
| MAC Byte满足度 VO/VI/BE/BK | 0.960/0.999/0.963/0.937；加权`S≈0.969` |
| VO 20 ms内完成率估计 | 约0.816（成功CCDF并保守计入最终失败） |
| VI 50 ms内完成率估计 | 约0.983 |
| 全AC最终失败率 | 约0.0178 |
| 全AC逐MPDU重传率 | 约0.419；它不是旧Reward的collision rate，不能直接代入 |
| 正奖励小计 | `S+0.8O_VO+O_VI≈2.605`，尚未扣碰撞、失败、公平性与切换 |

VO DL/UL p95为26.79/36.10 ms，已经越过20 ms，Reward-v2的VO按时率能够产生明显信号。VI DL/UL p95为28.81/34.83 ms，低于50 ms；只有约1%–2%的成功尾部和最终失败会改变VI项，梯度信号明显弱于VO。

存在三类风险：

1. `O_VO/O_VI`是阈值内/外的二值服务结果。全部低于阈值后达到1，不再奖励从15 ms降到5 ms；Reward-v1的p95惩罚也有同样的阈值以下平台。
2. Reward-v2的低优先级保护当前权重为0。BE/BK只有吞吐满足度中的1.0/0.5权重和全局公平性间接保护；BK在本Baseline中样本极少且UL为0，无法形成稳定学习信号。
3. TGax使用真实TCP，动作可能通过拥塞反馈降低应用发送量。若仍以“当前观察到的MAC offered”为分母，业务被压低甚至某AC暂时无offered时，会从active集合消失，存在以减少入MAC需求换取较高满足率的风险。Reward必须以外生需求或守恒账本核查这一点，但本轮不修改公式。

Reward还必须限定为BSS 0。旧单BSS代码天然只有一个BSS；TGax汇总已能按BSS分账，但尚无Reward适配器。邻居信息不能混入Reward，邻居只通过真实干扰影响BSS 0结果。

## 5. DDQN网络与训练超参数

### 5.1 网络与Double DQN实现

298维状态先经`Linear(298,256)-ReLU-Linear(256,128)-ReLU`。之后有一个共享标量Value头和五个Advantage头，输出尺寸`[16,3,6,11,3]`；每个分支使用

\[
Q_i=V+A_i-mean(A_i).
\]

beta1/beta2通过联合mask选择，其余分支独立argmax（`models/branching_ddqn.py:8-48`）。

TD更新中，Online网络在下一状态选择五个动作，独立Target网络评价这些动作；五个目标分支Q取平均得到单一bootstrap值，再把同一个TD目标用于五个已选分支的Smooth L1 loss（`rl/trainer.py:102-136`）。这满足Double DQN的“Online选择、Target评价”。

Target在创建时由`deepcopy(model)`得到，是独立副本，并被设为eval、`requires_grad=False`。当前正式历史配置采用每次梯度更新都执行的soft update，`tau=0.001`；`target_update_steps=500`仅在hard模式生效（`rl/trainer.py:42-68,181-185`）。

### 5.2 最新历史正式配置，而非TGax已选配置

| 参数 | Reward-v2 medium三seed实际值 |
|---|---:|
| Replay容量 / batch | 200000 / 128 |
| 开始更新阈值 | 128条transition |
| 折扣因子 | 0.98 |
| Q头学习率 / TCN+聚合器学习率 | 3e-4 / 1e-4 |
| 优化器 | PyTorch 2.12 `AdamW`；代码未覆盖默认betas=(0.9,0.999)、eps=1e-8、weight_decay=0.01 |
| 梯度裁剪 | 全参数norm上限10 |
| epsilon | 1.0线性降至0.05，用800个环境步完成 |
| Target | soft，tau=0.001，每次update |
| 每stage | 4 episode×10 s，每1 s决策；40个环境步；随后200次梯度更新 |
| 总stage | 20 |
| 每seed | 800个环境交互步；第4个stage达到warmup后，共3400次梯度更新 |
| 训练seed | 11、22、33（历史实验） |

配置依据：`configs/reward_v2_medium.yaml:2-16`及各历史run的`train/config.json`。Replay保存原始历史而非旧latent，均匀无放回采样（`rl/replay_buffer.py:8-32`）。

Episode最后一步写`truncated=True`但`done=False`；TD只用`done`屏蔽bootstrap，因此时间上限截断仍bootstrap。这对连续运行的EDCA控制任务是正确处理。当前环境从不生成真实terminal（`rl/trainer.py:245-254`）。训练以stage数结束，也可在stage边界人工暂停；非有限loss/梯度会抛异常。

### 5.3 Q-loss、Q值与Reward监控

现有代码已经记录：

- `losses.csv`：Smooth L1 TD loss、Online Q、Target Q、TD target、TD error、Reward batch均值、bootstrap比例、梯度norm、latent各维方差、Target距离及同步标记；
- `steps.csv`：每步总Reward、各Reward分项、epsilon、动作、场景和MAC指标；
- `episodes.csv`：episode平均Reward；
- stage checkpoint和独立seed评估曲线。

位置：`rl/trainer.py:121-142,252-280`、`rl/monitoring.py:24-101`、`scripts/plot_monitoring.py`。

Q-loss不要求逐步下降；应观察平滑loss、TD target、Online/Target Q距离和独立评估是否同时稳定。当前只对loss和梯度norm做有限性检查，NaN/Inf会立即抛`FloatingPointError`。Q爆炸通常会通过非有限loss被截断，但代码没有“Q绝对值持续过大”或“平滑loss持续上升”的在线阈值，也不会保存导致异常的batch诊断快照。因而不能仅凭短期Q-loss上升自动修改Target同步频率；若迁移后出现异常，应先检查Reward尺度、TD目标、统计守恒和Target距离。

## 6. TGax迁移兼容性

| 能力 | 旧单BSS框架 | TGax当前实现 | 结论 |
|---|---|---|---|
| BSS 0 STA/AP观测 | `S/B/D/P/G/R/X`事件完整 | 只在整次运行结束后写CSV；没有旧事件协议 | **未迁移** |
| 每秒向DDQN给状态 | 持久进程支持1 s `STEP` | `Simulator::Run()`一次跑到结束 | **未迁移** |
| 仿真不中断动态动作 | 每个STEP前`Apply()` | 只有启动时固定`targetProfile` | **未迁移** |
| 16 Profile | 已加载旧表 | 只有4个手写固定Profile | **未迁移** |
| 自定义CW增长 | vendor patch及旧后端调用已实现 | TGax未调用`SetResearchCwGrowth` | **未迁移** |
| 动作控制范围 | 单BSS AP+STA | 固定Profile仅AP 0 | **定义冲突，必须先确定** |
| 动作后Reward | 旧1 s账本可算v1/v2 | TGax有离线BSS/方向/AC账本，但无1 s Reward和deadline ledger | **部分基础已有，适配未完成** |
| Python连续交互 | `Ns3Backend` stdin/stdout协议 | TGax runner只启动完整独立进程 | **未迁移** |
| 多seed | 旧训练器和评估器支持 | TGax批处理支持多个独立seed | **两边各自具备，尚未接成训练流程** |

`sim/backend/ns3_backend.py:23-29,45-76`还硬性验证802.11a、20 MHz、无A-MPDU、无Block Ack，并查找`wifi-control`可执行文件。即使只把`num_sta`改成40，也会因TGax的802.11ax/80 MHz/A-MPDU配置被拒绝。原接口不能直接复用。

迁移时可保留Python的TCN、聚合器、BranchingQ、Replay、Double DQN和日志；必须新增TGax专用持久后端/适配器，在同一个`Simulator`生命周期内分段推进，并由BSS 0事件生成严格同口径状态和Reward。不能通过每秒重启ns-3替代，因为那会重置TCP、队列、退避、信道竞争和业务随机过程。

## 7. 训练前必须解决的问题

### 7.1 必须解决

1. **建立TGax持续STEP接口。** 让`tgax-s3`在2 s关联预热后保持进程，按1 s决策窗口继续同一仿真，并返回BSS 0观测、完整账本和仿真时间；关闭时再排空。当前一次性runner不能训练。
2. **迁移BSS 0观测。** 按40 STA×4 AC×10×6生成状态，定义TGax中`access_delay_ms`为一致的MAC最终服务时延区间统计，并从AP 0 PHY、队列、成功接收构造10维本地状态。禁止读取邻居内部队列或未来事件。
3. **实现并核验Async+Age反馈。** 明确报告是实际管理/数据帧、piggyback，还是经审计的抽象反馈；无论哪种方式，都要保持Fresh/Async/valid/version语义并记录反馈开销。旧G/R事件不能自动套用真实TCP流。
4. **固定动作控制范围。** 在“只改AP 0”与“改AP 0及其40 STA”中作出研究定义，然后只实现该范围；对BSS 1–6逐决策断言Default EDCA不变。当前旧/新后端行为冲突。
5. **把16 Profile和CW增长接入TGax动态MAC。** 处理P0不等于TGax Default的问题，验证AIFSN/CW/TXOP即时更新对已有QosTxop生效，并在每步导出实际值；调用已有CW patch并验证retry计数、成功复位及gamma后的行为。
6. **统一Reward账本。** Reward只使用BSS 0、每1 s窗口内的MAC入队至最终ACK/Block ACK结果；实现20/50 ms因果deadline ledger、最终失败、未完成和offered/acked Byte守恒。不能把TGax重传率冒充collision rate。
7. **确定TGax第一版训练配置。** 当前没有把候选环境、40 STA、Reward版本、训练seed和预算绑定起来的配置文件。必须在审核后明确选择；本报告中的Reward-v2 medium参数只是最后一个历史正式配置。
8. **处理低优先级可学习性。** 当前Baseline的BK UL为0、BK DL仅7个到达，且Reward-v2低优先级显式权重为0。开始训练前至少要证明在选定episode长度与多个训练seed中BE/BK统计不是系统性空值，并确认Reward不会因TCP offered下降而获利；无需在本轮改业务数量。
9. **迁移后做只读等价验证。** 动作固定为Default时，新STEP环境拼接得到的20 s结果应与现有一次性TGax Baseline在统计误差内一致；同时验证BSS 1–6参数从未变化。通过后才能启动第一条训练轨迹。

### 7.2 后续可以优化

- 为非有限或Q绝对值异常保存触发batch、模型与优化器快照，并增加可配置的诊断停止阈值。
- 比较真实反馈帧与抽象piggyback的空口开销；在第一版接口正确性之后再做。
- 在第一版训练跑通后，再评估时延阈值以下缺少连续改善激励、需求公平性和低优先级服务保障是否需要Reward消融。
- 对16个旧Profile在HE环境做固定动作预筛选，删除等价或明显劣化动作；这属于后续动作空间优化，本轮不执行。

## 审查结论

TCN、STA集合聚合、Branching Double DQN、独立Target、Replay、soft target和Q-loss/Q值诊断在旧框架中都有可执行实现，网络维度也能数学上容纳40 STA。当前不能启动TGax训练的原因不是神经网络缺失，而是TGax后端尚未接入在线观测、持续STEP、动态动作和同口径Reward。尤其需要先解决动作控制范围冲突、P0与TGax Default不一致、Reward-v2低优先级保护为0及BK样本稀少四个问题。完成第7.1节的最小迁移与Default等价验证后，才具备第一版训练条件。
