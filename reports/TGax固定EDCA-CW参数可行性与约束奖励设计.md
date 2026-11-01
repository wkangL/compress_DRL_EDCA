# TGax固定EDCA/CW参数可行性与约束奖励设计

日期：2026-10-28  
研究环境：`candidate_full40_multiac_v6`，7个同信道BSS、每BSS 40 STA，仅BSS0受控；BSS1–6始终为Default EDCA与原生BEB。  
本阶段只做固定动作实验和Reward设计审计，**没有启动DDQN训练，也没有实现Reward-v4**。

## 1. 结论先行

1. **当前动作空间确实存在可实现的Default改进空间。** 在seed 44/45的20秒配对实验中，`P0+CW-late24`的两seed均值相对Default：应用有效吞吐`+90.16%`，VO p95降低`12.07%`，VI p95降低`13.67%`，Jain提高`2.35%`，是本次五个正式候选中唯一在三个主目标上严格支配Default的动作。
2. 这个结论仍是**探索性证据，不是普适最优证明**。吞吐收益主要来自seed 44（0.4133→1.0165 Mbps）；seed 45几乎持平（0.2567→0.2576 Mbps）。不过两个seed的VO/VI p95和Jain都向好的方向变化，因此不是只挑有利seed得出的结论。
3. 当前最好候选保持P0的AIFSN、CWmin、CWmax和TXOP不变，只替换CW增长轨迹，说明本次改善主要来自**研究型CW增长规则**，不是更换EDCA Profile。P3原生BEB仅带来平均吞吐`+3.19%`，同时VO/VI p95分别恶化`16.69%/7.38%`。
4. 动态DDQN的必要性尚未得到证明：目前一个固定动作在两个正式seed上都没有被Default反超。只有在不同负载阶段、业务状态或邻居竞争强度下出现“最优动作交叉”，动态状态反馈和动作切换才有额外价值。
5. Default的总体应用有效载荷/生成需求均值为`5.22%`，与Reward-v3留出测试中的`4.70%`一致。主要瓶颈在MAC之前：只有`9.63%`的独立生成需求在测量窗内被socket接受；VI的独立需求约14.6–15.7 MB/seed，但socket提交只有约0.48–0.49 MB。这个比例是**外生需求满足率**，不是MAC交付率，不能直接当吞吐Reward的固定归一化基准。

## 2. 当前实现审计

### 2.1 865个动作与Default回退

结构枚举结果为：

- 16个EDCA Profile；
- 每个Profile有54个满足`beta2 >= beta1`和`alpha <= gamma`的CW参数组合；
- 1个独立的`Default EDCA + 原生BEB`回退；
- 合计`16×54+1=865`个唯一、可解码动作。

代码与证据：

- 动作解码：`rl/tgax_environment.py`的`ReducedActionCodec.decode()`；索引16明确映射P0并关闭自定义增长。
- 动态应用：`ns3_ext/tgax-control.cc`的`ApplyControlAction()`；仅动作改变时写入BSS0的AP和40个STA。
- 运行时审计：`results/raw/tgax_full40_multiac_v6_action_smoke_20261026/reduced_action_audit.json`，验证了原生回退、自定义P8、再回退Default以及邻居不变。
- 结构审计：`results/summary/tgax_fixed_feasibility_preflight_20261028/action_space_audit.json`。

必须准确表述为：**全部865个动作已做结构可达性核验；没有逐个执行865次ns-3仿真。** 运行时真实性由既有三步审计和本轮17个代表动作覆盖。

### 2.2 固定动作不会每秒重置CW

正式实验日志显示，time=1 s首次写入测量动作后，2–20 s的`action_changed`均为0。seed 44/45的全部候选都通过`constant_action_not_rewritten=true`，因此CW状态可以跨STEP自然演化，而不是每秒重置。

### 2.3 邻居BSS隔离

每次正式运行都检查`edca_step_runtime.csv`：BSS1–6的全部设备、全部AC始终保持VO `2/3/7/2080 µs`、VI `2/7/15/4096 µs`、BE `3/15/1023/0`、BK `7/15/1023/0`及原生BEB。两个seed均为`all_neighbors_default=true`。

### 2.4 Reward-v3与早期Lagrangian实现

当前Reward-v3为固定加权和：

\[
R=0.4T_{VO}+0.4T_{VI}+0.1S_{all}+0.1J.
\]

它解决了旧v6直接奖励低重传导致的目标错配，但实测仍会接受BE/BK严重退化。它不是吞吐主效用的CMDP实现。

早期单BSS代码的实际实现位于外部历史工程：

- `.../wifi_rl/env/reward_cost.py`：`RewardCost.compute()`、`_throughput_tanh_scene_reward()`、`_scene_costs()`、`update_lambdas()`；
- `.../wifi_rl/configs/reward.json`；
- `.../wifi_rl/train/run_dueling_o1.py`。

其基础效用是按场景权重聚合的`tanh(2×(goodput/offered−0.5))`；Cost包括VO/VI p95超标、公平性低于0.85、碰撞率高于0.25。乘子初值0.1、上限20、步长0.01，按episode执行：

\[
\lambda_i\leftarrow\mathrm{clip}[\lambda_i+0.01(\bar C_i-b_i),0,20].
\]

旧实现的关键缺陷是Replay只保存已组合的标量Lagrangian reward；乘子变化后不会重算历史样本。因此它不能原样迁移。Jain 0.85和“碰撞率0.25”也不适合直接搬到当前TGax环境：当前需求归一化Jain约0.15–0.25，而且TGax后端只能可靠给出重传，不能精确归因碰撞。

## 3. 固定动作实验设计

### 3.1 快速筛选

- 预先声明17个代表动作，覆盖Default、P0纯CW变化、P1/P3/P5/P6/P8、已知保守的P9/P10/P15以及Profile+CW组合；
- seed 41/42，每个动作5秒，合计34次运行；
- 综合时延公式在正式seed前冻结：

\[
D=0.5\left(\frac{p95_{VO}}{20\,ms}+\frac{p95_{VI}}{50\,ms}\right).
\]

- 按应用有效吞吐、由成功MPDU原始样本聚合得到的VO/VI p95、Jain以及BE/BK饥饿门槛，预先选择5个正式候选。

筛选配置和选择记录见：

- `configs/tgax_fixed_feasibility_candidates.json`；
- `results/summary/tgax_fixed_feasibility_screen_20261028/screen_selection.json`。

### 3.2 20秒正式配对实验

正式候选：Default、P3+BEB、P3+CW、P0+CW-strong、P0+CW-late24。seed 44/45均采用相同拓扑、相同外生需求、相同预热和排空；共10次运行。所有运行通过：280/280关联、逐STEP配对需求、邻居Default、固定动作不重复写入。

时延不是上下行p95的平均值，而是把该AC上下行成功MPDU原始时延样本合并后重新计算p95。无成功样本则为未定义。

## 4. 正式结果

### 4.1 两seed均值

| 动作 | 应用有效吞吐 Mbps | 相对Default | VO p95 ms | VI p95 ms | Jain | VO/VI及时率 | 三目标Pareto |
|---|---:|---:|---:|---:|---:|---:|---|
| Default | 0.3350 | — | 21.75 | 28.53 | 0.1964 | 74.40% / 98.09% | 否 |
| P3+BEB | 0.3457 | +3.19% | 25.38 | 30.63 | 0.2011 | 72.22% / 98.28% | 是 |
| P3+CW | 0.3650 | +8.97% | 20.75 | 31.16 | 0.1965 | 76.44% / 97.73% | 否 |
| P0+CW-strong | 0.3707 | +10.66% | 21.16 | 28.15 | 0.1928 | 74.80% / 98.59% | 否 |
| **P0+CW-late24** | **0.6371** | **+90.16%** | **19.13** | **24.63** | **0.2010** | **76.66% / 98.89%** | **是；严格支配Default** |

这里的“严格支配”只针对本报告预先固定的三个均值目标：总应用吞吐更高、综合VO/VI时延更低、Jain更高，不代表所有AC、所有seed、所有辅助指标都更好。

### 4.2 配对seed稳定性

| seed | 动作 | 吞吐 Mbps | VO p95 ms | VI p95 ms | Jain |
|---:|---|---:|---:|---:|---:|
| 44 | Default | 0.4133 | 14.77 | 26.01 | 0.2392 |
| 44 | P0+CW-late24 | 1.0165 | 13.70 | 21.54 | 0.2460 |
| 45 | Default | 0.2567 | 28.73 | 31.05 | 0.1536 |
| 45 | P0+CW-late24 | 0.2576 | 24.55 | 27.72 | 0.1560 |

两个seed的时延和Jain方向一致，但吞吐收益高度依赖seed 44。仅两颗正式seed不能给出置信区间，也不能宣称固定动作已被证明普适优于Default。

### 4.3 各AC吞吐、失败与重传

| 动作 | VO | VI | BE | BK | VO最终失败 | VI最终失败 | BE最终失败 | BK最终失败 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Default | 0.1028 | 0.1922 | 0.0400 | 0 | 19.46% | 1.68% | 69.25% | 92.61% |
| P0+CW-late24 | 0.0970 | 0.4576 | 0.0824 | 0.00007 | 19.47% | 0.95% | 58.18% | 90.99% |

单位为应用有效载荷Mbps。P0+CW-late24的总吞吐提升主要来自VI和BE，VO吞吐略降；BK虽从零变为非零，但仍近似饥饿，不能称为BK问题已解决。其重传率由Default的VO/VI/BE/BK `79.03%/67.33%/44.25%/12.50%`变为`78.14%/65.32%/35.09%/29.17%`。重传仅用于机制解释，BK样本很少，比例不稳定。

最终失败率采用STEP汇总中完整risk population加权，不使用“测量窗内失败/测量窗内新到达”这种可能因预热积压跨窗而大于1的错误口径。

### 4.4 机制归因

P0+CW-late24的EDCA参数与Default完全相同，区别只有CW序列：

| AC | Default原生BEB | P0+CW-late24 |
|---|---|---|
| VO | 3, 7, 7… | 3, 6, 7… |
| VI | 7, 15, 15… | 7, 12, 15… |
| BE/BK | 15,31,63,127,255,511,1023… | 15,25,41,100,242,583,1023… |

因此它在早期失败阶段比BEB更温和地增长CW，在后期再加速到CWmax。真实日志同时显示VI/BE成功交付增加、VI/BE最终失败下降；这支持“更合适的失败后退避轨迹改善了当前竞争状态”的解释，但不能把重传直接叫作碰撞，也不能仅凭相关性宣称唯一因果机制。

![吞吐—时延—公平性Pareto分布](../results/figures/tgax_fixed_feasibility_formal_20261028/throughput_delay_fairness_pareto.png)

![代表动作核心指标](../results/figures/tgax_fixed_feasibility_formal_20261028/representative_core_metrics.png)

![各AC应用有效吞吐](../results/figures/tgax_fixed_feasibility_formal_20261028/application_throughput_by_ac.png)

![CW增长轨迹](../results/figures/tgax_fixed_feasibility_formal_20261028/representative_cw_trajectories.png)

## 5. 4.7%应用需求满足度的含义

### 5.1 Default的实际链路

两seed总体平均：

| 阶段 | 字节/20秒/seed | 相对独立生成需求 |
|---|---:|---:|
| 独立应用生成需求 | 16.225 MB | 100% |
| socket接受提交 | 1.561 MB | 9.63% |
| 应用最终收到有效载荷 | 0.838 MB | 5.22% |

MAC入队和MAC ACK字节不能直接放在同一百分比链条里相除，因为它们包含IP/传输层封装、真实TCP反向ACK以及重传相关MPDU，而应用账本只统计有效载荷。原始表仍保留这两列供机制核查：`demand_mac_delivery_pipeline.csv`。

### 5.2 各AC原因

- **VO**：独立生成需求全部能提交socket，但应用收到的比例在seed 44/45分别为48.18%/22.64%；主要损失发生在MAC竞争、最终失败和测量窗截止。
- **VI**：每个seed生成约14.6–15.7 MB，但socket只接受约0.48–0.49 MB，即约3.1%–3.3%。这是总体低满足度的最大来源。大量视频需求滞留于TCP发送侧/拥塞窗口之前，低MAC服务时延只描述已进入MAC的少量包，不能代表视频业务需求得到满足。
- **BE**：同一测量窗的接收字节有时会高于该窗新生成字节，因为预热及历史TCP积压跨窗口交付；因此单AC的“received/generated”不是严格同一批字节的队列转化率。seed 45的BE提交约363 KB但应用接收为0，显示短窗口和TCP阶段造成强随机性。
- **BK**：Default seed 44/45的应用接收均为0；MAC最终失败约92.6%，属于严重饥饿。

### 5.3 统计窗口影响

FTP/HTTP包含文件/页面过程和阅读间隔。20秒窗既可能只看到连接或积压阶段，也可能碰到历史积压集中交付，因此跨seed差异很大。4.7%/5.2%是“独立外生需求在有限窗口内被应用层收到的比例”，不是物理链路容量利用率，也不是MAC成功率。

当前场景仍适合做**高压力机制可行性**和安全约束研究，但不应成为唯一的平稳训练分布。若未来要证明动态策略价值，应预先定义多个业务阶段/负载状态或更长评估窗，同时保持最终测试seed独立；不能通过修改生成器来事后制造优势。

## 6. 下一版CMDP/Lagrangian Reward建议（仅设计）

建议恢复“吞吐是效用，时延、公平性和饥饿是约束”的结构：

\[
r_t=U_t-\lambda_D C_{D,t}-\lambda_J C_{J,t}-\lambda_L C_{L,t}.
\]

### 6.1 吞吐效用

以应用层实际收到的有效载荷为准，采用按AC归一化的凹效用，避免VI绝对字节量淹没其他AC：

\[
U_t=\frac{1}{\sum w_a}\sum_{a\in\{VO,VI,BE,BK\}}w_a
\log\left(1+\frac{G_{a,t}}{D^{ref}_{a,t}+\epsilon}\right).
\]

`G`为本窗口应用有效载荷；`D_ref`必须来自独立需求账本和开发seed上预先冻结的尺度，不能用“实际进入MAC量”，也不能直接把4.7%当基准。没有本窗新需求但有历史债务的AC仍算活跃。

### 6.2 三个Cost

1. **时延/超期Cost `C_D`**：对VO、VI分别使用原始成功MPDU聚合p95的相对超标，加上期限已到但最终失败/仍未完成的miss率。缺少成功样本但存在到期需求时必须产生Cost，不能记0。
2. **公平Cost `C_J`**：`max(0,(J_min-J)/J_min)`；J基于实际有需求STA的需求满足比例。`J_min`应由Default和固定动作的开发seed可实现区间预先标定。当前0.15–0.25表明旧0.85不可行。
3. **长期无服务Cost `C_L`**：对每个有需求AC维护跨窗债务，计算低于最低服务份额或连续无应用交付的正缺口，再对四AC聚合。它直接约束BE/BK饥饿，不用重传率替代业务服务。

重传率继续作为诊断指标，不进入首版CMDP Cost；TGax后端也不把重传伪称碰撞。

### 6.3 阈值与乘子

- 用新的开发seed上的Default分布与固定动作Pareto范围冻结`D_target/J_min/L_budget`，之后再锁定独立测试seed；seed 41/42/44/45已用于探索，不能作为未来最终留出测试。
- 每个episode更新一次乘子，避免1秒窗口的TCP随机波动：

\[
\lambda_i\leftarrow\mathrm{clip}
\left(\lambda_i+\eta_i(\overline C_i-b_i),0,\lambda_{max}\right).
\]

当平均Cost低于budget时该式会让乘子回落；应使用较慢时间尺度、有限上限和可记录的EMA，避免单episode异常把训练永久推向保守动作。
- Replay必须保存`(state, action, U, C_D, C_J, C_L, next_state, done)`，采样时用**当前**乘子重组reward；不得只保存旧乘子下的标量reward。
- 训练日志分别画`U`、三个Cost、三个lambda和最终组合reward，安全门槛继续独立检查吞吐、VO/VI p95及Jain，不能只按组合reward选checkpoint。

本报告没有为这些阈值填入未经验证的数值，也没有实现或训练Reward-v4。

## 7. 是否进入动态DDQN训练

目前可以确认“动作空间有优化潜力”，但还不能确认“动态选择有必要”：

- 本轮的最优改善来自P0上的CW增长，而非Profile切换；
- P3+BEB与P0+CW-late24同时位于三目标Pareto前沿，说明存在权衡，但没有展示随状态发生最优动作交叉；
- seed 44/45的吞吐收益幅度差异很大，提示TCP阶段和外生状态会影响动作收益，值得继续验证，却不是动态策略优越的直接证据；
- 当前16个Profile没有专门让BE/BK温和获得更多竞争机会的配置，P0已是低优先级相对最积极的基准之一，动作空间对BK饥饿的可调范围有限。

因此建议的下一步不是立即重启DDQN，而是先用全新开发seed或预先定义的不同负载阶段复核少数固定候选，检查最佳动作是否发生交叉。若始终由P0+CW-late24获胜，应优先使用固定规则或审查动作空间；只有出现可由观测识别的动作收益切换后，再实现上述CMDP并恢复TCN-DDQN训练。

## 8. 交付索引

- 正式逐seed指标：`results/summary/tgax_fixed_feasibility_formal_20261028/formal_per_seed.csv`
- 均值、改善百分比与Pareto：`results/summary/tgax_fixed_feasibility_formal_20261028/formal_mean_and_percentages.csv`
- 需求—socket—MAC—应用链路：`results/summary/tgax_fixed_feasibility_formal_20261028/demand_mac_delivery_pipeline.csv`
- 正式实验状态与审计：`results/raw/tgax_fixed_feasibility_formal_20s_seed44_45_20261028/status.json`
- 分析脚本：`scripts/analyze_tgax_fixed_feasibility.py`
- 图：`results/figures/tgax_fixed_feasibility_formal_20261028/`

