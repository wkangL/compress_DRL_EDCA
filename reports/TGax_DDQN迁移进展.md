# TGax TCN-DDQN迁移进展

更新时间：2026-10-23

## A. Default等价验证

- 新增持久TGax STEP后端；一个ns-3进程在2 s关联预热后按1 s窗口连续推进。
- P0已更新为HE Default参数：VO `2/3/7/2.080 ms`、VI `2/7/15/4.096 ms`、BE `3/15/1023/0`、BK `7/15/1023/0`。
- `P0 + custom_growth=false`不调用动态EDCA写接口，保持ns-3原生Default EDCA和BEB。
- 与既有同seed、同业务、同拓扑5 s一次性Baseline逐项比较：BSS 0成功包、p95、到达、ACK、最终失败、重传和尝试次数最大相对差异为0。A阶段通过。

数据：`results/raw/tgax_ddqn_v1_default_step_5s_r3_20261023`；比较：`results/summary/tgax_ddqn_v1_20261023/default_equivalence.json`。

## B. 动作生效验证

- 第1秒Default；第2秒应用P5及`alpha=1,beta1=1.0,beta2=3.0,gamma=5`。
- BSS 0的AP与40 STA共164个AC实例全部匹配P5；BSS 1–6共246台设备、984个AC实例全部保持Default。
- 记录1317次动作后CW变化；VO观测`3/11/15`，VI `7/23/31`，BE `63/191/575`，均属于公式可达集合。

数据：`results/raw/tgax_ddqn_v1_action_validation_2s_20261023`；验证：`results/summary/tgax_ddqn_v1_20261023/action_validation.json`。B阶段通过，可以进入单seed小规模训练。

## C. 首次训练诊断与修正

第一次10步、50更新训练数值稳定，但同seed评估发现learned动作从首窗口开始影响TCP连接完成时刻，使VI应用生成需求只有Default约42%。这组对比不能用于判断算法效果，已保留在`results/raw/tgax_ddqn_v1_small_seed41_20261023`作为诊断结果。

修正方式：在每个训练/评估episode的2 s关联预热后，增加1 s统一Default EDCA业务预热；TCP连接、初始业务状态和反馈历史均在相同条件下建立。预热期间未服务的应用生成需求作为债务带入第一个决策窗口。该修正不改变业务模型、拓扑或无线参数。
# 2026-10-23 最终状态

第一版闭环迁移已完成，最终有效目录为 `results/raw/tgax_ddqn_v1_small_seed41_r4_20261023`。修复了 TCP 需求受建连门控、交互排空时刻、共同 SSID 导致 20 个 STA 漫游以及无样本比例显示为 0 等问题。固定关联后 280/280 STA 匹配；一次性与 STEP Default 五类 MAC 计数零差异；动作验证中 BSS0 164 行生效、邻居 1,968 行保持 Default；学习评估中 5,904 条邻居 EDCA 记录无偏离。

训练完成 10 环境步、50 更新，数值稳定但策略 Reward 比 Default 低 2.28%，不能说明收敛或算法有效。正式结论和图表见 `reports/TGax TCN-DDQN第一版训练报告.md`。

# 2026-10-25 固定动作与CW冗余阶段

- 完成 A=P0+原生BEB、B=P0+自定义CW、C=P8+原生BEB、D=P8+自定义CW 的 seed 41/42 配对20 s实验；四组逐STEP外生需求相同，280/280关联正确，邻居BSS无参数偏离。
- 持久后端增加相同动作缓存，固定动作只在发生变化时写入，避免每个1 s STEP重置EDCA/CW状态。
- P0上自定义CW在两种子均提高Reward和Jain公平性，但VO及时率均下降；P8上增加自定义CW在两种子均使Reward、VO/VI及时率及四个VO/VI p95恶化。因此确认机制有效，但尚无稳定综合性能收益。
- 按实际公式和重试上限枚举7344个合法动作，仅有3446个Profile内不同四AC联合轨迹，数学完全重复比例53.08%。提出865个保守候选或242个严格轨迹代表候选，本阶段未修改动作空间。
- 发现正式测量窗内BK成功样本近零：FTP主要在1 s业务预热内传输，随后进入长阅读间隔。应用TCP可达，但当前20 s窗不足以评价BK控制收益。

正式报告：`reports/TGax固定动作与CW冗余分析.md`。

# 2026-10-26 400步延长训练阶段

- 完成Default seed 41/42逐MPDU时延阈值离线分析。VI=50 ms在seed 41接近饱和（99.385%），seed 42为94.030%；30 ms区分更明显，但本轮训练仍严格使用VO=20 ms、VI=50 ms。
- 完成seed 41的20 episode × 20步训练，共400个环境transition、340次梯度更新；每个episode先冻结模型收集20步，再更新20次，整体更新/样本比0.85。
- 数值稳定：未出现NaN/Inf、Q值爆炸、梯度异常或TCN表示坍缩；Q-loss分段均值由0.2878降至0.0417，最大梯度范数3.391，低于裁剪阈值10。
- 完成100/200/300/400步在seed 43/44的独立20 s贪心验证和配对Default。按两seed平均验证Reward选择300步checkpoint；相对Default均值提高4.52%，但200步在seed 44曾退化66.54%，且seed 44公平性持续偏低。
- 完成未参与训练及checkpoint选择的seed 45留出测试。300步最佳checkpoint的Reward比Default低1.15%，Jain需求满足公平性低20.63%；UL VI p95虽降低46.52%，成功吞吐同时下降84.20%，不能视为真实改善。400步最终checkpoint Reward低1.74%。
- 策略已摆脱第一版持续P8，但后期标量分支近乎固定于`alpha=3、beta2=2.8、gamma=5`，Profile主要在P3/P0/P8之间变化；尚未证明动态控制优于合理固定动作。
- 最终结论：训练闭环与数值学习成立，但未证明策略收敛或稳定优于Default。本阶段按要求停止，不启动多seed正式训练。

正式报告：`reports/TGax TCN-DDQN延长训练实验报告.md`。原始结果：`results/raw/tgax_ddqn_extended_seed41_400step_20261025`。

# 2026-10-26 多业务高并发场景重建（进行中）

- 已实现并测试40 STA固定业务覆盖生成器：每个STA均有上行VO/VI，BE覆盖32个STA，BK覆盖16个STA；各业务继续调用原TGax随机生成器，未修改分布参数。
- 候选v1的FTP-only BK在5 s内没有成功样本；v2用持续Gaming形成BK压力后，20 s Default的BK最终失败率达到97.52%，不适合作为正式训练环境；v3改用HTTP/FTP覆盖后，5 s仍为40次BK入队、0次成功，表明高优先级持续负载本身使Default BK在500 ms队列寿命内无法接入。这些失败门控均保留，不作为算法结果。
- 继续门控表明，只替换持续Gaming仍不能解除AP内部VO/VI对BK的虚拟碰撞压制。场景最终冻结为`candidate_full40_multiac_v6`：1对双向VoIP，其余39个STA各有上行VoIP；1路下行Gaming、1路上行BV3；20路HTTP下行与12路FTP上行映射BE；16路HTTP下行与8路FTP上行映射BK。40/40 STA仍有高优先级上行，BE/BK分别覆盖32/16 STA，五类原TGax生成器均保留。5 s Default中四AC均有入队和成功，账本守恒；BK仅3/59成功，明确作为高压已知项而不再改场景。
- 865动作方案已接入：16个Profile × 54组CW参数，加1个独立`Default EDCA + 原生BEB`回退动作。发现并修复了从非默认动作切回Default时旧EDCA/CW状态未恢复的问题；三步实测确认P0原生BEB→P8自定义CW→P0原生BEB真实切换，BSS1–6始终保持Default。
- Reward的及时交付项已与独立应用需求满足率耦合，避免通过减少进入MAC或成功服务的样本获得虚假低时延收益；VO=20 ms、VI=50 ms阈值暂保持不变，待最终Default分布确认后即冻结。
- v6三步动作审计已通过：`Default原生BEB → P8自定义CW → Default原生BEB`均真实写入BSS0并恢复，BSS1–6的3,936条运行时记录全部保持Default；全项目测试为69 passed、7 skipped。20 s正式Default正在执行。
- v6的20 s Default已通过覆盖、四AC活动和MAC账本守恒检查。BSS0按AC合并结果：VO p95 25.95 ms、最终失败19.43%；VI p95 38.24 ms、最终失败1.89%；BE p95 493.86 ms、最终失败41.42%；BK仅3个成功样本、最终失败89.66%。BK时延统计不足，后续选择checkpoint时不得把无样本或少样本低p95解释成改善。正式训练配置由此冻结。

# 2026-10-27 Reward-v3阶段（训练前检查完成）

- v6三训练seed在200步验证后安全停止；最近完整checkpoint分别保留在260、260、280步，Replay、随机状态、训练日志及验证结果均未删除。旧自动续跑任务已删除，旧训练不再占用CPU。
- 旧Reward的目标错配得到复核：它奖励重传和最终失败下降，使P10等保守动作通过减少发送尝试提高Reward，却造成VO/VI数百毫秒等待及及时率、吞吐、公平性退化。
- 新增Reward-v3：`0.40*T_VO(20 ms)+0.40*T_VI(50 ms)+0.10*S_all+0.10*J`；不含重传项。无期限样本、无业务需求的窗口不赋满分。
- C++交互协议新增逐STA外生生成需求与应用有效载荷交付增量；Python按STA跨窗口结转未交付债务，并由有需求STA的交付满足比例计算在线Jain。短时实测中40个有需求STA均进入统计，逐STA与逐AC生成/交付账本逐字节一致。
- 用历史v6 seed44/45精确期限账本和应用账本离线重算。历史数据没有逐窗口逐STA账本，因此只对Jain使用最终episode汇总代理并明确标注。六组配对全部满足`Reward-v3(Default)>Reward-v3(旧DDQN@200)`；训练seed41对应seed44为0.758对0.262，seed45为0.668对0.115。
- Reward单元测试和全项目测试通过：72 passed、7 skipped。新配置与v6的网络、业务、TCN、聚合器、动作空间、训练超参数和soft Target配置逐项相同，仅Reward、单seed运行计划及100步验证间隔不同。

离线核查：`results/summary/tgax_reward_v3_offline_20261027`；后端冒烟：`results/raw/tgax_ddqn_reward_v3_smoke_20261027`。下一步为从随机Q网络启动唯一训练seed 41的400步实验。

# 2026-10-28 Reward-v3单seed训练完成

- 完成seed 41共400个环境步、340次梯度更新，以及100/200/300/400步在seed 44/45的独立配对验证和seed 46/47留出测试。Q-loss前/后20更新均值为0.07744/0.01016，Q绝对值最大1.256，梯度范数最大0.574，无NaN、Inf或发散。
- 100步checkpoint在所有验证与留出窗口均选择独立Default回退动作，性能与Default逐项相同，是唯一通过及时率、吞吐和公平性门槛的checkpoint，但不构成智能优化。
- 200/300/400步策略不再复现旧v6的VO/VI及时率崩溃，但均因Jain公平性及BE/BK吞吐退化未通过门槛。400步主要选择P9及`alpha=1,beta1=2.0,beta2=2.0,gamma=5`，使BE/BK接入显著保守。
- 留出seed 46/47均值中，400步相对Default的Reward几乎不变（+0.01%），VO及时率+1.26%、VI及时率-0.23%；但BE吞吐-94.03%、Jain-15.78%、应用有效载荷/生成需求-10.45%，VO聚合p95从24.49 ms升至34.25 ms。因此不能判定有效。
- 结论：Reward-v3修复了“通过降低重传换取高Reward并让VO/VI时延崩溃”的主要错配，但`S_all/J`的实际尺度和权重仍不足以阻止低优先级饥饿。单seed实验按计划停止，不扩展多seed。

正式报告：`reports/TGax Reward-v3与TCN-DDQN训练实验.md`；原始结果：`results/raw/tgax_ddqn_reward_v3_seed41_400step_20261027`；汇总与PNG：`results/summary/tgax_ddqn_reward_v3_seed41_400step_20261028`、`results/figures/tgax_ddqn_reward_v3_seed41_400step_20261028`。

# 2026-10-28 固定EDCA/CW可行性与约束奖励设计阶段

- 暂停DDQN训练，完成865动作结构可达性、原生Default回退、固定动作不重复写入及邻居BSS隔离审计。865个动作由16 Profile×54个合法CW参数组合和1个原生Default回退构成；代表动作运行审计全部通过。
- 完成17动作×seed 41/42×5秒快速筛选，并在正式seed 44/45上完成5个候选×20秒配对实验；所有运行满足需求配对、280/280关联正确、邻居Default及固定动作连续性检查。
- `P0+CW-late24`在两seed均值上相对Default使应用有效吞吐提高90.16%、VO/VI原始样本聚合p95降低12.07%/13.67%、Jain提高2.35%，是唯一三目标严格支配Default的候选。吞吐收益主要来自seed 44，seed 45近乎持平，因此仅能证明存在优化空间，不能宣称普适固定最优。
- 改善保持P0 EDCA不变，只修改CW增长轨迹，说明本次主要收益来自CW机制；P3原生BEB仅小幅增吞吐且恶化VO/VI时延。BK仍严重饥饿。
- 新增应用socket提交字节审计。Default总体只有9.63%的独立生成需求在20秒测量窗内被socket接受，应用最终收到5.22%；VI大量需求滞留于MAC之前。明确4.7%/5.2%是有限窗的外生需求满足率，不是MAC交付率或链路容量。
- 审计早期单BSS Lagrangian源码：旧Replay保存的是已组合标量reward，lambda更新后不会重算历史样本；旧Jain 0.85和碰撞0.25阈值不能迁移。提出吞吐凹效用加时延、公平性、长期无服务三个Cost的CMDP设计，并要求Replay保存原始效用/Cost、采样时按当前lambda组合。本阶段未实现Reward-v4、未训练DDQN。

正式报告：`reports/TGax固定EDCA-CW参数可行性与约束奖励设计.md`；正式原始结果：`results/raw/tgax_fixed_feasibility_formal_20s_seed44_45_20261028`；汇总与PNG：`results/summary/tgax_fixed_feasibility_formal_20261028`、`results/figures/tgax_fixed_feasibility_formal_20261028`。

# 2026-10-28 单BSS MAC Full Buffer Default基线

- 新建独立`1 AP + 40 STA`单BSS环境，复用原BSS0坐标、TGax传播和802.11ax HE SU参数；旧7 BSS TCP/UDP环境与历史结果未覆盖。
- 新增MAC直注Full Buffer源：不安装IPv4栈，不经过TCP/UDP/IP/Socket；数据经`WifiMac::Enqueue`进入正常QoS队列，保留EDCA、原生BEB、TXOP、A-MPDU、Block ACK和重传。
- 20 STA持续VO、另20 STA持续VI，另有20个BE源和12个BK源，共72个`STA×AC`队列；使用32/64包水位和100 us轮询，避免把500包队列上限长期灌满。
- 冒烟及seed 41/42/43三组20秒正式运行全部通过40/40关联、四AC活动、无IP流量、零重复注入、账本守恒、队列饱和和VO/VI聚合门禁。
- Default三seed总成功有效载荷吞吐为40.70±3.06 Mbps、信道忙89.19%；VO/VI p95为178.96±18.28/221.20±18.98 ms，20/50 ms及时率为49.92%/73.68%。BE/BK吞吐仅0.582/0.0367 Mbps，最终失败约79.1%/84.2%，存在严重低优先级饥饿。
- 结论：该环境适合下一阶段固定EDCA/CW机制搜索，但必须联合约束VO/VI及时率、BE/BK最低服务、失败/过期和公平性；它是MAC饱和机制环境，不替代真实应用端到端场景。本阶段未训练DDQN、未修改Reward、未搜索动作。

正式报告：`reports/单BSS-MAC-FullBuffer-Default基线实验.md`；原始结果：`results/raw/mac_fullbuffer40_default_20s_3seed_20261028`；汇总与PNG：`results/summary/mac_fullbuffer40_default_20s_3seed_20261028`、`results/figures/mac_fullbuffer40_default_20s_3seed_20261028`。

# 2026-10-29 单BSS MAC随机到达Default复测

- 在相同1 AP、40 STA、HE SU网络上新增独立`mac-arrival`模式；保留VoIP通话/静默、Gaming随机包长/间隔、BV3 Weibull帧与Gamma扰动、FTP文件/阅读、HTTP页面/对象/请求模型，但不安装TCP/UDP/IP/Socket。
- 复用`candidate_full40_multiac_v6`的BSS0业务分配：40路UL VoIP、1路DL VoIP、1路DL Gaming、1路UL BV3、20 HTTP+12 FTP映射BE、16 HTTP+8 FTP映射BK；36个HTTP反向请求使运行时共135个生成器。
- MB级视频帧、文件和页面保存在应用侧需求队列并分段成最大1460 B MSDU；随机化500 us admission轮询、64包MAC高水位避免瞬间灌满500包队列。生成、提交和待提交字节独立守恒。
- 5秒全Trace与seed 41/42/43三组20秒正式运行全部通过关联、五类模型生成、无IP流、四AC上下行活动、需求守恒和MAC账本门禁。
- 三seed总MAC有效载荷吞吐17.80±2.00 Mbps、信道忙31.81%。旧版“平均4.44”实际是瞬时MAC队列非空源节点且包含AP，不是有业务STA数；该口径已在后续版本修复为分别输出严格STA数和全部源节点数。VO/VI上下行p95约0.72–1.16 ms，20/50 ms及时率接近100%；低时延源于随机到达下高优先级竞争较轻，不是TCP节流。
- FTP平均仍有4.98 MB停留在应用侧待提交；DL-BK p95约321.9 ms、最终失败8.18%，低优先级积压依然存在。该随机到达场景与Full Buffer饱和上界并行保留，不能互相替代。

正式报告：`reports/单BSS-MAC随机到达-Default复测.md`；原始结果：`results/raw/mac_arrival40_default_20s_3seed_20261029`；汇总与PNG：`results/summary/mac_arrival40_default_20s_3seed_20261029`、`results/figures/mac_arrival40_default_20s_3seed_20261029`。

# 2026-10-29 单BSS MAC高Gaming随机到达Default复测

- 在上一版随机到达环境上只增加Gaming并发：每个40 STA配置2路Gaming UL与2路Gaming DL，共160路；Gaming随机包长和到达间隔、其他VoIP/BV3/FTP/HTTP模型、拓扑和Default EDCA均保持不变。本配置明确标为TGax模型驱动的自定义高压扩展。
- 修复竞争强度统计口径：`active_stas`只统计MAC队列非空STA，另行输出包含AP的`active_nodes`。三seed严格活跃STA平均31.84、峰值40，证明已形成广泛的高优先级随机到达竞争。
- seed 41短时冒烟及seed 41/42/43三组20 s正式运行均通过40/40关联、无IP栈、五类模型活动、四AC上下行活动、需求守恒、无重复准入和MAC账本守恒检查。
- 相对原随机到达场景，信道忙比例由31.81%升至65.55%；VO-UL p95由0.83升至25.23 ms，VI-UL由1.16升至54.43 ms；VO/VI及时率降至70.76%/90.13%。因此此前低时延确实主要来自高优先级随机并发不足。
- 总成功MAC有效载荷吞吐由17.80降至10.94 Mbps；三seed平均108.65 MB生成需求中仅41.82 MB在窗口内进入MAC，66.83 MB仍在应用侧待准入。BE/BK p95约432–460 ms，失败与过期严重，表明160路Gaming已是过载压力点，后续评价不能只看成功样本时延。
- 本阶段未训练DDQN、未修改Reward、未搜索EDCA/CW。该配置建议作为高压测试点，与原随机到达轻载点和Full Buffer上界并行使用，而非直接替代全部正式场景。

正式报告：`reports/单BSS-MAC高Gaming随机到达-Default复测.md`；原始结果：`results/raw/mac_arrival40_gaming160_default_20s_3seed_20261029`；汇总与PNG：`results/summary/mac_arrival40_gaming160_default_20s_3seed_20261029`、`results/figures/mac_arrival40_gaming160_default_20s_3seed_20261029`。

# 2026-10-29 高Gaming场景CMDP训练迁移（进行中）

- 用户确认采用VO/VI优先效用：四AC需求满足度先经`log1p(9S)/log(10)`凹变换，再按VO/VI/BE/BK=`0.4/0.4/0.1/0.1`加权。VO/VI生成到最终MAC ACK及时率、需求归一化Jain及BE/BK长期服务作为独立Cost；重传率和忙比例仅作诊断。
- 新增`rl/cmdp_reward.py`与原始效用/Cost Replay字段。历史样本在TD采样时使用当前Lagrange乘子重新组合，避免旧实现把过时乘子固化进Replay。乘子按20步episode更新，初值0.1、步长0.01、EMA 0.2、上限1。
- 持续STEP后端新增`mac-arrival`模式：不安装IPv4栈，保留原随机VoIP/Gaming/BV3/FTP/HTTP生成器和真实WifiMac；逐AC、逐STA记录外生生成、MAC准入与最终ACK字节，并以MSDU生成时刻而非MAC入队时刻启动VO 20 ms/VI 50 ms期限。
- 两轮后端冒烟通过8个方向/AC、4个AC需求、40个STA需求和期限因果账本。首次发现并修复UL/DL共用期限计数导致的重复输出。完整环境2步测试通过，生成到ACK口径下seed 41短窗VO/VI及时率为74.43%/31.35%，证明不能复用旧MAC入队到ACK约71%/90%的预算。
- 当前正在并行运行seed 41/42/43各20秒Default正式标定；训练配置中的Cost预算保持`null`，脚本会在预算未冻结时拒绝启动训练。标定完成后按Default 95%冻结VO/VI及时率与Jain预算，按Default 80%冻结BE/BK最低需求满足度，再进入单seed 400步训练。

代码与配置：`configs/mac_arrival40_gaming160_cmdp.yaml`、`configs/mac_arrival40_gaming160_cmdp_training.yaml`、`scripts/calibrate_mac_arrival40_cmdp.py`。标定目录：`results/raw/mac_arrival40_gaming160_cmdp_default_calibration_20261029`。

## 标定完成并启动训练

- seed 41/42/43各20秒Default标定全部PASS。生成到ACK口径的聚合及时率为VO 70.3355%、VI 36.0079%；平均需求满足度为VO 74.6042%、VI 32.2882%、BE 19.0716%、BK 8.9406%；需求归一化Jain为0.675479。
- 按预先确认规则冻结Cost预算：`[C_VO=0.3318128, C_VI=0.6579253, C_J=0.3582952, C_L=1.0]`；BE/BK最低满足度冻结为15.2573%/7.1525%。配置保存为`configs/mac_arrival40_gaming160_cmdp_calibrated.yaml`。
- Default等价门禁通过：持续STEP相对一次性正式基线的总成功MAC吞吐偏差在三个seed分别为0.207%、0.027%、0.252%，均小于1%。全项目测试为78 passed、7 skipped。
- 动作生效门禁通过：原生Default与P8+自定义CW分别真实写入全部41节点×4 AC；P8 EDCA参数与Default不同，记录到68,420次运行时CW事件。
- 已启动训练seed 44的400步任务，目录为`results/raw/mac_arrival40_gaming160_cmdp_seed44_400step_20261029`。每20步冻结一次episode并更新Lagrange乘子，每100步做独立验证；Replay保存原始utility/Cost并用当前lambda重组TD reward。

## 2026-10-30 单BSS高Gaming CMDP训练完成

- 完成seed 44的400环境步、340次梯度更新，seed 45/46的100/200/300/400步独立验证，以及seed 47留出测试。400步checkpoint在预先固定的及时率、总交付、BE/BK与公平性门禁下合格，并因验证CMDP utility最高被选中。
- 数值稳定：Q-loss前/后20更新均值0.04056/0.00192，Online Q最大0.8195，梯度范数最大0.5112，无NaN、Inf、Q值或梯度异常。最终lambda为`[0.0697, 0.0932, 0.1701, 0.1493]`。
- 留出seed 47上，学习策略相对Default使VO/VI及时率从67.36%/36.11%提高到90.82%/39.52%，Jain从0.1380提高到0.2151，UL-VI p95从70.55 ms降到58.90 ms；但UL-VO成功样本p95由25.77 ms升到29.87 ms。
- 学习策略显著增加BE/BK服务，但FTP/HTTP状态机的后续需求受服务完成时刻影响，导致BE生成需求没有严格配对；因此低优先级改善百分比只能作为观测结果，不能作为严格因果结论。VO/VI总生成量一致，比较更可信。
- 结论：CMDP设计纠正了此前只降重传、牺牲时延和低优先级业务的目标错配，并在单训练seed/单留出seed上显示真实优化潜力；尚不能宣称多seed稳定优于Default。下一步应先固定外生需求轨迹，再扩展多训练seed。

正式报告：`reports/单BSS高Gaming-CMDP-TCN-DDQN训练报告.md`；原始结果：`results/raw/mac_arrival40_gaming160_cmdp_seed44_400step_20261029`；汇总与PNG：`results/summary/mac_arrival40_gaming160_cmdp_seed44_400step_20261030`、`results/figures/mac_arrival40_gaming160_cmdp_seed44_400step_20261030`。

## 2026-10-30 移除无规范依据的64包准入高水位

- 确认64包高水位不是IEEE TGax参数，也不是ns-3要求；它源于早期Full Buffer环境与64 MPDU Block ACK窗口相容、避免500包队列瞬间灌满的工程启发式。高Gaming环境中AP VI队列长期贴近64包，且动作会通过队列周转改变MAC准入量，因此该机制不再用于后续正式实验。
- 保留旧配置与全部历史结果，并在旧训练报告顶部标记“带自定义64包高水位的历史环境”。新增`configs/mac_arrival40_gaming160_cmdp_no_watermark.yaml`，设置`direct_high_watermark=0`且清空旧Cost预算，禁止直接复用旧标定和checkpoint结论。
- C++后端新增无水位语义：每个外生生成MSDU立即提交给原生`WifiMacQueue`，只由ns-3的500包容量、500 ms寿命、EDCA、重传和丢弃机制处理；旧非零水位路径仅用于复现实验。
- 无水位1秒STEP冒烟通过：四AC生成字节与提交字节逐项相等，8个方向/AC账本全部守恒，40 STA需求、期限因果和动态事件输出正常。无水位事件量显著增加，1秒仿真墙钟约3分钟，说明后续Default正式标定需要重新评估CPU成本。

无水位冒烟：`results/raw/mac_arrival40_gaming160_no_watermark_smoke_v2_20261030`。本阶段未启动新训练；必须先在无水位环境重新完成Default基线、方向级及时率核查及同口径Cost预算标定。

## 2026-10-30 新VO/VI并发配置三seed延长训练（进行中）

- 新建独立业务输入，不覆盖上一版：VoIP UL 40、VoIP DL 20、Gaming DL 40、BV3 UL 10；未点名的Gaming UL保持80，BE/BK的HTTP/FTP数量保持不变。五类业务仍使用既有随机到达/包长模型，只改变并发实例数。
- 正式配置继续采用无自定义准入水位模式：外生生成的MSDU立即提交原生`WifiMacQueue`。seed 41的一秒冒烟已通过四AC生成=提交、8个方向/AC MAC账本守恒、40 STA需求账本及期限因果检查。
- 三个训练seed固定为42、66、88；训练预算从历史400步延长到每seed 600个1秒决策步（30个20秒episode），warmup、网络、动作空间、学习率、soft Target更新与CMDP结构保持不变。
- 按用户要求关闭100/200/300等中途策略评估和checkpoint择优。训练结束后仅使用600步final checkpoint，在seed 44/55/77上与配对Default比较。
- 当前先在独立seed 41/43/45上并行进行20秒Default STEP标定；预算按实际乘子更新所用的一秒尺度冻结，避免把累计20秒指标与逐步Cost混用。标定完成后流水线自动启动三seed训练。

输入与配置：`results/raw/tgax_mac_arrival40_voip40_20_gaming80_40_bv3ul10_inputs_20261030`、`configs/mac_arrival40_voip40_20_gaming80_40_bv3ul10_cmdp.yaml`。流水线状态：`results/raw/mac_arrival40_voip40_20_gaming80_40_bv3ul10_cmdp_600step_3seed_20261030/status.json`。

### 三seed训练与final-only评估完成

- seed 42/66/88均完成600环境步、540次梯度更新；未进行中途checkpoint性能评估或择优，正式比较严格使用各自600步final snapshot和seed 44/55/77配对Default。逐秒外生生成需求在所有配对中完全一致。
- 数值稳定：三个seed前20更新Q-loss约0.029，后20更新分别为0.00739/0.00528/0.00933；全程最大绝对Q值1.134、最大梯度范数0.734，无NaN、Inf、Q值或梯度爆炸。
- 九组final评估均值相对Default：VO 20 ms及时率40.62%→52.00%，VI 50 ms及时率21.07%→33.08%；VO/VI需求满足度89.52%/54.28%→95.50%/78.19%；Jain 0.4323→0.4752；Reward 0.3065→0.4738。三个训练seed各自的VO/VI及时率、Reward和Jain均高于Default均值。
- 方向级p95存在明确权衡：UL-VO 208.87→105.47 ms、UL-VI 485.74→464.21 ms、DL-VI 279.69→239.41 ms改善，但DL-VO 55.38→68.21 ms恶化。UL-VO/UL-VI最终失败率分别由11.93%/37.21%降至0.94%/16.01%。
- BE从近零服务改善；BK由Default零交付提升到约0.3%需求满足度，但样本仍很少且高度饥饿，BK相对提升百分比因Default为0而明确记为未定义。
- post-training首次因final-only模式未创建`validation/`目录而停止，但三个600步checkpoint和训练数据完整。修复后直接从final snapshot完成评估，没有重跑训练、没有改变模型。最终项目测试79 passed、7 skipped。

正式报告：`reports/单BSS新业务配置-CMDP-TCN-DDQN三seed训练报告.md`；原始结果：`results/raw/mac_arrival40_voip40_20_gaming80_40_bv3ul10_cmdp_600step_3seed_20261030`；汇总与PNG：`results/summary/mac_arrival40_voip40_20_gaming80_40_bv3ul10_cmdp_600step_3seed_20261030`、`results/figures/mac_arrival40_voip40_20_gaming80_40_bv3ul10_cmdp_600step_3seed_20261030`。

## 2026-11-01 两Cost Reward 100步门禁与独立评估

- seed42完成100个环境步、40次更新后按用户决定停止，不继续额外100步。Q-loss的连续10-update均值由0.03355降至0.00315，最终Q值0.287--0.521、梯度范数0.0164，无数值发散。
- 使用未参与训练的seed66完成20秒Default/final配对评估。Final相对Default：Reward +54.69%、Utility +31.35%、VO/VI及时率分别由42.91%/19.31%升至75.35%/28.64%，Jain由0.4778升至0.5330。
- UL-VO p95由204.56 ms降至48.10 ms；但DL-VI p95由196.44 ms恶化至345.45 ms，失败率由6.72%升至13.66%。QoS Cost 0.472>0.40预算，饥饿Cost约1.0>0.90预算，BE/BK满足度仅1.16%/0.42%，因此尚不是满足约束的最终策略。
- 独立评估全部选择Profile 7，仅gamma分支变化，状态自适应性有限。当前结论仅基于一个独立seed，应在解决DL-VI与低优先级不可行问题后再做多seed复核。

报告：`reports/MAC-CMDP两Cost-100步诊断与评估.md`；结果与PNG：`results/experiments/mac_cmdp_two_cost_fast_seed42_eval66_20261101/summary`、`results/experiments/mac_cmdp_two_cost_fast_seed42_eval66_20261101/figures`。

## 2026-11-02 三Cost Reward 100步训练与独立评估

- 新增最大方向QoS Cost，形成`qos_avg/qos_max/starvation`三个独立约束，预算0.40/0.65/0.90。首轮v5因未进入精确QoS账本门控而无效，已保留并标注`INVALID_RUN.md`；修复版r2从随机初始化重跑。
- 修复版真实ns-3门禁和训练日志确认`Q_max>=Q_avg>0`，三个lambda均按预算违反更新。100步完成40次梯度更新，Q-loss连续10-update均值由0.02038降至0.00282，Q值和梯度稳定。
- seed66独立配对中，Final相对Default：Utility +31.35%，Q_avg下降18.85%，Q_max下降15.25%，VO/VI及时率分别由42.91%/19.31%升至75.35%/28.64%，Jain由0.4778升至0.5330。
- 仍不满足可行性：Final的Q_avg 0.472>0.40、Q_max 0.700>0.65、Starvation约1.0>0.90；DL-VI p95由196.44 ms恶化至345.45 ms，失败率由6.72%升至13.66%。策略20步均选择Profile 7，仅gamma少量变化。

报告：`reports/MAC-CMDP三Cost-100步训练与评估.md`；结果：`results/experiments/mac_cmdp_three_cost_fast_r2_seed42_eval66_20261101`。

## 2026-11-02 三Cost Reward 300步训练与独立评估

- seed42从随机初始化完成10个30秒episode、300个环境步和240次梯度更新；最终checkpoint仅在未见seed66上进行20秒Default/Final配对评估。Q-loss最终约0.00202，Q值和梯度有限，无NaN/Inf或数值爆炸。
- Final相对Default：Reward +129.98%、Utility +42.01%、VO/VI及时率分别提高10.79%/76.61%，Jain提高25.53%；平均/最大方向QoS Cost分别下降9.87%/22.33%。
- 主要收益在上行：UL-VO p95下降30.53%、失败率由11.99%降至0.49%；UL-VI吞吐提高71.84%、失败率由39.72%降至13.11%。
- 仍有明确退化：DL-VO p95由46.95 ms升至102.21 ms，DL-VI p95由197.52 ms升至237.44 ms。BK仍零交付，饥饿Cost保持1.0。
- Final仅满足最大方向Cost预算（0.6435≤0.65）；平均Cost 0.5425>0.40、饥饿Cost 1.0>0.90，因此不是满足全部CMDP约束的可行策略。结论仅来自一个训练seed和一个评估seed，不能宣称多seed稳定优于Default。

报告：`reports/MAC-CMDP三Cost-300步训练与评估.md`；结果与PNG：`results/experiments/mac_cmdp_three_cost_300step_gaming80_40_seed42_eval66_20261102/summary`、`results/experiments/mac_cmdp_three_cost_300step_gaming80_40_seed42_eval66_20261102/figures`。

## 2026-11-02 五方向Cost 100步快速门禁

- 将VO DL、VO UL、VI DL、VI UL拆成四个独立精确QoS Cost，并保留BE/BK饥饿Cost；预算为0.40/0.61/0.49/0.79/0.90。seed42完成100步和40次更新，最后Q-loss 0.00093，Q值及梯度有限。
- seed66配对评估中，Final相对Default使VO/VI及时率提高78.38%/34.86%、Utility提高30.47%、Jain提高24.08%。VO DL、VO UL和VI UL Cost分别下降17.27%/48.27%/14.13%，均达到预算。
- VI DL仍失败：Cost 0.5123→0.5755，p95 197.52→330.90 ms，失败率13.88%→17.72%；饥饿Cost仍约1.0。Final的20步全部选择P7，Profile级自适应性不足。
- seed66未参与模型训练，但参与了预算设计，因此不是完全design-held-out；本轮只能证明独立方向Cost能暴露并保护部分方向，不能证明泛化。

报告：`reports/MAC-CMDP五Cost-100步训练与评估.md`；结果：`results/experiments/mac_cmdp_five_cost_100step_r2_seed42_eval66_20261102`。

## 2026-11-02：AP/STA独立EDCA Profile控制

- 新增`action.profile_application_mode`配置开关：`unified`保留原五分支动作并把同一Profile应用到AP和全部STA；`split_ap_sta`使用`(AP Profile, STA Profile, alpha, beta1, beta2, gamma)`六分支动作。
- 分离模式只增加STA Profile分支。四个CW增长参数仍共享，业务、STA/AP原始观测、TCN与聚合器、Reward和Cost公式均未改变；当前动作编码由21维增加到38维。
- 独立Default回退按角色处理：AP或STA任一侧可单独选择P0+原生BEB；只有两侧都回退时四个标量索引才规范化为0。
- C++持久后端新增向后兼容的`STEP2`协议；旧`STEP`仍保持统一Profile语义。运行时每个节点依据`role`选择AP或STA Profile，BSS范围仍限定为BSS0。
- 真实ns-3短时门禁使用AP=P3、STA=P8：日志核对1个AP、40个STA、4个AC共164条EDCA记录，AIFSN/CWmin/CWmax/TXOP全部与对应Profile表一致。
- 完整测试通过：109 passed、7 skipped。本阶段只实现并验证控制能力，未启动新训练。

