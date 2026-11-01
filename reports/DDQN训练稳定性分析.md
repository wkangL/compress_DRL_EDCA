# DDQN训练稳定性分析

更新时间：2026-10-20  
结论适用范围：阶段 C 的实现核查与短预算诊断。三组诊断只有一个训练种子，不构成收敛或正式性能证据。

## 1. 为什么进行本阶段

旧版完整训练使用 `target_update_steps=500` 的硬同步。11800 次更新中，TD loss 的平滑包络与 Q 值总体上升，并在目标网络同步处出现阶梯。阶段 C 的目标是区分实现错误、时间截断错误和目标更新机制的影响，并为后续受控训练选一个数值上更稳的候选配置。

历史运行只读保留在 `results/raw/full_absolute_resume98_20261020`。新诊断全部使用统计 schema 3、10维 AP 可观测状态和新的 checkpoint schema，不能与旧模型续接，也不能把短诊断 reward 当作历史正式性能的直接对比。

## 2. 实现核查与修正

### 2.1 完整且独立的目标网络

当前 target 是整个 `ControlNetwork` 的独立深拷贝，包含：

- Target TCN encoder；
- Target STA aggregator；
- Target branching dueling Q network。

target 处于 `eval()`，全部参数 `requires_grad=False`，且不加入优化器。Double DQN 下一动作由 online 完整网络选择，该动作的值由 target 完整网络计算。单元测试逐项验证：初始化相等但对象和参数存储独立；online 更新不会改变 target；硬同步覆盖全部参数；软更新按给定 tau 更新。

### 2.2 TD目标和合法动作

五个分支分别取已执行动作的 Q 值，再对同一个标量 TD target 计算 Smooth L1 loss。下一动作由 online 网络的分支贪心动作给出，其中 beta1/beta2 仍经过同一个合法组合掩码；target 网络只负责估值。当前 bootstrap 为：

`reward + discount * (1 - terminal) * target_Q(next_state, online_argmax)`。

### 2.3 time-limit修正

旧实现把每个固定时长 episode 的最后一步写成 `done=True`，使时间上限处 bootstrap 归零。网络仿真并未在该时刻进入吸收终态，因此这是实现错误。

修正后：

- `done` 只表示真实 terminal；
- 固定时长结束记录为 `truncated=True`；
- truncated 样本继续 bootstrap；
- checkpoint 保存并恢复 done/truncated、replay、所有 RNG 和训练日志。

三组短诊断的最终 replay 均有 120 条 transition、0 条 terminal、24 条 truncated，符合 24 个定长 episode 的结构。精确恢复测试验证 checkpoint 前后的样本标记和随机序列一致。

### 2.4 新增训练诊断量

每次更新现在保存：online Q 的均值/标准差/极值、target Q 的均值/标准差/极值、TD target、TD error、绝对 TD error、reward、bootstrap/terminal/truncated 比例、裁剪前梯度范数、latent 方差、目标网络同步标记和同步前后参数 RMS 距离。每阶段还保存 replay 的场景比例、五个动作分支直方图、探索比例和独立 greedy 评估。

## 3. 旧版 Q-loss 阶梯核查

旧版日志没有 target Q、TD target 和 TD error，无法对每次阶梯做完整因果分解；但同步时刻与 loss 跃升具有一致对应关系。以同步前20次和同步后20次更新的均值计算：

|同步更新|TD loss 前→后|mean Q 前→后|
|---:|---:|---:|
|500|0.101 → 0.369|0.209 → 0.489|
|1000|0.165 → 0.420|0.651 → 0.961|
|2000|0.283 → 0.472|1.603 → 2.007|
|3500|0.388 → 0.581|3.064 → 3.646|
|6000|0.546 → 0.644|5.953 → 6.396|
|9000|0.612 → 0.764|8.243 → 9.325|
|11000|0.697 → 0.789|10.232 → 11.125|

其余绝大多数500步边界也呈同方向变化。最后200次更新的平均 TD loss 为0.7686，平均 Q 为11.391，Q 极值绝对值达到27.75，平均梯度范数为2.654。直接机制是硬同步将已经变化的 online 表征和 Q 估计整体复制到 bootstrap 路径，造成 TD target 离散移动；随后优化器重新追赶新目标。Q 值长期上移使每次目标刷新带来的目标差也逐渐变大，所以阶梯包络持续抬升。

这能解释“为什么在同步点跳升”，但旧日志不足以证明 Q 值增长究竟有多少来自奖励尺度、采样分布或过估计。已确认的 time-limit 错误会在 episode 边界错误切断 bootstrap，也会改变值函数尺度；它应修复，但不能单独解释同步点上的规律性跃升。

## 4. H500、S005、S001受控短诊断

保持状态、网络、reward-v1、业务、replay、环境交互、初始化 seed 11 和评估 seed 301 相同。每组6阶段、24个episode、120个transition、500次更新；每阶段在四场景上做无探索 greedy 评估。第一阶段只有20条样本，未达到32条 warmup，因此三组阶段1完全一致。

|配置|target更新次数|前50 loss|后50 loss|后50平均绝对TD误差|最大绝对Q|最大绝对target Q|最大梯度范数|最大同步前距离|阶段6平均评估reward|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|H500|1|0.6755|0.0331|0.1575|3.565|0.194|3.540|0.01665|-0.0209|
|S005|500|0.6810|0.0772|0.2521|5.324|2.620|7.197|0.00767|-0.0147|
|S001|500|0.6765|0.0381|0.1701|3.510|0.446|3.907|0.01235|-0.0361|

原始表位于：

- `results/summary/diagnostic_target_updates_20261020/stability_summary.csv`
- `results/summary/diagnostic_target_updates_20261020/independent_evaluation_by_stage.csv`
- `results/summary/diagnostic_target_updates_20261020/replay_coverage.csv`

图位于：

- `results/figures/diagnostic_target_updates_20261020/target_update_stability.pdf`
- `results/figures/diagnostic_target_updates_20261020/independent_evaluation.pdf`

三组 replay 覆盖完全相同：S1/S2/S3/S4 分别25/40/30/25条，探索样本占55.83%；五个动作分支分别覆盖16/3/6/11/3种离散值。因而短诊断差异不是场景数量或探索预算不同造成的。

## 5. 如何解释结果

H500 的末段 loss 最低，但500次更新内 target 几乎一直是初始化网络，只在最后一次更新后同步；该短程低 loss 主要说明 online 网络容易拟合一个固定旧目标，不能说明长训练稳定。旧版11800步结果已经显示重复硬同步会产生阶梯。

S005 的 online/target 距离最小，但 target 跟随过快；本次出现更大的 Q 幅度、target Q、梯度峰值和末段 TD error。因此 tau=0.005 不作为下一阶段默认候选。

S001 消除了离散同步，且 Q 幅度、梯度峰值和末段 TD error接近H500，明显低于S005。它仍保持可见的 target 滞后，适合作为下一轮中等预算、多训练seed验证的候选。因此默认配置改为 `target_update_mode=soft, target_tau=0.001`；历史和显式H500配置不改。

独立评估没有呈单调改善：四场景平均 reward 在阶段1约0.139，阶段中间一度改善，但三组阶段6均转为轻微负值；goodput和交付率也有回落。单 seed、500次更新不足以区分策略波动和系统性退化。阶段 C 的结论是“实现错误已修复，S001数值表现更适合作为候选”，不是“已经收敛”。

## 6. 准入结论和后续要求

- 完整 target、Double DQN估值路径、合法动作约束和time-limit语义均有自动测试。
- TD/Q/target/replay/独立评估监控已经补齐，后续能直接定位同步和数值漂移。
- 阶段 C 通过，可以进入业务场景重新标定和固定动作实验。
- 新正式训练仍被禁止，直到阶段 D/E 明确典型负载与动作空间的可优化范围。
- S001必须在至少3个独立训练seed的中等预算下再次验证；若 Q 值或独立评估继续系统性恶化，再单独对学习率、梯度裁剪或 replay warmup做一次一因素消融。

## 7. 验证状态

阶段 C 完成前的完整测试为45项，包含真实ns-3后端集成。报告冻结时将重新运行全部 `pytest`，并保存源码快照、配置、原始CSV和图表哈希；诊断失败目录及首次缺少SciPy导致的中断均保留，未覆盖或删除。
## 2026-10-21：Reward-v2 三 seed 中等预算复核

固定 Reward-v2、Compressed Async + Age、soft target `tau=0.001` 和现有网络/业务，使用训练 seed 11/22/33。每个 seed 运行 20 stages、800 次环境交互、3400 次更新；阶段 1/4/8/12/16/20 用独立 seed 301/302 做 greedy 评估，最终测试 seed 101–105 未使用。

三个 seed 均完整结束，无 NaN/Inf，最大梯度范数为 4.83/4.77/5.28，未触及 10 的裁剪阈值。末 200 次 TD loss 均值为 0.00155/0.00236/0.00307；末 200 次各 latent 维平均方差的最小值为 0.00233/0.00393/0.00422，高于 `1e-5` 的坍缩阈值。

独立评估从阶段 8 起较稳定。S4 VI p95 在阶段 8/12/16/20 的跨 seed 均值为 6.400/6.487/6.396/6.277 ms；阶段 20 的 seed 间标准差约 0.104 ms。S4 BE 服务满足度对应为 0.932/0.931/0.931/0.928。阶段 20 相对 Default：S2/S4/S5 的 VI p95 改善约 48.2%/8.2%/44.8%，碰撞率改善约 70.7%/62.6%/68.7%；S1/S3 的低负载 VI p95 分别恶化约 11.0%/20.4%。

Q 均值尚未稳定：三 seed 的末 200 次 mean Q 约 8.85/8.81/8.88，target Q 约 6.26/6.22/6.30，并随更新持续上升。这里的 episode 时间上限按 truncated 正确 bootstrap，环境没有真实 terminal，且每步 Reward-v2 多为正；折扣 0.98 下共同价值基线会逐步接近持续任务的长期回报。三个 seed 的 Q 轨迹一致、TD error 有限，当前没有数值发散证据，但同样没有价值函数已收敛的证据。因此本阶段结论限定为“策略层独立评估在该预算后半段趋稳”，不写“DDQN 已收敛”。

数据：`results/summary/reward_v2_medium_3seed_20261021`。图：`results/figures/reward_v2_medium_3seed_20261021/medium_training_diagnostics.png`、`medium_independent_qos.png` 以及逐 AC 时延 CCDF。

