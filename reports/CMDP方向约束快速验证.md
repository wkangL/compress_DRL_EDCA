# CMDP方向约束快速验证

日期：2026-11-01

## 1. 结论

原四Cost版本虽然先分别计算VO/VI、UL/DL组指标，但最终把四组Delay和Loss分别压成一个平均值，因此允许表现好的方向补偿表现差的方向。新的方向版使用六个Cost：

\[
[C_{delay,UL},C_{delay,DL},C_{loss,UL},C_{loss,DL},C_{fair},C_{starve}]
\]

Delay预算分别为0.40，Loss预算分别为0.08，Fairness预算为0，Starvation预算为0.30。

## 2. Loss的准确含义

本文的Loss是MAC最终丢包率，不是重传率或碰撞率：

\[
C_{loss,d}=\frac{1}{|A_d|}\sum_{a\in A_d}
\frac{N_{failed,a,d}}{N_{ACKed,a,d}+N_{failed,a,d}}
\]

其中`failed`包括入队失败、队列寿命过期、达到重试上限、QoS旧包以及其他MAC最终失败。窗口末仍处于pending状态的MPDU不提前算丢包，也不进入当前Loss分母；其后出现最终结果时再记入对应窗口。

## 3. Delay定义

方向内VO、VI等权。每个成功MPDU先按VO 20 ms、VI 50 ms参考值计算归一化超额时延：

\[
c_i=\frac{\min(\max(d_i/D_a-1,0),3)}{3}
\]

若一个方向/AC存在风险MPDU但没有成功样本，该组Delay Cost记为1，防止零成功被误判为零时延。

## 4. 乘子尺度

为避免方向拆分后初始惩罚翻倍，原Delay乘子0.1均分为UL/DL各0.05，原Loss乘子0.15均分为UL/DL各0.075；Fairness 0.1和Starvation 0.2保持不变。之后六个乘子分别依据各自预算更新。

## 5. 合成反例门禁

| 场景 | 旧平均Cost | 新方向Cost | 旧预算结果 | 新预算结果 |
|---|---|---|---|---|
| 健康 | Delay=0，Loss=0 | 全部为0 | 通过 | 通过 |
| 仅VO UL时延恶化 | Delay=0.25 | Delay UL=0.50 | 0.25≤0.40，错误通过 | 0.50>0.40，正确违反 |
| 仅VI DL无成功包 | Delay=0.25 | Delay DL=0.50 | 0.25≤0.40，错误通过 | 0.50>0.40，正确违反 |
| 仅VO UL最终丢包20% | Loss=0.05 | Loss UL=0.10 | 0.05≤0.08，错误通过 | 0.10>0.08，正确违反 |

这证明方向版消除了最关键的UL/DL互相补偿漏洞。由于初始乘子按方向均分，合成场景的初始TD Reward尺度与旧版保持一致；差异来自方向版拉格朗日乘子随后会对真正违反的方向增加，而旧平均版会错误地认为预算已满足。

## 6. 验证范围与限制

专项及全项目测试结果为92 passed、7 skipped。旧训练CSV没有保存逐MPDU时延样本，不能用p95反推平均超额Delay Cost，因此本轮没有伪造真实轨迹的离线Delay结论。下一门禁应使用短时固定动作或100步单seed训练，直接保存六个Cost、六个乘子及UL/DL真实指标。

复现文件：

- `configs/mac_cmdp_experiment_directional_fastcheck.yaml`
- `scripts/validate_directional_cmdp_reward.py`
- `results/summary/cmdp_directional_reward_fastcheck_20261101/validation.json`
- `results/summary/cmdp_directional_reward_fastcheck_20261101/validation.csv`

本轮未启动新的DDQN训练。
