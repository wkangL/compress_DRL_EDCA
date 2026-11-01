# AP 可观测状态审计

更新时间：2026-10-20（Asia/Shanghai）

## 结论

旧模型输入的所谓 global state 实际是两个 16 维窗口拼成 32 维。训练和正式评估的决策周期均为 1 秒，因此 `recent=[t-1,t]` 与 `period=[start,t]` 在正常 step 中完全相同，32 维中有 16 维是逐值重复。单个 16 维窗口还混入全方向成功吞吐、按成功字节计算的全网 Jain、按成功包判断的 active STA，以及反馈 age/coverage；其中 age/coverage 已在 STA 聚合器中再次输入。

本轮将 global state 固定为 10 维 AP 本地状态：1 维 AP PHY busy、4 维 AP 实际接收的分 AC UL goodput、4 维 AP 自身分 AC 下行队列、1 维关联 STA 数。只保留一个决策窗口。反馈 age、valid 和 coverage 仍由 `StaAggregator` 处理，不再复制到 global state。

状态维度和语义已改变，配置 `schema_version` 从 1 升到 2。旧 checkpoint 的第一层尺寸及训练语义均不兼容，只能用于历史结果复现，不能续训或直接加载到新网络。

## 旧 16 维单窗口逐项审计

| 旧索引 | 特征 | 来源 | AP可观测性 | 问题与处理 |
|---:|---|---|---|---|
| 0 | busy fraction | AP PHY `State/B` | 是 | 保留 |
| 1 | AP RX failure ratio | AP节点 `F` 与成功 UL 包 | 是 | 失败可能对应 DATA/RTS，分母却是成功业务包，语义不一致；移除，精确失败统计留给评估/reward |
| 2 | total goodput / C_ref | 全部 `D`，UL+DL | AP可推导 | 与索引5–8之和重复；移除 |
| 3 | Jain fairness | 每STA全部成功字节 | AP可推导但属于全局评估目标 | 混合 AC/方向后掩盖业务满足率，并把当期结果直接塞入状态；移除 |
| 4 | active STA fraction | 当期至少一个成功包的 STA | 可推导但语义错误 | 表示“成功过的 STA”，不是关联 STA；改为关联表数量 |
| 5–8 | 四 AC goodput / C_ref | 全部 `D`，UL+DL | AP可推导 | 改为 AP 实际收到的 UL goodput，方向明确 |
| 9–12 | AP 四 AC queue bytes | AP节点 `S` | 是 | 保留，使用 `log1p(queue)/log1p(1e6)` |
| 13–15 | report age mean/p95/coverage | AP report cache | 是 | 已在 permutation-invariant STA 聚合器中按 age/valid 输入；从 global state 移除 |

旧代码又把上述窗口分别计算为 `recent` 和 `period` 后拼接。在 1 秒决策周期下两者边界相同；这不是多尺度信息，只是重复特征。reset 时两个窗口也同为空值。

## 新 10 维状态

| 新索引 | 特征 | 计算与归一化 | 来源 |
|---:|---|---|---|
| 0 | busy fraction | 合并决策窗口内 AP PHY 非 IDLE 区间，除以窗口长度 | AP PHY |
| 1–4 | UL VO/VI/BE/BK goodput | AP 成功接收的业务 payload bit/s，除以 `C_ref` | `AckedMpdu` 对应 `D`，发送节点为 STA |
| 5–8 | AP VO/VI/BE/BK DL queue | `log1p(queue_bytes)/log1p(1e6)` | AP MAC 四个 EDCAF queue |
| 9 | associated STA count | `min(N_assoc/64,1)` | AP 关联表；当前静态拓扑为10 |

新特征不读取 STA 私有队列、STA 端 retry/drop、未上报的历史或评估器汇总。当前 ns-3 场景没有动态关联事件，关联数由已建立拓扑给出；以后加入漫游或离线 STA 时应改为真实关联表回调，维度不变。

## 网络尺寸影响

- STA TCN：7,304 参数，保持不变。
- STA 聚合器：1,441 参数，输出 268 维，保持不变；每 AC 仍含 attention、max、age mean/p95、coverage。
- 当前动作编码：20 维，保持不变。
- Q backbone 输入：由 `32 + 268 + 20 = 320` 降为 `10 + 268 + 20 = 298`。
- Q 模块：由 120,232 参数降为 114,600；完整网络由 128,977 降为 123,345。

## 验证

- 新增 AP state 单测，验证重叠 busy 区间只计一次、DL 成功不混入 UL goodput、10维顺序与配置一致、非法时间/归一化尺度被拒绝。
- 模型与 TD 更新单测已适配 `GLOBAL_DIM=10` 并通过。
- `ControlEnvironment` 不再维护额外 `past` 窗口，observation 只计算本次决策区间。
- 阶段 B 只改变状态定义，尚未启动训练。后续 checkpoint 必须使用新的配置、源码指纹和 checkpoint schema；历史 checkpoint 保留但不兼容。

