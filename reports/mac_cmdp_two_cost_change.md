# MAC-CMDP双Cost重构说明

日期：2026-11-01

## 1. 新CMDP目标

当前正式入口`configs/mac_cmdp_experiment.yaml`只生成两个Cost：

\[
\max_\pi J_U(\pi),\qquad J_Q(\pi)\le B_Q,\quad J_L(\pi)\le B_L
\]

TD训练使用当前乘子重新组合Replay中的原始分量：

\[
R_t=U_t-\lambda_QQ_t-\lambda_LL_t
\]

预算不在TD Reward中重复相减，只用于episode结束后的乘子更新。

### 吞吐效用

原需求满足度和凹效用保持不变：

\[
U_t=\sum_a w_a\frac{\ln(1+9S_{a,t})}{\ln 10}
\]

现有实现只对有当前需求或历史债务的AC启用相应权重，并在有效权重内归一化。权重仍为VO/VI/BE/BK=`0.4/0.4/0.1/0.1`。

### 高优先级QoS Cost

每个成功ACK的VO/VI MPDU精确计算：

\[
q_p=\frac{D_p}{D_p+\tau_a}
\]

最终丢弃MPDU的`q_p=1`。`D_p`来自MAC入队至最终ACK的逐MPDU `Y`事件；最终失败数来自逐方向/AC的`M`事件增量。VO和VI参考尺度由YAML中的20/50 ms指定。

每个1秒STEP分别得到`qos_vo_ul/qos_vo_dl/qos_vi_ul/qos_vi_dl`，对有最终完成样本的有效组等权平均：

\[
Q_t=\operatorname{mean}(Q_{VO,UL},Q_{VO,DL},Q_{VI,UL},Q_{VI,DL})
\]

在当前高并发配置中四组均有完成样本时，该式严格等于四组除以4。

### 低优先级Starvation Cost

\[
L_t=1-\min_{a\in A_t}(S_{a,t}),\qquad A_t\subseteq\{BE,BK\}
\]

`A_t`只包含有新生成需求或历史未交付债务的AC。若BE/BK都无需求，Starvation Cost标为无效，不作为零成本样本参与episode Cost均值；若只有一个AC有效，则使用该AC满足度。

## 2. 与旧四Cost版本的区别

正式入口不再生成`delay/loss/fairness/starvation`四Cost，也不再读取`delay_excess_cap`、`minimum_jain`、`minimum_satisfaction`或四维乘子。历史`tgax_cmdp_v1/v2/directional_v3`解析和旧checkpoint兼容代码仍保留，但当前入口的两Cost配置若混入旧字段会明确报错。

Jain、p95、20/50 ms及时率、最终丢包率和重传率继续作为评估指标，不额外进入Reward。

## 3. 跨窗口与特殊情况

- ACK或最终丢弃在哪个STEP发生，就在该STEP且仅一次计入QoS Cost。
- pending MPDU不提前算成功、零成本或最终失败；后续ACK/最终丢弃时再反馈。
- 最终失败包括入队失败、队列寿命过期、达到重试上限、QoS旧包和其他MAC最终失败。
- 有pending但当前无最终结果的方向组标为无效，并记录`pending_end`，不会伪装成零Cost。
- 某组没有业务且没有完成样本时从有效组平均中排除，不获得“服务良好”的零Cost。
- 现有需求债务跨STEP结转，所以FTP/HTTP阅读间隔内若仍有历史债务，BE/BK依旧保持active。

局限：训练episode结束时仍未完成的MPDU随独立ns-3 episode终止，无法在下一episode继续追踪，因此不会产生最终QoS完成成本。它仍会在episode内通过需求债务降低Utility，但这不是完整的跨episode MPDU结转。正式报告必须保留该限制，不能宣称所有截断末尾pending都已转化为QoS Cost。

## 4. Lagrange与Replay

Cost、预算和乘子顺序严格为：

```text
[qos, starvation]
[lambda_qos, lambda_starvation]
[budget_qos, budget_starvation]
```

episode内对有效STEP Cost求均值，EMA及投影更新保持原实现：

\[
\lambda_i\leftarrow\operatorname{clip}(\lambda_i+\eta(EMA(C_i)-B_i),0,\lambda_{max})
\]

Replay继续保存`utility`及原始二维`costs`。采样训练时使用当前乘子重组Reward，Transition中历史标量Reward不会覆盖这一过程。

## 5. 当前YAML（正式训练前需人工确认）

```yaml
qos_delay_reference_ms: {VO: 20.0, VI: 50.0}
cost_budgets: {qos: 0.40, starvation: 0.90}
multiplier_initial: [0.1, 0.1]
multiplier_learning_rate: 0.01
multiplier_ema_alpha: 0.2
multiplier_maximum: 3.0
```

以上预算和乘子只是用户指定的临时可运行值，代码不会通过Default或评估结果自动修改。

当前入口配置为seed 42训练100步、seed 66执行20秒final-checkpoint配对评估，属于快速Reward门禁，不是多seed正式收敛实验。

## 6. 修改文件

- `configs/mac_cmdp_experiment.yaml`：正式两Cost唯一入口。
- `project_config.py`：两Cost生成配置验证及历史版本兼容。
- `rl/cmdp_reward.py`：效用、逐MPDU QoS函数、双Cost和通用Lagrange向量。
- `rl/tgax_environment.py`：使用现有Y/M事件构造四方向精确完成事件成本。
- `scripts/run_mac_cmdp_experiment.py`：YAML迁移检查及两Cost配置物化。
- `scripts/run_tgax_ddqn_extended.py`：二维Cost训练、有效掩码及日志名称。
- `scripts/smoke_mac_cmdp_two_cost.py`：两STEP真实后端烟雾测试。
- `tests/test_cmdp_reward.py`、`tests/test_mac_cmdp_experiment_entry.py`、`tests/test_rl.py`：公式、空样本、pending、乘子、Replay和反向传播测试。

没有修改`ns3_ext/`、业务生成器、TCN、聚合器、Branching DDQN或动作空间。

## 7. 测试

- 专项测试：33 passed。
- 全项目：98 passed，7 skipped。
- 真实ns-3两STEP烟雾测试：通过；状态保持`40×4×10×6`历史张量与10维AP本地状态，Cost顺序为`[qos, starvation]`，四个方向组均产生有限结果。
- 烟雾结果：`results/raw/mac_cmdp_two_cost_smoke_r2_20261101/smoke_summary.json`。

未启动正式训练，也未执行180秒评估。

## 8. Linux服务器同步

无需重新编译ns-3，因为C++与补丁均未修改。服务器需要同步上述Python、YAML和测试文件；报告可选同步。不要同步`results/`、checkpoint、CSV或PNG。

项目已有Git远程仓库。本轮未自动commit、push或pull。建议commit message：

```text
Refactor MAC CMDP to exact QoS and starvation costs
```

在本地人工提交并推送后，等待或停止服务器当前旧实验，再在服务器执行：

```bash
cd /data/4t/lwk/compress_DRL_EDCA
git status
git pull --ff-only origin main
source .venv/bin/activate
python -m pytest tests/test_cmdp_reward.py tests/test_mac_cmdp_experiment_entry.py tests/test_rl.py -q
```

旧服务器训练仍使用启动时已经物化的旧配置；不建议在它运行期间混合部署并启动新训练。
