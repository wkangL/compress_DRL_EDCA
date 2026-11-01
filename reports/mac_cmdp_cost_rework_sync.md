# MAC-CMDP Cost小范围重构与服务器同步说明

## 1. 结论与修改边界

本轮已将当前单BSS、40 STA、MAC直注随机业务训练入口切换到 `tgax_cmdp_v2`。新的四维Cost及乘子顺序固定为：

```text
[delay, loss, fairness, starvation]
```

本轮没有修改业务数量、拓扑、MAC业务生成器、ns-3 C++、TCN、STA聚合器、299维状态、865动作编解码、DDQN网络、学习率、Replay容量或Target软更新；没有启动训练，也没有运行20秒/180秒评估。

## 2. 审计结果

### 2.1 原实现

- 主效用已经来自MAC层独立生成需求与跨窗口债务，而不是TCP/socket提交量。
- 原四项Cost依次为 `1-VO及时率`、`1-VI及时率`、`1-Jain`、旧低优先级Cost。
- 旧低优先级Cost把BE/BK合入一个最大项，并可由连续零服务规则抬高到2.0；旧校准配置的第四个预算曾为1.0。这就是旧“low priority cost budget”的实际含义，它不是BE/BK最低满足度本身。
- Replay的 `Transition` 已经独立保存 `utility` 和四维 `costs`，无需修改Replay结构。
- `td_update()` 已经能在采样时按当前乘子重新组合Reward；本轮修正了v2的组合公式，并保留v1兼容开关。
- 乘子原本已经按episode聚合Cost、EMA平滑、投影到 `[0, maximum]`，本轮保留该机制。

### 2.2 可复用的MAC账本

现有Python每1秒能得到VO/VI上下行的成功服务时延样本，以及 `pending_start`、`arrived`、`acked`、`failed`、`pending_end`。账本恒等式为：

```text
pending_start + arrived = acked + failed + pending_end
```

现有C++用节点标签与Packet UID去重；每个MPDU只在首次MAC入队时登记，成功ACK或最终Drop时从pending表移除。因此最终失败不会跨窗口重复记账。`failed`包含ns-3报告的入队失败、队列寿命到期、达到重试上限、QoS old packet和其他最终Drop；测量截止仍pending的MPDU不算成功，也不算最终失败。

MAC服务时延起点是 `WifiMac::MacTx` 首次观察到该MPDU、即进入MAC服务账本的时刻，终点是 `AckedMpdu`。它包含MAC队列等待、AIFS/退避和重传等待，但不包含业务生成器在提交给MAC之前的时间。

## 3. 新Utility与Cost公式

### 3.1 凹吞吐效用

四AC满足度为 `S_a = received_payload / (carried_debt + newly_generated_payload)`，并截断到 `[0,1]`。对当前有需求或历史债务的AC，保留既有凹效用：

```text
f(S) = log(1 + 9S) / log(10)
U = sum(w_a * active_a * f(S_a)) / sum(w_a * active_a)
w = [0.40, 0.40, 0.10, 0.10]
```

历史未交付需求跨窗口结转，当前窗口没有新到达但仍有债务的AC仍是active。

### 3.2 delay_cost

对VO UL、VO DL、VI UL、VI DL分别计算成功ACK MPDU的超额服务时延：

```text
e_i = min(max(delay_i / reference - 1, 0), cap) / cap
reference_VO = 20 ms
reference_VI = 50 ms
cap = 3.0
```

每组先对MPDU取均值，再对有活动的方向组等权平均，避免大样本VI UL掩盖小样本VO DL。若某方向存在风险MPDU但本窗口一个成功样本都没有，该方向Cost记为1，而不是虚假的零时延。当前后端没有逐个pending MPDU的实时等待年龄，因而不扩展C++；pending压力由该无成功保护、最终丢包Cost和跨窗口需求不足共同约束。正式p95仍只用于评估，不进入每秒Cost。

### 3.3 loss_cost

每个VO/VI上下行组采用：

```text
group_loss = failed / (acked + failed)
loss_cost = 有最终结果的四组group_loss等权平均
```

pending既不是失败也不是成功。某组没有任何最终结果时不伪造零损失，而是标为无效，不参与该episode该Cost的有效性统计；如果它有风险但没有成功，`delay_cost=1`保护仍生效。

### 3.4 fairness_cost

沿用有实际需求STA的需求归一化Jain指标：

```text
fairness_cost = max(minimum_jain - actual_jain, 0) / minimum_jain
```

没有有效需求STA时该窗口公平性Cost标为无效，不用虚假的0推动乘子。

### 3.5 starvation_cost

BE与BK分别计算，不能相互抵消：

```text
d_BE = max(min_BE - S_BE, 0) / min_BE
d_BK = max(min_BK - S_BK, 0) / min_BK
starvation_cost = 活跃BE/BK缺口的等权平均
```

存在历史债务即视为活跃；完全无新需求且无债务的AC标为无效并从均值排除。原始BE/BK满足度和两个缺口都写入step指标。

### 3.6 Lagrangian与Replay

Replay保存原始 `utility` 与 `[delay, loss, fairness, starvation]`。采样时用当前乘子重组：

```text
R = U - lambda_delay*C_delay - lambda_loss*C_loss
      - lambda_fairness*C_fairness - lambda_starvation*C_starvation
```

预算不在TD Reward里重复扣减。每个episode只对有效Cost计算均值，然后保持原有更新：

```text
EMA <- (1-alpha)*EMA + alpha*C_episode
lambda <- clip(lambda + eta*(EMA-budget), 0, lambda_max)
```

旧 `tgax_cmdp_v1` 检查点仍可走历史Cost函数和历史“只罚预算超额”组合方式；新入口不会再把 `vo_timely/vi_timely` 当训练Cost。

## 4. YAML唯一参数入口

正式训练前需人工确认 [configs/mac_cmdp_experiment.yaml](../configs/mac_cmdp_experiment.yaml) 的以下值；程序不会自动标定或改写：

| 参数 | 当前占位值 | 含义 |
|---|---:|---|
| `utility_weights` | VO/VI/BE/BK = .4/.4/.1/.1 | 凹需求满足效用权重 |
| `delay_reference_ms` | VO=20, VI=50 | 时延超额参考值 |
| `delay_excess_cap` | 3.0 | 超额截断上限 |
| `minimum_jain` | 0.50 | Jain最低要求 |
| `minimum_satisfaction` | BE=.02, BK=.005 | 两个低优先级AC独立最低满足度 |
| `cost_budgets` | .30/.15/.0/.25 | delay/loss/fairness/starvation预算 |
| `multiplier_initial` | 四项均.1 | 初始乘子，顺序同Cost |
| `multiplier_learning_rate` | .01 | 乘子步长 |
| `multiplier_ema_alpha` | .2 | episode Cost EMA系数 |
| `multiplier_maximum` | 3.0 | 乘子投影上限 |

这些数值只是完整可运行的占位配置，不代表已通过实验确定。生成的 `train.json` 会整体替换父配置的Reward段，因此不会暗中继承旧Cost字段。

## 5. 精确修改文件清单

| 相对路径 | 修改内容 | 服务器必须同步 | ns-3重编译 | 类型 |
|---|---|---:|---:|---|
| `configs/mac_cmdp_experiment.yaml` | 新四Cost阈值、预算、乘子参数的唯一入口 | 是 | 否 | 配置 |
| `project_config.py` | 校验v2 Cost顺序、范围及乘子参数 | 是 | 否 | Python |
| `scripts/run_mac_cmdp_experiment.py` | 从YAML生成纯v2 Reward段，禁止旧字段继承/自动预算标定 | 是 | 否 | Python |
| `rl/cmdp_reward.py` | 新Utility/四Cost、v1兼容、标准CMDP Reward、乘子校验 | 是 | 否 | Python |
| `rl/tgax_environment.py` | v1/v2分发、v2指标与有效性掩码接入 | 是 | 否 | Python |
| `rl/trainer.py` | Replay采样时按当前λ重组v2 Reward；v1兼容开关 | 是 | 否 | Python |
| `scripts/run_tgax_ddqn_extended.py` | 识别v2、四Cost有效掩码、日志λ名称和状态顺序 | 是 | 否 | Python |
| `tests/test_cmdp_reward.py` | 四Cost、空成功样本、方向权重、动态λ等测试 | 建议 | 否 | 仅测试 |
| `tests/test_mac_cmdp_experiment_entry.py` | YAML和生成配置无旧字段测试 | 建议 | 否 | 仅测试 |
| `reports/mac_cmdp_cost_rework_sync.md` | 本说明与同步/回滚步骤 | 建议 | 否 | 文档 |

未修改 `rl/replay_buffer.py`，因为其现有结构已正确保存原始Utility与Cost；未修改任何 `ns3_ext/*.cc`。

## 6. 验证结果

本地只运行Python测试和语法检查，没有启动正式训练或评估：

```text
python -m pytest tests -q
88 passed, 7 skipped
```

覆盖内容包括：四Cost有限且在0~1、无成功VO/VI方向不会显示良好时延、上下行等权、pending不算最终丢包、BE不能掩盖BK饥饿、乘子升降、Replay动态重组、配置无旧字段，以及TCN/聚合器/DDQN反向传播。跳过项是原项目按环境条件标记的测试，不是本轮失败。

## 7. 最小服务器上传与运行前检查

项目根目录没有 `.git`，因此不能可靠提供Git提交同步。本次不需要上传整个项目、历史结果、CSV/PNG、Checkpoint、模型权重或ns-3构建产物，也不需要重新编译ns-3。推荐把ZIP上传到服务器项目根目录后执行：

```bash
cd /data/4t/lwk/compress_DRL_EDCA
mkdir -p ../mac_cmdp_cost_rework_backup
tar -czf ../mac_cmdp_cost_rework_backup/before_cost_v2_$(date +%Y%m%d_%H%M%S).tar.gz \
  configs/mac_cmdp_experiment.yaml project_config.py \
  scripts/run_mac_cmdp_experiment.py scripts/run_tgax_ddqn_extended.py \
  rl/cmdp_reward.py rl/tgax_environment.py rl/trainer.py \
  tests/test_cmdp_reward.py tests/test_mac_cmdp_experiment_entry.py

unzip -o /data/4t/lwk/mac_cmdp_cost_rework.zip -d /data/4t/lwk/compress_DRL_EDCA
source .venv/bin/activate

python -c "from rl.cmdp_reward import COST_ORDER; print(COST_ORDER)"
python -c "from pathlib import Path; from scripts.run_mac_cmdp_experiment import read_yaml,validate; p=Path('configs/mac_cmdp_experiment.yaml'); validate(read_yaml(p)); print('YAML OK')"
python -m pytest -q tests/test_cmdp_reward.py tests/test_mac_cmdp_experiment_entry.py tests/test_models.py
python -m py_compile project_config.py rl/cmdp_reward.py rl/tgax_environment.py rl/trainer.py scripts/run_mac_cmdp_experiment.py scripts/run_tgax_ddqn_extended.py

grep -n '^  root:' configs/mac_cmdp_experiment.yaml
```

期望第一条输出 `('delay', 'loss', 'fairness', 'starvation')`。最后一条显示正式结果根目录；可在训练前自行修改YAML中的 `output.root`。

若不用ZIP，可从本机项目根目录以最小目录结构同步：

```bash
rsync -avR \
  configs/mac_cmdp_experiment.yaml project_config.py \
  scripts/run_mac_cmdp_experiment.py scripts/run_tgax_ddqn_extended.py \
  rl/cmdp_reward.py rl/tgax_environment.py rl/trainer.py \
  tests/test_cmdp_reward.py tests/test_mac_cmdp_experiment_entry.py \
  reports/mac_cmdp_cost_rework_sync.md \
  lwk@SERVER:/data/4t/lwk/compress_DRL_EDCA/
```

## 8. 回滚

若服务器验证失败，不要删除新结果目录。回到项目根目录，将第7节生成的备份包覆盖恢复：

```bash
cd /data/4t/lwk/compress_DRL_EDCA
tar -xzf ../mac_cmdp_cost_rework_backup/before_cost_v2_YYYYMMDD_HHMMSS.tar.gz -C .
```

因为没有数据库迁移、C++ ABI变化或ns-3重编译，回滚只涉及上述Python/YAML文件。历史v1配置和实验结果均未删除。
