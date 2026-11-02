# MAC-CMDP：一个YAML、两个命令

## 你只需要编辑一个文件

固定入口配置是`configs/mac_cmdp_experiment.yaml`。

业务数量、STA分配、Cost约束、训练seed、评估seed、训练长度、学习率、Replay参数和输出目录都写在这个YAML中。不需要复制模板，不需要单独生成业务文件，也不需要单独运行Default Cost标定。

## 两个命令

训练、final checkpoint评估并自动绘图：

```powershell
.\.venv\Scripts\python.exe scripts/run_mac_cmdp_experiment.py train
```

已有结果只重新生成CSV和PNG：

```powershell
.\.venv\Scripts\python.exe scripts/run_mac_cmdp_experiment.py plot
```

临时指定输出目录：

```powershell
.\.venv\Scripts\python.exe scripts/run_mac_cmdp_experiment.py train --output results/experiments/my_run_02
```

可选地指定另一个YAML：

```powershell
.\.venv\Scripts\python.exe scripts/run_mac_cmdp_experiment.py train --config configs/my_experiment.yaml --output results/experiments/my_run
```

## `train`自动完成的工作

1. 检查YAML字段和数值范围；
2. 按`traffic`自动生成业务实例及STA分配；
3. 在输出目录内部生成解析后的底层JSON配置；
4. 对全部训练seed训练或从checkpoint恢复；
5. 只使用final checkpoint在评估seed上与Default配对评估；
6. 生成汇总CSV和PNG。

第2和第3步只是内部准备，不再要求用户单独运行命令。

## AP与STA的Profile控制方式

在同一个YAML里设置：

```yaml
action:
  profile_application_mode: unified
```

可选值：

- `unified`：保留旧行为，一个Profile同时写入AP和全部40个STA，动作仍为5分支；
- `split_ap_sta`：AP使用一个Profile分支，全部STA共同使用另一个Profile分支，动作变为6分支。

分离模式只拆分EDCA Profile。`alpha/beta1/beta2/gamma`继续由AP和STA共享；某一侧选择独立Default回退索引时，该侧使用P0与原生BEB，另一侧仍可使用自定义Profile/CW。当前动作编码由21维变为38维（两个17维Profile one-hot加4个标量），因此DDQN输入维度相应增加17维。MAC业务、STA/AP原始观测、TCN压缩、Reward和Cost公式不变。

旧YAML没有该字段时默认为`unified`。模式改变会改变Q网络输入和分支数，必须从随机初始化重新训练，不能加载统一模式的Q网络checkpoint继续训练。

## Cost阈值不再使用Default标定

Cost阈值直接由YAML中的人工约束确定：

```yaml
cmdp:
  minimums:
    vo_timely: 0.50
    vi_timely: 0.30
    jain: 0.45
    be_satisfaction: 0.02
    bk_satisfaction: 0.01
  low_priority_cost_budget: 1.0
```

内部四个预算为：

\[
B=(1-T_{VO}^{min},\ 1-T_{VI}^{min},\ 1-J^{min},\ B_{low}).
\]

低优先级Cost定义为：

\[
C_{low}=\max\left(\frac{1-S_{BE}}{1-S_{BE}^{min}},
                        \frac{1-S_{BK}}{1-S_{BK}^{min}}\right).
\]

所以`low_priority_cost_budget: 1.0`并不表示“不惩罚”：`C_low <= 1`恰好要求BE和BK分别达到对应最低满足度。预算小于1更严格，大于1更宽松。连续三个窗口零服务时，该Cost至少升到2。

训练器在每个训练seed开始前仍有一次短暂Default状态采样，仅用于拟合神经网络输入归一化尺度。它不是Cost标定，不会根据Default表现改变人工阈值。

## 修改业务数量

直接修改`traffic`中的`count`：

```yaml
traffic:
  - {name: voip_ul, model: VOIP, direction: UL, ac: VO, count: 40, assignment: spread}
  - {name: gaming_dl, model: GMG, direction: DL, ac: VI, count: 40, assignment: spread}
```

- `count`：独立业务实例数；
- `model`：VOIP、GMG、BV3、BV6、HTTP、FTP或HTTP_REQ；
- `direction`：UL或DL；
- `ac`：VO、VI、BE或BK；
- `assignment`：`spread`、`round_robin`或`explicit`；
- `sta_offset`：可选STA起始偏移；
- `stas`：`explicit`时使用的STA编号列表。

修改数量只改变并发实例数，不改变对应模型的包长和到达过程。数据仍直接进入真实QoS Wi-Fi MAC，不经过TCP/UDP/IP。

## 修改训练参数

全部在同一个YAML的`training`段，例如：

```yaml
training:
  stages: 30
  episode_s: 20.0
  updates_per_stage: 20
  q_lr: 0.0003
  representation_lr: 0.0001
  batch_size: 32
  replay_capacity: 20000
  warmup: 64
  discount: 0.98
  epsilon_start: 1.0
  epsilon_end: 0.05
  epsilon_steps: 450
  grad_clip: 10.0
  target_update_mode: soft
  target_tau: 0.001
```

当前1秒决策周期下，每个seed环境步数为`stages × episode_s`。例如30×20=600步。

## 输出位置

默认由同一YAML指定：

```yaml
output:
  root: results/experiments/mac_cmdp_custom
```

所有内容集中在该目录：

```text
results/experiments/mac_cmdp_custom/
├─ inputs/                 自动生成的topology.csv、flows.csv和manifest
├─ generated_config/       自动生成的底层训练配置
├─ seed_42/                训练、checkpoint和final评估
├─ seed_66/
├─ seed_88/
├─ summary/                汇总CSV
├─ figures/                PNG图
├─ logs/                   运行日志
└─ pipeline_status.json    整体流程状态
```

输出目录记录YAML哈希。修改YAML后入口会拒绝覆盖旧实验；此时修改`output.root`或传入新的`--output`即可。

## 恢复规则

- 完成的seed自动跳过；
- 存在`train/checkpoint.pt`时自动恢复；
- 训练完成但final评估未完成时只补final评估；
- 不做中途checkpoint择优，比较固定使用final checkpoint。
