# 单BSS高Gaming场景CMDP TCN-DDQN训练报告

> **历史环境说明（2026-10-30补充）：** 本报告对应的已完成训练使用了自定义64包MAC准入高水位。该参数没有IEEE TGax或ns-3规范依据，后续正式实验已停用；本报告与原始结果仅用于保留历史审计，不作为无水位新环境的性能结论。

## 1. 实验结论

本轮完成了Default预算标定、STEP等价门禁、动态动作生效门禁、单训练seed 44的400步训练、seed 45/46阶段验证和seed 47留出测试。训练数值稳定，400步checkpoint被验证规则选中。

留出seed 47上，学习策略相对Default同时提高了VO/VI及时交付、VI尾时延、四AC有效服务和需求归一化Jain；但VO上行p95增加15.9%，且BE外生生成量未做到严格配对。因此，本轮结果证明CMDP目标比此前的标量Reward更有希望，并证明动态EDCA/CW存在优化潜力；它仍只是单训练seed的初步结果，不能宣称算法已经稳定优于Default。

## 2. 固定环境与CMDP定义

- 网络：单BSS，1 AP + 40 STA，802.11ax HE SU，5 GHz，80 MHz，HE MCS7。
- 业务：MAC直注随机到达；保留VoIP、Gaming、BV3、FTP、HTTP的状态、间隔和包长模型，不经过TCP/UDP/IP。
- 压力：每个STA配置2路Gaming上行与2路Gaming下行，共160路Gaming；其他业务保持高Gaming复测配置。
- 控制周期：1 s；每episode 20步；共20 episode、400环境步。
- 动作：EDCA Profile、alpha、beta1、beta2、gamma五分支，含独立原生Default+BEB回退动作。

效用为

\[
U_t=0.4f(S_{VO})+0.4f(S_{VI})+0.1f(S_{BE})+0.1f(S_{BK}),\quad
f(S)=\frac{\log(1+9S)}{\log 10}.
\]

### Cost的准确计算

当前四个Cost都**不区分上行和下行**。后端仍分别保存UL/DL账本和性能指标，但进入CMDP时先在同一AC内合并两个方向：

\[
T_{VO}=\frac{N^{UL}_{VO,on-time}+N^{DL}_{VO,on-time}}
{N^{UL}_{VO,deadline}+N^{DL}_{VO,deadline}},\qquad
C_{VO}=1-T_{VO},
\]

\[
T_{VI}=\frac{N^{UL}_{VI,on-time}+N^{DL}_{VI,on-time}}
{N^{UL}_{VI,deadline}+N^{DL}_{VI,deadline}},\qquad
C_{VI}=1-T_{VI}.
\]

VO期限为20 ms，VI期限为50 ms；期限从业务有效载荷生成时刻开始，到最终获得成功MAC ACK/Block ACK结束。最终失败、过期以及到期时仍未完成的需求均不能计入及时成功。如果某窗口没有到期样本，该Cost不参与本episode的乘子更新。

对40个有需求STA，先计算每个STA在当前窗口的需求满足比例`s_i`，再计算需求归一化Jain：

\[
J=\frac{(\sum_i s_i)^2}{n\sum_i s_i^2},\qquad C_J=1-J.
\]

这里每个STA的`s_i`也合并了该STA的全部AC及上下行业务。低优先级Cost使用合并方向后的BE/BK需求满足度：

\[
C_L=\max\left(
\frac{1-S_{BE}}{1-\theta_{BE}},
\frac{1-S_{BK}}{1-\theta_{BK}}
\right),
\]

其中`theta_BE=0.1525731`、`theta_BK=0.0715251`。若BE或BK有需求但连续3个1秒窗口没有获得任何有效载荷服务，则强制`C_L>=2`。

Default seed 41/42/43标定得到预算：

| Cost | 预算d | 标定含义 |
|---|---:|---|
| C_VO | 0.3318128 | VO及时率不低于Default聚合值的95% |
| C_VI | 0.6579253 | VI及时率不低于Default聚合值的95% |
| C_J | 0.3582952 | Jain不低于Default均值的95% |
| C_L | 1.0000000 | BE和BK均达到各自Default满足度的80% |

TD训练使用当前乘子重新组合Replay中保存的原始效用与Cost，并且只惩罚超过预算的部分：

\[
r_t=U_t-\sum_i\lambda_i\max(0,C_{i,t}-d_i).
\]

乘子初值0.1、学习率0.01、EMA系数0.2、上限1.0，每20步使用episode平均Cost更新一次。Replay保存原始utility/cost，采样时使用当前lambda重组，未把旧lambda固化进历史样本。

当前设计方向不分离的直接影响是：UL样本很多时会主导VO/VI及时率，少量DL退化可能被UL掩盖；反过来也无法对UL、DL设置不同预算。本轮场景以大量上行STA竞争为主体，因此采用了AC级合并Cost，但报告仍分别输出UL/DL时延、吞吐和失败率。若下一阶段需要同时保证上下行，应扩成`C_VO_UL/C_VO_DL/C_VI_UL/C_VI_DL`或至少对两个方向分别设门禁。

### Cost预算口径复核中发现的限制

补写公式时复核发现，VO/VI预算使用20秒聚合及时率标定，与逐窗口Cost的加权结果较接近；但Jain和低优先级预算使用20秒最终累计满足度标定，而训练Cost按1秒窗口计算，二者时间口径不一致。Default三seed的逐窗口平均Cost与冻结预算如下：

| Cost | Default逐窗口均值 | 冻结预算 |
|---|---:|---:|
| C_VO | 0.296 | 0.332 |
| C_VI | 0.640 | 0.658 |
| C_J | 0.743 | 0.358 |
| C_L | 1.317 | 1.000 |

因此Default在VO/VI预算下基本可行，但在当前`C_J/C_L`逐窗口口径下并不满足以自身累计结果生成的预算。该问题不会改变已经记录的原始性能数据和动作生效事实，但会影响“约束相对Default校准”的严格解释，也会让Jain/低优先级乘子承担比原计划更强的压力。下一版训练前应统一口径：要么用Default逐窗口Cost的分布标定预算，要么把Jain和BE/BK Cost改成与标定一致的跨窗口累计统计；本报告不追溯修改本次已完成训练。

## 3. 训练参数与门禁

| 项目 | 数值 |
|---|---:|
| 训练seed | 44 |
| 环境步 / 梯度更新 | 400 / 340 |
| Replay容量 / warmup / batch | 20000 / 64 / 32 |
| 折扣因子 | 0.98 |
| Q网络学习率 | 3e-4 |
| TCN与聚合器学习率 | 1e-4 |
| ε | 1.0线性降至0.05，300步 |
| Target更新 | soft，tau=0.001 |
| 梯度裁剪 | norm 10 |

Default STEP等价门禁通过：seed 41/42/43相对一次性Default的总吞吐偏差分别为0.207%、0.027%、0.252%。P8+自定义CW动作生效门禁也通过，41个节点×4 AC均真实写入，记录到68,420次CW运行事件。

## 4. 训练稳定性

Q-loss前20次更新均值为0.04056，后20次为0.00192；Online Q最大0.8195，Target Q最大0.1828，梯度范数最大0.5112。全部340次更新均为有限值，没有NaN、Inf、Q值爆炸或梯度异常。

最终Lagrange乘子为`[0.0697, 0.0932, 0.1701, 0.1493]`。公平性与低优先级Cost对应乘子高于VO/VI，说明训练确实感知到了公平性和低优先级服务压力，而不是只优化高优先级及时率。

![训练诊断](../results/figures/mac_arrival40_gaming160_cmdp_seed44_400step_20261030/training_diagnostics.png)

Q-loss单图：

![Q-loss](../results/figures/mac_arrival40_gaming160_cmdp_seed44_400step_20261030/q_loss.png)

## 5. 独立验证与checkpoint选择

100/200/300/400步均在seed 45/46上独立运行20秒贪心验证，验证数据不进入Replay。四个checkpoint均通过预先固定的及时率、总交付、BE/BK和公平性门禁；400步的平均CMDP utility最高（0.5030），因此选中400步。最后checkpoint也保留，恰好与选中checkpoint相同。

400步验证中，VO及时率为81.16%/89.18%，VI及时率为39.08%/39.69%；早期checkpoint并非单调变化，说明训练曲线仍存在明显方差，不能只看单次Reward上升判断收敛。

![独立验证](../results/figures/mac_arrival40_gaming160_cmdp_seed44_400step_20261030/validation_learning_curve.png)

## 6. 留出seed 47：Default与学习策略

| 指标 | Default | CMDP-DDQN | 变化 |
|---|---:|---:|---:|
| CMDP utility | 0.2764 | 0.4632 | +67.59% |
| VO 20 ms及时率 | 67.36% | 90.82% | +23.47个百分点 |
| VI 50 ms及时率 | 36.11% | 39.52% | +3.40个百分点 |
| 需求归一化Jain | 0.1380 | 0.2151 | +55.95% |
| VO需求满足度 | 71.64% | 97.88% | +36.61% |
| VI需求满足度 | 38.10% | 41.28% | +8.35% |
| BE需求满足度 | 17.92% | 56.98% | +217.93%* |
| BK需求满足度 | 0.40% | 39.23% | +9762.46%* |
| UL VO p95 | 25.77 ms | 29.87 ms | **+15.90%（恶化）** |
| UL VI p95 | 70.55 ms | 58.90 ms | -16.51% |
| UL VO吞吐 | 0.195 Mbps | 0.267 Mbps | +37.00% |
| UL VI吞吐 | 9.228 Mbps | 9.411 Mbps | +1.99% |
| UL BE吞吐 | 1.435 Mbps | 5.963 Mbps | +315.46%* |
| UL BK吞吐 | 0.0245 Mbps | 2.309 Mbps | +9315.61%* |

星号项必须谨慎解释。seed 47的VO、VI、BK总生成需求在Default与学习策略间相同，但BE生成量不相同：Default约0.213 MB，学习策略约3.448 MB。FTP/HTTP类状态机包含服务完成后才触发的后续对象/阅读阶段，动作改变会影响有限窗口内实际产生的后续需求，因此`paired_generated_demand_equal_each_step=False`。学习策略的BE/BK服务改善是真实观测值，但BE改善百分比不是严格外生需求配对下的因果估计。VO/VI的生成总量一致，及时率与吞吐比较更可信。

另外，VO及时率明显改善但成功样本p95反而从25.77 ms升至29.87 ms。两者分母不同：及时率包含到期需求的成功/失败/未完成结果，p95只统计最终成功确认的MPDU。Default丢掉更多困难包，会让成功样本p95看起来较低，因此不能单独用p95否定及时率改善。

![留出核心指标](../results/figures/mac_arrival40_gaming160_cmdp_seed44_400step_20261030/held_out_core_metrics.png)

下面的CCDF直接来自留出seed 47的`events.log`，严格截取控制时间1–21秒的全部逐MPDU成功确认样本，不包含此前额外的1秒Default业务预热，并按VO/VI及上下行分别绘制。主图纵轴为P0/P50/P80/P90/P95/P98/P99，只展示到P99；更深尾部另存为审计图而不作为主结论。横轴是**MAC入队到成功ACK**，不包含业务生成后在直注待提交队列中的准入等待；它只包含最终成功ACK的MPDU，必须与采用“业务生成到ACK”口径的期限及时率、失败率、未完成数量及需求债务联合解释。四条曲线存在交叉，图中保留真实交叉关系，不把某个区间裁剪成算法全程占优。

![VO/VI MAC服务时延CCDF](../results/figures/mac_arrival40_gaming160_cmdp_seed44_400step_20261030/held_out_vo_vi_delay_ccdf.png)

不区分上下行的AC级CCDF直接合并20秒受控窗口内的全部UL/DL成功MPDU，因此自然按实际成功包数量加权，不会放大少量DL样本。

![VO合并方向CCDF](../results/figures/mac_arrival40_gaming160_cmdp_seed44_400step_20261030/held_out_vo_delay_ccdf_combined.png)

![VI合并方向CCDF](../results/figures/mac_arrival40_gaming160_cmdp_seed44_400step_20261030/held_out_vi_delay_ccdf_combined.png)

## 7. 学到的动作行为

### VI下行专项复核

严格截取20秒受控窗口后，VI下行的生成到ACK 50 ms及时率在Default和CMDP-DDQN下均为0/86,180=0%。学习策略把成功吞吐从0.345提高到0.960 Mbps、成功MPDU从1,952提高到5,470，重传率从5.37%降至4.74%；但成功样本p95从12.16 ms升至60.68 ms，MAC最终失败率从3.59%升至5.08%。这表明学习策略显著提高了DL-VI准入和服务数量，却没有清除业务侧既有准入积压，所有在窗口内到期的DL-VI需求仍错过50 ms期限。当前合并方向的VI Cost被上行及时样本主导，不能保证VI下行期限，下一版应拆分方向Cost或设置DL硬门禁。

留出seed 47的20个决策窗口没有退化为单一动作：Profile选择P10 11次、P13 5次、P6和P14各2次；alpha物理值3选择19次、1选择1次；beta1物理值1.6/2.0各10次；beta2物理值2.8选择19次、2.4选择1次；gamma物理值6选择20次。策略主要在P10与P13之间切换，并在运行中调整beta1，而非固定复现某个单一EDCA配置。原始动作数组保存的是候选列表索引，例如beta1索引1表示2.0，并不表示beta1=1.0。

![动作分布](../results/figures/mac_arrival40_gaming160_cmdp_seed44_400step_20261030/held_out_action_distribution.png)

## 8. 最终判断与下一步

本轮达成了工程目标：生成到最终ACK期限账本、原始utility/cost Replay、动态lambda、TCN-DDQN反向传播、在线动作和独立评估闭环均已打通；400步训练数值稳定，并在一个留出seed上同时改善了VO/VI及时率、VI p95、公平性和低优先级服务。

但当前证据不足以宣称“稳定优于Default”，原因有三点：只有一个训练seed和一个最终留出seed；checkpoint间性能存在波动；FTP/HTTP完成驱动的后续需求使BE有限窗生成量不能严格配对。下一阶段若继续，应先把所有业务生成器的外生随机流与服务过程完全解耦或预生成需求轨迹，再做至少3个训练seed与多个严格配对留出seed。无需因为本轮Q-loss正常下降而修改Target更新。

## 9. 结果位置

- 原始训练、checkpoint、验证和留出结果：`results/raw/mac_arrival40_gaming160_cmdp_seed44_400step_20261029`
- 汇总CSV：`results/summary/mac_arrival40_gaming160_cmdp_seed44_400step_20261030`
- PNG图：`results/figures/mac_arrival40_gaming160_cmdp_seed44_400step_20261030`
- 标定结果：`results/raw/mac_arrival40_gaming160_cmdp_default_calibration_20261029/calibration.json`
- 冻结配置：`configs/mac_arrival40_gaming160_cmdp_calibrated.yaml`
