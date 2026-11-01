# Wi-Fi时序压缩反馈与EDCA控制实验工程

## 当前单BSS MAC-CMDP实验入口

日常复现实验只编辑[`configs/mac_cmdp_experiment.yaml`](configs/mac_cmdp_experiment.yaml)，然后使用两个命令：

```powershell
# 自动生成业务配置、训练/恢复、final checkpoint评估并绘图
.\.venv\Scripts\python.exe scripts/run_mac_cmdp_experiment.py train

# 已有结果只重新生成CSV和PNG
.\.venv\Scripts\python.exe scripts/run_mac_cmdp_experiment.py plot
```

业务数量、人工Cost阈值、训练参数、seed和输出目录都在同一个YAML中；命令行`--output`可临时覆盖输出目录。不再进行Default Cost阈值标定。完整字段说明见[`reports/MAC-CMDP通用YAML实验入口说明.md`](reports/MAC-CMDP通用YAML实验入口说明.md)。下文保留的是TGax多BSS环境及历史实验复现说明。

## 当前阶段：TGax-based 7 BSS唯一候选配置（暂停强化学习）

唯一候选环境为`subset7 / candidate_ftp_bk`：TGax reuse-3几何中的中心同信道BSS和六个最近同信道邻居，7个AP、每AP 40个STA，共280个STA。每BSS在`mix_min1`基础上保留各一路官方业务行为，AC映射为VoIP→VO、BV3/BV6/Gaming→VI、HTTP→BE、FTP→BK；只有FTP应用优先级改变，随机业务模型和真实TCP行为不变。BSS 0是未来唯一智能控制对象；本阶段全部节点使用Default EDCA，不启动TCN/DDQN。完整官方100%业务及全部历史结果继续保留。

官方依据见[`reports/TGax官方参数核对表.md`](reports/TGax官方参数核对表.md)，当前实现、验证和偏差见[`reports/TGax仿真环境说明书.md`](reports/TGax仿真环境说明书.md)。

在PowerShell中从项目根目录执行：

```powershell
# 1. 重新生成确定性坐标、完整官方业务分配和扩展负载
python scripts\build_tgax_s3_inputs.py
python scripts\build_tgax_load_variants.py --scene subset7 --seed 20261021

# 2. 官方100%负载的5秒与20秒正确性/CPU基准
python scripts\run_tgax_calibration_matrix.py --scene subset7 --variants load100 --warmup 2 --duration 5 --drain 1 --jobs 1 --output-root results\raw\tgax_subset7_benchmark_paired_5s_20261022
python scripts\run_tgax_calibration_matrix.py --scene subset7 --variants load100 --warmup 2 --duration 20 --drain 1 --jobs 1 --output-root results\raw\tgax_subset7_benchmark_paired_20s_20261022

# 3. Default EDCA负载扫描；load100是官方计数，其余均为扩展场景
python scripts\run_tgax_calibration_matrix.py --scene subset7 --variants mix_min1,mix_min2,load025,load050,load075,load100,load125,load150,rt125,rt150 --warmup 2 --duration 5 --drain 1 --jobs 2 --output-root results\raw\tgax_subset7_loadscan_paired_5s_20261022

# 4. 汇总与绘图
python scripts\summarize_tgax_subset7_calibration.py --root results\raw\tgax_subset7_loadscan_paired_5s_20261022 --root results\raw\tgax_subset7_benchmark_paired_20s_20261022 --output results\summary\tgax_subset7_calibration_20261022
python scripts\plot_tgax_subset7_calibration.py --input results\raw\tgax_s3_inputs_20261021\subset7 --summary results\summary\tgax_subset7_calibration_20261022 --official-run results\raw\tgax_subset7_benchmark_paired_20s_20261022\load100\seed20261021\default --official-duration 20 --output results\figures\tgax_subset7_calibration_20261022
```

唯一候选配置的复现命令：

```powershell
python scripts\build_tgax_load_variants.py --scene subset7 --seed 20261021
python scripts\run_tgax_calibration_matrix.py --scene subset7 --variants candidate_ftp_bk --warmup 2 --duration 5 --drain 1 --jobs 1 --output-root results\raw\tgax_subset7_candidate_ftp_bk_5s_20261022
python scripts\run_tgax_calibration_matrix.py --scene subset7 --variants candidate_ftp_bk --warmup 2 --duration 20 --drain 1 --jobs 1 --output-root results\raw\tgax_subset7_candidate_ftp_bk_20s_20261022
python scripts\analyze_tgax_candidate.py --input results\raw\tgax_s3_inputs_20261021\subset7\candidate_ftp_bk --run results\raw\tgax_subset7_candidate_ftp_bk_20s_20261022\candidate_ftp_bk\seed20261021\default --output results\summary\tgax_subset7_candidate_ftp_bk_20s_20261022
python scripts\plot_tgax_candidate.py --input-root results\raw\tgax_s3_inputs_20261021\subset7 --summary results\summary\tgax_subset7_candidate_ftp_bk_20s_20261022 --output results\figures\tgax_subset7_candidate_ftp_bk_20s_20261022
```

候选配置的业务分配、指标定义和Baseline结果见[`reports/TGax_7BSS唯一候选配置验证报告.md`](reports/TGax_7BSS唯一候选配置验证报告.md)。

### VoIP8/Gaming4高优先级压力候选（未通过固定环境门禁）

`candidate_voip8_gaming4_ftp_bk`只增加VoIP与Gaming并发数，保留TGax业务生成模型。5秒和20秒通用协议审计通过，但20秒中BSS 0的BK上行没有进入MAC或成功交付，VO上行最终丢包率达到9.89%，因此该配置只保留为压力测试，不替代当前`candidate_ftp_bk`。结果与新旧对比见[`reports/TGax_高优先级并发Baseline分析.md`](reports/TGax_高优先级并发Baseline分析.md)。本轮未启动DDQN。

批处理脚本拒绝覆盖不完整输出。正式运行默认使用`traceLevel=summary`，关闭逐包文本以降低CPU与磁盘负担，同时保留关联、逐应用流、逐BSS、逐AC队列、运行时EDCA、最终MAC结果和CCDF。最终MAC服务时延从MPDU首次进入MAC计到成功ACK/Block ACK；A-MPDU逐MPDU记账，最终失败、队列过期和仿真截止未完成分开输出。详细帧轨迹仍可按需启用。

当前TGax阶段基于ns-3.44真实Wi-Fi模块。Yans后端用于正式CPU可承受的标量同信道干扰评估；Spectrum后端保留用于短时协议和频谱交叉核验。两者都运行真实802.11ax HE PHY/MAC，未删除TCP/UDP、EDCA、聚合、Block ACK、重传或邻居BSS。

## 环境安装

本机：Windows、Python 3.14.4、CPU版PyTorch。项目内已准备`.tools/mingw64`，无需改系统PATH；Python环境位于`.venv`。在PowerShell进入工程目录后运行：

```powershell
.\.venv\Scripts\Activate.ps1
python -m unittest discover -s tests -v
```

重建Python环境：

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`--system-site-packages`用于复用本机已有PyTorch/NumPy；希望完全隔离时去掉此参数并安装全部依赖。实际包版本快照保存于`reports/environment_packages.txt`。

## ns-3固定版本与编译

源码固定到官方ns-3.44标签，commit `43dce6710b8df69685e3479c1d33a571dda714ea`。源码和便携编译器保存在项目中，未安装到系统。

```powershell
git clone --depth 1 --branch ns-3.44 https://github.com/nsnam/ns-3-dev-git.git vendor/ns-3.44
```

Windows便携编译器来源及版本：

```powershell
curl.exe -L --fail -o .tools/winlibs.7z https://github.com/brechtsanders/winlibs_mingw/releases/download/14.2.0posix-12.0.0-ucrt-r3/winlibs-x86_64-posix-seh-gcc-14.2.0-mingw-w64ucrt-12.0.0-r3.7z
```

压缩包SHA256：`0475a96b4d687a3b85dd979a982883ed9172c55b2516e953664232d46471096f`。用7-Zip解压到`.tools`，确保`.tools/mingw64/bin/g++.exe`存在。

```powershell
python scripts/build_ns3.py
```

该命令核验源码commit，应用`ns3_ext/cw_growth.patch`和`ns3_ext/queue_drop_trace.patch`，添加复用原测试的`wifi-txop-nonht`套件，复制独立场景文件并编译Wi-Fi和测试程序。后者仅补齐队列拒绝入队的原生丢包通知，不修改入队/丢弃决策。标准BEB默认关闭研究扩展；增强STA和AP才启用自定义增长。原始上游测试保留不变。Python3.14下不使用不兼容的`python ns3 configure`启动入口。

首次编译需要数分钟。更改`ns3_ext/wifi-control.cc`后必须重新编译和验证，旧验证凭据会阻止训练。

## 按阶段运行

```powershell
# Stage 0：真实EDCA正确性、七类场景、两个种子；失败返回非零
python experiments/run_edca_validation.py

# 在当前PHY下重新标定容量
python scripts/calibrate_capacity.py

# 模块单元测试
python -m unittest discover -s tests -v

# 六组方法，quick两个训练seed、两个评估seed
python run.py --config configs/quick.yaml

# 单独训练完整方案
python run.py --config configs/quick.yaml --phase train --method compressed_async_age --seed 1

# 评估已有模型，路径替换为实际checkpoint
python run.py --config configs/full.yaml --phase evaluate --method compressed_async_age --seed 1 --checkpoint results/raw/RUN/train/checkpoint.pt

# full五个独立训练seed；计算量显著大于quick
python run.py --config configs/full.yaml
```

`run_quick.ps1`和`run_full.ps1`分别是入口包装。full默认100阶段×4 episode×30s，不能把quick结果改名当作full。模型在每个采集阶段固定，阶段末才进行梯度更新，下一episode清空缓存并统一Encoder版本。

主实验入口要求Stage 0通过且源码指纹相符；不提供绕过检查的训练开关。`--reuse-run PATH`只复用同研究配置已经完整训练的checkpoint，保留来源文件，重新评估；允许进程worker数量不同，不允许修改研究定义。

后续生成的checkpoint还保存原始窗口replay、优化器、目标网络及全部随机状态，可用`--phase train --resume PATH`从阶段边界准确续跑。研究设置、seed、method必须相同，仅允许延长`training.stages`及调整worker数量。第一轮早期quick checkpoint没有replay/RNG，只能用于评估，续跑会明确拒绝，不能伪装成无缝恢复。

恢复开发后新增原子保存及checkpoint内阶段日志。若训练尚未达到总阶段数，`--resume`可以直接使用原配置；只有已达到总阶段数时才需增加`training.stages`。以下命令仅运行full的第一阶段，保留完整100阶段研究定义，随后可接续：

```powershell
python run.py --config configs/full.yaml --phase train --method compressed_async_age --seed 1 --stop-after-stage 1
python run.py --config configs/full.yaml --phase train --method compressed_async_age --seed 1 --resume results/raw/RUN/train/checkpoint.pt
```

full在启动前将epsilon衰减从100000修正为9600步，与12000步总采集预算匹配；第42阶段才达到warmup。详细变更理由见实验设计。每阶段进度见`train/status.json`，paused状态不代表训练完成。

已保存的full首阶段为`results/raw/full_20261018_144844/train/checkpoint.pt`（完整方案seed 1，120步、0次更新）。从该点继续时，用此实际路径替换上方`--resume`示例的路径。

最新已完成quick为`results/raw/quick_20261018_141420`。可先核验全部方法和seed，再读取生成的报告：

```powershell
python scripts/audit_run.py --run results/raw/quick_20261018_141420 --require-full-matrix
$env:WIFI_INTEGRATION='1'
python -m unittest discover -s tests -v
```

## 结果、绘图和报告

每次运行创建新目录`results/raw/quick_时间戳`或`full_时间戳`。每种方法、训练seed分别保存：

- `train/config.json`、`metadata.json`、`checkpoint.pt`。
- `train/episodes.csv`、`steps.csv`、`losses.csv`：原始奖励、epsilon、动作、online/target Q、TD target/error、梯度、target距离和latent方差；`replay_audit.csv`保存场景与动作覆盖。
- 每个episode的ns-3参数快照、原始事件`events.log`及stderr。
- `eval/periods.csv`、`evaluation.csv`：逐周期记录和每次评估的汇总。最终p95从成功包逐包时延计算，不取各周期p95平均值。
- Stage 0另保存完整`mac_trace.csv`，包括真实RTS/CTS/DATA/ACK、CW、TXOP和丢弃原因。

使用实际运行目录替换下方`RUN`：

```powershell
python scripts/make_figures.py --run results/raw/RUN
python scripts/plot_training_diagnostics.py --run results/raw/RUN
python scripts/make_comparisons.py --run results/raw/RUN
python scripts/make_report.py --run results/raw/RUN
python scripts/export_profiles.py
```

图表保存于`results/figures/RUN`，每张图同时导出PDF和300dpi PNG。均值和95%置信区间CSV位于`results/summary/RUN`。中文报告位于`reports/实验报告.md`，并在该运行目录中保存副本。仅使用已测量CSV，不生成不存在实验的曲线。

时延图现采用逐成功包CCDF（P(delay>x)，纵轴对数），VO/VI/BE/BK分别展示S1–S4；历史p95图保留但不再作为新报告时延图。新评估保存`delay_samples.npz`，旧结果从原始`events.log`读取。每个训练seed内等权平均评估seed，再跨训练seed等权平均；空成功包分布不伪装为零时延，样本数另存CSV。

`baseline_vs_compressed`子目录包含仅Default EDCA和Compressed Async + Age的15组图。`comparison_percentages.csv`包含全部方法相对baseline的变化；`baseline_vs_compressed.csv/.md`给出完整方案的相对变化及按指标方向计算的改善率。baseline为0时为N/A，负改善率表示恶化，百分比不代表统计显著。

## 续训、阶段独立评估和最终测试

```powershell
python scripts/continue_full.py --output results/raw/full_absolute_monitored_NEW

# 仅当 checkpoint 与当前后端及完整研究配置严格相同时才能断点续训
python scripts/continue_full.py --resume results/raw/COMPATIBLE_RUN/train/checkpoint.pt --output results/raw/full_absolute_resumed_NEW
```

输出目录必须是新目录。旧 `full_monitored_20261018_155026` 由修复丢包记账及重设负载前的配置生成，只作归档，不能续接到当前正式配置。

阶段监测配置为`configs/monitoring.json`，独立于训练研究配置。历史 hard-500 运行评估阶段为1/42/44/50/60/70/80/90/100；阶段 C 后默认使用 soft tau=0.001，不再额外插入“首次硬同步”阶段，常规节点为1/42/50/60/70/80/90/100。每次四场景×seed201/202×30s，greedy且不训练。baseline使用完全相同的监测定义，只评估一次。评估前后恢复Python/NumPy/Torch随机状态，模型单独加载；真实后端测试验证插入评估不改变续训轨迹。

每阶段原子保存完整`train/checkpoint.pt`；每10阶段及评估节点保存`snapshots/stage_XXX.pt`轻量模型快照（仅评估用，不含replay，不能续训）。监测结果在`monitoring/learning_curve.csv`，训练过程的独立评估曲线在`results/figures/运行名/independent_*.png`；阶段CCDF及两方案图在`stage_XXX`子目录。监测阴影是两个评估seed的范围，不是训练seed置信区间。

100阶段完成后自动按原full定义（每场景60s、seed101–105）分别测试baseline及完整方案，输出`final`结果和比较图表。监测seed与最终测试seed隔离；训练仍然只有seed 1，不能据此声称五训练seed实验完成或策略收敛。进展文档和独立实时状态页由运行脚本按阶段更新。

## 工程边界

| 目录 | 职责 |
|---|---|
| configs | 全部研究参数、旧Profile、四场景、sanity及容量标定 |
| ns3_ext | 真实MAC场景、窄范围CW补丁、可审查non-HT测试补充 |
| sim/edca | 无torch依赖的Profile合法性及CW数学规则 |
| sim/backend | 类型化动作、事件及持久ns-3进程接口 |
| sim/traffic | 明确包长分布及实际期望负载 |
| sim/statistics | 全网测量和AP可观测状态分开处理 |
| reporting | 历史窗口、生成快照、成功接收缓存及版本检查 |
| models | 共享TCN、按AC attention/max汇聚、五分支Dueling Q |
| rl | replay、归一化reward、阶段训练和独立评估 |
| experiments | Stage 0及后续实验入口 |
| tests | 协议数学、因果缓存、Scaler、模型置换/掩码/梯度测试 |

## 当前限制

1. 仅已验证802.11a、20MHz、普通DATA/ACK；完整HE TXOP上游测试失败保留在问题记录中。
2. 报告是附着于真实上行发送成功的逻辑捎带事件，没有标准字段编码和实际FP32字节空口开销。FP32 equivalent只表示工程参考开销。
3. Profile采用同步API配置，不计真实广播延迟。不能将其作为完整协议实现性能结论。
4. 随机PHY错误配置目前会明确拒绝，不会默默忽略；完成主实验后再实现并验证该扩展。
5. Full及消融是否完成以最新中文报告为准。quick预算不足以说明最优latent维度、最佳决策周期或渐进部署收益。
6. Checkpoint使用PyTorch序列化；只加载本工程生成且来源可信的文件。

## 2026-10-20 阶段 C：DDQN稳定性诊断

- 修复固定时长episode被错误记为terminal的问题：时间上限现在记为`truncated=True`并继续bootstrap。checkpoint schema 3保存完整replay标记、target更新计数和训练日志。
- target为完整`ControlNetwork`的独立冻结副本，覆盖TCN、STA汇聚器和Q网络；Double DQN由online选动作、target估值。
- 同预算完成H500、soft tau=0.005、soft tau=0.001短诊断。tau=0.005出现更大的Q幅度、梯度峰值和末段TD误差；tau=0.001消除硬同步跳变且数值接近H500，因此作为下一轮多seed候选。默认配置已改为soft tau=0.001。
- 三方案独立评估都随阶段明显波动，第6阶段平均reward均转负；500次更新、单训练seed不能证明收敛。正式训练仍需等待业务标定和固定动作分析。
- 诊断表与图位于`results/summary/diagnostic_target_updates_20261020`和`results/figures/diagnostic_target_updates_20261020`；详细解释见`reports/DDQN训练稳定性分析.md`。显式开启真实后端后45项pytest全部通过。
## 2026-10-20 场景 v2

旧 S2/S4 保留为历史持续过载对照；新的控制实验使用 `configs/scenarios_v2.yaml`。S2/S4 根据 accounting schema 3 多 seed 负载扫描选为旧组成的 0.625 倍，并新增错相两状态动态场景 S5。Default EDCA 的 v2 基线、VO/VI CCDF 和选择依据见 `reports/业务场景与负载标定.md`。S1/S3 中默认 EDCA 已接近理想，这一结果不会为制造 DDQN 优势而改写。

S5 只改变业务到达间隔，不改变 ns-3 的 ACK、AIFS、退避、CW、重试、队列寿命、TXOP 或碰撞路径。后端修改后已重新通过 Stage 0，重新标定容量仍为 27.30225 Mbit/s。

固定动作诊断见 `reports/EDCA固定动作分析.md`。12动作粗扫和3 seed复核表明，P13在S2/S5有稳定收益，S4中P8/P10/P13分别代表实时尾部、总交付、分AC服务的不同权衡；S1/S3默认EDCA本身已很好。固定动作结果只作为学习上界和动作敏感性参考，不按仿真器场景标签部署。

### 阶段 F：因果期限记账与 Reward-v2（2026-10-21）

新增 schema-4 逐包期限记账，VO/VI 期限为 20/50 ms；计数已经通过 ns-3 原生逐 UID 轨迹复算。Reward-v1、Reward-v2 和 Reward-v2+BE/BK 服务保障完成同预算 quick 对照。当前选择 Reward-v2 进入多 seed 中等预算验证；quick 单 seed 结果不构成收敛或显著性证据。详见 [奖励函数对照实验](reports/奖励函数对照实验.md)。





