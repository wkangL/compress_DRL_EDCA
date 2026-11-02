# 平衡版EDCA Profile设计说明

## 1. 调整原因

历史 `configs/profiles.json` 的16个Profile主要探索VO/VI接入能力。除Default P0外，多数Profile增大BE/BK的AIFS或CWmin，且所有BE/BK TXOP均为0，因此动作空间缺少主动保障低优先级业务的控制手段。

本轮新增 `configs/profiles_v2_balanced.json`。历史Profile不删除、不覆盖，旧checkpoint仍只能与旧Profile表配套使用。新训练从随机参数初始化。

## 2. Profile分组

| Profile | 定位 | 关键变化 |
|---|---|---|
| P0 | Default基准 | 与ns-3 Default EDCA参数逐项一致 |
| P1–P4 | 代表性旧动作 | 保留轻度高优先级、长TXOP和保守配置 |
| P5 | BE轻度保护 | BE CWmin 15→7、CWmax 255、TXOP 1.504 ms |
| P6 | BK轻度保护 | BK AIFSN 7→5、CWmax 255、TXOP 1.504 ms |
| P7 | BE/BK联合轻度保护 | 同时采用P5、P6的低优先级参数 |
| P8 | BE中度保护 | BE AIFSN 3→2、CWmin 7、TXOP 2.528 ms |
| P9 | BE强保护 | BE AIFSN 2、CWmin 3、TXOP 3.008 ms |
| P10 | BK中度保护 | BK AIFSN 7→3、CWmax 127、TXOP 2.528 ms |
| P11 | BK强保护 | BK AIFSN 2、CWmin 7、TXOP 3.008 ms |
| P12 | BE/BK联合中度保护 | BE采用P8，BK采用P10 |
| P13 | BE/BK联合强保护 | BE采用P9，BK采用P11 |
| P14 | 低优先级偏置 | 适度收紧VO/VI，同时联合保护BE/BK |
| P15 | 低优先级应急动作 | 明显收紧VO/VI，为BE/BK提供AIFSN 2和4.096 ms TXOP |

其中P5–P15提供11种具有不同强度和对象的低优先级服务动作。P14/P15用于验证在严重饥饿状态下主动让渡高优先级竞争机会是否有价值，不代表推荐的固定EDCA配置。

## 3. 合法性和实现门禁

- 16个Profile均包含VO/VI/BE/BK完整参数。
- CWmin/CWmax均为 `2^m-1` 且满足CWmin≤CWmax。
- AIFSN范围为2–7。
- 所有TXOP均为32 μs整数倍。
- P0与历史P0完全相同。
- 真实ns-3 STEP2门禁采用AP=P13、40个STA=P15，`edca_step_runtime.csv`记录的164项运行参数全部匹配；验证文件为 `results/smoke/balanced_profiles_split_gate_20261102/verification.json`。

BE/BK非零TXOP和更激进AIFS/CW属于TGax-based研究动作，而非IEEE默认EDCA参数。策略仍需同时接受四项VO/VI方向Cost与Starvation Cost约束。

## 4. 400步实验配置

- AP/STA Profile分离，四个CW增长参数共享；合法动作仍为15553种。
- 训练seed 42，从随机初始化训练20个episode×20步，共400步。
- Replay warmup 64，epsilon在300步由1.0衰减到0.05。
- Final checkpoint固定使用step 400，不进行中途checkpoint择优。
- 评估seed 66，Default与Final各运行40 s。
- 业务、Reward、五项Cost预算、TCN、聚合器、学习率和soft Target更新均保持上一轮不变。
