# TGax参数驱动的简化3-BSS环境交付清单

更新时间：2026-10-21  
状态：Default EDCA环境阶段完成，暂停进入TCN/DDQN。

## 主配置与实现

- `configs/tgax_s3_3ap_10sta.json`：3 AP、10 STA/BSS、AP 0为未来目标、RL关闭。
- `scripts/build_tgax_s3_inputs.py`：生成确定性坐标、最强AP关联及逐行四分之一TGax业务。
- `ns3_ext/tgax-s3.cc`：真实ns-3.44 HE SU、TCP/UDP、Default EDCA、A-MPDU、Block ACK及事件追踪。
- `scripts/run_tgax_s3.py`：`main3`为默认运行变体。

## 说明和核对

- `reports/TGax官方参数核对表.md`
- `reports/TGax仿真环境说明书.md`
- `reports/TGax与ns3差异清单.md`
- `reports/TGax实验进展.md`
- `README_CN.md`

## 输入和正式运行

- 输入：`results/raw/tgax_s3_inputs_20261021/main3/`
- 冒烟：`results/raw/tgax_s3_main3_smoke_20261021/`
- 正式Default EDCA：`results/raw/tgax_s3_main3_full_default_edca_20261021/`
- 验证：正式运行目录中的`validation.md`与`validation.json`，状态PASS。

## 图形

- PNG与交互HTML：`results/figures/tgax_s3_3ap_20261021/`
- PDF：`output/pdf/01_topology_actual.pdf`至`08_delay_ccdf.pdf`

八组图依次覆盖实际拓扑、信道分配、传播/CCA、业务配置、应用时间线、MAC交换时间线、逐BSS Default EDCA结果和MAC接入时延CCDF。图中的坐标、事件和数值均来自正式输入或ns-3轨迹。

## 已通过的检查

- 6项输入生成单元测试。
- 30/30 STA关联与最强BSSID匹配。
- 75个配置应用分量及真实HTTP反向TCP请求。
- QoS DATA、MAC ACK、Block ACK、A-MPDU、纯TCP ACK及VO/VI/BE实际使用。
- TGax业务事件审计。
- 42/42与42/58信道隔离配对检查。
- ns-3 Wi-Fi EDCA、聚合、Block ACK、重传、inter-BSS、MAC队列及Spectrum PHY测试套件。
- 8份PDF已重新栅格化并检查可读性。

## 审核边界

当前只有seed `20261021`的5秒正式环境验证。它能证明场景链路、竞争、协议和统计工作正常，不能用于证明任何学习算法有效，也不能当作多seed性能结论。完整Model D小尺度多径、HTTP连接复用和长时FTP第二文件连接仍在差异清单中明确保留。
