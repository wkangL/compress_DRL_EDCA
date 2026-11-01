"""Generate the no-results-yet report from measured Stage 0 evidence only."""
from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from project_config import ROOT


def main() -> None:
    target=ROOT/"reports/实验报告.md"
    if target.exists():
        raise SystemExit("Existing report retained; use make_report.py with the selected completed run")
    validation=json.loads((ROOT/"results/summary/edca_validation.json").read_text(encoding="utf-8"))
    capacity=json.loads((ROOT/"results/summary/capacity.json").read_text(encoding="utf-8"))
    content=f'''# Wi-Fi时序压缩反馈与EDCA控制阶段报告

本报告记录已完成的协议与工程验证。核心六组方法正在运行，尚无完整可比较的双种子结果，不能评价压缩、异步或age的性能收益。

## 已测量事实

- 当前802.11a参考后端Stage 0：{"通过" if validation["passed"] else "未通过"}。
- C_ref：{capacity["C_ref_mbps"]:.5f} Mbps，来自54/81Mbps高负载平台测试，各3个seed。
- 单STA低负载实测碰撞率为0；竞争随STA数量增加；VO/VI时延整体低于BE/BK。
- 隐藏节点参考吞吐约20.23→11.00Mbps，RTS/CTS恢复至约19.28Mbps。
- 自定义beta1=beta2=2、gamma=7与标准BEB统计相同。
- 完整方案已完成单seed闭环检查：160个transition、300次TD更新，未出现非有限损失。

## 系统与研究设置

详见[实验设计说明](实验设计说明.md)和[完整Profile表](EDCA_Profile参数表.md)。采用STA每100ms采样、每AC最近1s的10×6窗口，共享TCN输出每AC8维；报告只有真实成功上行机会才能更新AP缓存。AP按AC attention+max汇聚，并结合自身观测和当前动作。五分支DDQN输出Profile及alpha、beta1、beta2、gamma。

## 目前不能回答的问题

双seed主对比尚未结束；不能报告压缩损失、异步损失、age收益。5/10s决策周期、0.5/3s窗口、4/16维latent、设备共存和学习策略鲁棒性均未完成，不能下结论。Full至少5个训练seed尚未启动。

## 限制

当前PHY为802.11a普通DATA/ACK，无A-MPDU/Block Ack。完整HE TXOP上游测试越界，未宣称通过。报告为逻辑捎带标签，不代表已计入真实字段空口开销；FP32等价只作工程参考。Profile采用理想同步配置，不含Beacon分发延迟。无量化和真实设备实验。

结果完成后使用 `python scripts/make_report.py --run results/raw/RUN` 生成18节完整中文报告；保留所有原始CSV和seed，不按优劣筛选。
'''
    target.write_text(content,encoding="utf-8")


if __name__=="__main__": main()
