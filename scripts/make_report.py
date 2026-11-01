"""Write an honest Chinese report from measured results; absent experiments stay absent."""
from __future__ import annotations
from pathlib import Path
import argparse,csv,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from project_config import ROOT
from scripts.make_figures import collect,summarize,METHODS,LABELS,METRICS


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--run",type=Path,required=True); args=parser.parse_args()
    root=args.run.resolve(); summary=summarize(collect(root)); config=json.loads((root/"config.json").read_text())
    lookup={(r["method"],r["scenario"],r["metric"]):r for r in summary}; scenarios=sorted({r["scenario"] for r in summary})
    capacity=json.loads((ROOT/"results/summary/capacity.json").read_text())
    lines=["# Wi-Fi时序状态压缩反馈与DDQN信道接入实验报告","",f"本报告仅使用 `{root}` 中保留的原始CSV。当前模式为 **{config['run']['mode']}**；quick只验证流程和初步趋势，不构成收敛或正式性能结论。","",
    "## 1 实验目的","","分别检验时序状态压缩、异步陈旧报告、显式report age对EDCA控制的影响。没有Decoder，也不以重构R²作为研究目标。","",
    "## 2 系统框架","","```mermaid","flowchart LR","S[STA每AC采样]-->T[共享TCN]-->R[成功上行机会捎带]-->C[AP缓存]-->P[按AC汇聚]-->Q[Branching DDQN]-->E[EDCA与CW增长]-->S","G[AP本地统计]-->Q","```","",
    "## 3 EDCA仿真设置","","采用ns-3.44 Wi-Fi，单BSS、1 AP、10 STA、20 MHz 802.11a，数据54 Mbps、控制6 Mbps。ACK开启，主场景RTS/CTS关闭、无隐藏节点，不额外注入PHY随机丢包；无A-MPDU/Block Ack。四AC独立队列和EDCAF。retry limit=7，队列限额100/200/500/500包，生存期10 s。Profile标准参数来自旧项目，已保留64行完整参数表。", "",
    f"C_ref重新标定为 **{capacity['C_ref_mbps']:.5f} Mbps**。四场景负载因子为0.55/0.85/0.85/1.20。复用原AC比例及UL/DL混合，S1仍以BE为主，并非四类业务各占25%。Poisson到达；VO/BE/BK为160/1500/1600B，VI为裁剪对数正态。", "",
    "## 4 STA观测","","每100ms记录队列字节末值、成功包MAC入队至ACK时延均值、Retry增量、Failed增量、Discarded增量及Delay Validity。没有有效时延时delay=0、valid=0。10个采样点构成每AC的10×6窗口。log1p及标准化参数只从训练采集数据拟合。","",
    "## 5 压缩方法","","所有STA及AC共享一套TCN：3个因果残差卷积块，32通道、kernel=3、dilation=1/2/4，时间均值池化后输出每AC8维。四AC合计32个FP32分量。训练由TD loss反向传播；episode内冻结，统一阶段末更新Encoder版本。没有重构预训练。","",
    "## 6 AP输入","","按AC对STA表示进行共享MLP、单头attention pooling和max pooling；附带有效报告的age均值、P95和coverage。STA编号置换不改变汇聚结果。AP同时使用自身PHY忙比例、接收失败代理、全网成功吞吐、公平性、活跃比例、各AC吞吐及自身下行队列。最近1s和整个决策周期分别统计。AP不读取STA当前队列填补陈旧报告。","",
    "## 7 DDQN动作","","16组Profile以及alpha∈{1,2,3}、beta1∈{1.0,1.2,…,2.0}、beta2∈{1.0,1.2,…,3.0}、gamma∈{5,6,7}。beta2>=beta1通过联合约束掩码保证。CW失败时按分段倍率ceil增长并受CWmax限制，超过gamma保持，成功后恢复CWmin。完整Profile参数见[参数表](EDCA_Profile参数表.md)。","",
    "## 8 Reward","","`r = 1.5 U - 0.5 c - 0.5 l - 0.8 d_vo - 0.6 d_vi - 0.2(1-Jain) - 0.02 s`。U按活跃AC的吞吐满足度加权，权重1.2/1.1/1.0/0.5；VO/VI时延门限20/50ms，超限比例截断到[0,2]。无成功时延样本时惩罚为0且记录无效，过载失败通过drop等项体现。所有主体指标均为比例或速率。","",
    "## 9 EDCA正确性测试","","当前802.11a范围通过DCF/EDCA、信道访问、队列、non-HT TXOP上游测试及七类场景检查。隐藏节点吞吐约从20.23降至11.00 Mbps，RTS/CTS恢复至19.28 Mbps。BEB等效参数与标准BEB统计相同。详见[正确性检查](EDCA仿真正确性检查.md)。完整HE TXOP套件曾越界，未被宣称通过。","",
    "## 10 训练过程","",f"训练种子：{config['run']['seeds']}；评估种子：{config['evaluation']['seeds']}。每个学习方法独立训练，共享同一组研究定义。Quick使用{config['training']['stages']}阶段×{config['training']['episodes_per_stage']} episode×{config['training']['episode_s']}s，不能据此宣称已收敛。检查TD loss、梯度、latent方差和各seed性能后决定是否推进正式训练。每次实验保存配置、原始CSV、checkpoint、seed和运行时间。","",
    "## 11 核心对比结果","","95%区间使用Student t：学习方法先对同一训练seed的评估seed取均值，再跨独立训练seed计算；Default直接跨评估seed。仅两个seed时区间很宽是预期现象，不能据此宣称显著优越。区间可能超出物理范围，保留原值而不截断美化。",
    "", "Full-State Fresh是信息充分的参考，有限训练预算下并不是严格性能上界。若Raw Async或压缩方法优于它，可能包含优化难度、随机初始化和动作探索的影响，不能直接认定减少信息本身提高性能。需充分训练、多seed及动作分布共同解释。"]
    training_index=lines.index("## 11 核心对比结果")
    lines[training_index:training_index]=[
        f"![Training reward](../results/figures/{root.name}/training_reward.png)","",
        "横坐标是训练episode编号，纵坐标是该episode平均reward。淡线保留每个训练seed原值，粗线为跨seed均值的最近4个episode滑动平均，不同颜色和线型对应五种学习方法。场景随机切换也会引起reward波动，不能把平滑曲线上升直接视为策略收敛；需结合固定场景独立评估和TD诊断。",""]
    budget_audit=ROOT/"results/summary"/root.name/"learning_budget_audit.csv"
    if budget_audit.exists():
        with budget_audit.open(encoding="utf-8") as f: budgets=list(csv.DictReader(f))
        primary=[r for r in budgets if r["method"]=="compressed_async_age"]
        evidence=[f"![Q loss](../results/figures/{root.name}/q_loss_main.png)","",
            "横坐标为梯度更新次数，纵坐标为五个动作分支的Smooth L1 TD loss。淡线是每个minibatch原值，实线为最近20次更新滑动平均，不同颜色对应独立训练seed。loss下降只说明当前replay与bootstrap目标下的拟合误差降低，不能证明策略收敛。", "",
            "| 训练seed | episode数 | 决策步数 | 更新数 | 目标网络同步数 | 末步epsilon | 前50次平均loss | 后50次平均loss |",
            "|---|---|---|---|---|---|---|---|"]
        for row in primary:
            evidence.append("|"+"|".join(row[k] if k in ("seed","episodes","transitions","updates","target_syncs") else f"{float(row[k]):.5g}"
                for k in ("seed","episodes","transitions","updates","target_syncs","epsilon_last","loss_first50","loss_last50"))+"|")
        if primary and all(int(r["target_syncs"])==0 for r in primary):
            evidence += ["","**本轮quick尚未经历第一次目标网络同步。** 目标网络每500次更新同步一次，而本轮仅300次更新；因此这组数据不能验证多轮目标网络更新后的学习稳定性。训练过程也尚无按checkpoint生成的独立评估曲线。目前只确认短程loss下降、数值有限及流程可运行，不能写成已收敛。",""]
        evidence += [f"[全部五种学习方法的逐seed Q loss](../results/figures/{root.name}/q_loss_all.png)",""]
        training_index=lines.index("## 11 核心对比结果")
        lines[training_index:training_index]=evidence
    for metric,(filename,label) in METRICS.items():
        lines += ["",f"### {label}","","| 方法 | "+" | ".join(s.split('_')[0] for s in scenarios)+" |","|---|"+"---|"*len(scenarios)]
        for method,title in zip(METHODS,LABELS):
            cells=[]
            for s in scenarios:
                r=lookup.get((method,s,metric))
                cell="未运行" if r is None else f"{r['mean']:.4g} ± {r['ci95']:.3g}" if r["ci95"] is not None else f"{r['mean']:.4g}（仅1 seed）"
                baseline=lookup.get(("default",s,metric))
                if method!="default" and r and baseline:
                    from reporting.comparison import relative_change
                    change=relative_change(baseline["mean"],r["mean"])
                    cell+="；变化N/A" if change is None else f"；{change:+.2f}%"
                cells.append(cell)
            lines.append("|"+title+"|"+"|".join(cells)+"|")
        lines += ["","表中百分比为相对Default均值的变化，负值表示降低；并非显著性检验。baseline为零时为N/A。"]
        if "_delay_" not in metric:
            lines += ["",f"![{label}](../results/figures/{root.name}/{filename}.png)","",
                  f"横坐标为四个业务场景S1–S4，纵坐标为{label}；不同颜色、线型和标记对应六种方法，误差线为95%置信区间。先比较同一场景内的方法，再检查差异是否跨seed稳定。当前quick曲线只能支持数值比较，不能代替正式显著性结论。"]
    for ac in ("vo","vi","be","bk"):
        figure=ROOT/"results/figures"/root.name/f"delay_ccdf_{ac}.png"
        if figure.exists():
            lines += ["",f"### {ac.upper()} 时延CCDF", "",f"![{ac.upper()} delay CCDF](../results/figures/{root.name}/delay_ccdf_{ac}.png)","",
                "横坐标为成功包从MAC入队至ACK的时延（ms），纵坐标为超过该时延的概率P(D>x)，使用对数轴。四个子图是S1–S4，不同曲线对应各方法；曲线越靠左下，成功包时延尾部越小。先在每个训练seed内等权平均评估seed的CCDF，再等权平均训练seed。未成功包不在分布内，必须结合丢包率和队列解读；零概率只保留在CSV，不画到对数轴上。"]
    pair_table=ROOT/"results/summary"/root.name/"baseline_vs_compressed.md"
    if pair_table.exists():
        lines += ["","### 仅baseline与完整方案的对比", "",pair_table.read_text(encoding="utf-8").replace("# Default EDCA 与 Compressed Async + Age 对比",""),
            "",f"[两方案图集](../results/figures/{root.name}/baseline_vs_compressed)",""]
        for ac in ("vo","vi","be","bk"):
            lines += [f"![{ac.upper()} 两方案CCDF](../results/figures/{root.name}/baseline_vs_compressed/delay_ccdf_{ac}.png)","",
                "横轴是逐包时延，纵轴是超越概率；灰色为Default EDCA，橙色为完整方案，四个子图对应S1–S4。这里只显示两种方法，成功包统计口径及限制同上。",""]
    lines += ["","比较关系：Full-State Fresh与Compressed Fresh隔离压缩影响；Compressed Fresh与Compressed Async + Age隔离异步陈旧性；No-Age与+Age检验时效输入；Raw Async与Compressed Async + Age检验同类报告机会下的压缩差异。下表给出goodput相对变化，负值表示后一方法吞吐更低。","",
              "| 对比（后者相对前者） | "+" | ".join(s.split('_')[0] for s in scenarios)+" |","|---|"+"---|"*len(scenarios)]
    for left,right in [("full_fresh","compressed_fresh"),("compressed_fresh","compressed_async_age"),("compressed_async_no_age","compressed_async_age"),("raw_async","compressed_async_age")]:
        cells=[]
        for s in scenarios:
            a,b=lookup.get((left,s,"goodput_mbps")),lookup.get((right,s,"goodput_mbps"))
            cells.append(f"{(b['mean']/a['mean']-1)*100:+.2f}%" if a and b and a["mean"] else "未运行")
        lines.append(f"|{left} → {right}|"+"|".join(cells)+"|")
    lines += ["","### 训练诊断与结果解读",""]
    diagnostic_path=ROOT/"results/summary"/root.name/"training_diagnostics.json"
    if diagnostic_path.exists():
        diagnostics=json.loads(diagnostic_path.read_text(encoding="utf-8"))
        finite=sum(bool(d["finite"]) for d in diagnostics)
        compressed=[d for d in diagnostics if d["run"].startswith("compressed")]
        warnings=sum(bool(d["low_variance_dimensions"]) for d in compressed)
        lines += [f"保留的{len(diagnostics)}个训练任务中，{finite}个的TD loss、Q值和梯度统计均为有限值；{len(compressed)}个压缩模型中有{warnings}个触发末100次更新的latent低方差阈值。原始窗口方法中常量特征的零方差不能当作Encoder坍缩。动作频率及逐seed诊断见`results/summary/{root.name}`。",""]
    if config['run']['mode']=="quick":
        lines += ["以下比较只解释已有quick数据。平均吞吐增加必须同时结合时延、公平性与跨训练seed波动；两个seed不足以证明收敛或显著收益。",""]
        for s in scenarios:
            a=lookup.get(("default",s,"goodput_mbps")); b=lookup.get(("compressed_async_age",s,"goodput_mbps"))
            if a and b:
                interval=f"，完整方案95%区间半宽{b['ci95']:.3f} Mbps" if b['ci95'] is not None else ""
                lines += [f"- {s.split('_')[0]}：完整方案平均{b['mean']:.3f} Mbps，默认EDCA为{a['mean']:.3f} Mbps{interval}。"]
    for number,title,body in [(12,"决策周期","主对比默认1s。5/10s action hold尚未测量，不能回答最大可接受更新间隔。"),
                              (13,"历史窗口","主对比1s。0.5/3s尚未独立训练评估，不能确定最合适窗口。"),
                              (14,"latent维度","主对比每AC8维。4/16维尚未对照，不能声称8维最优。"),
                              (15,"共存实验","0/20/40/60/80/100%增强比例的模型评估尚未完成，不能据现有结果判断渐进部署收益。"),
                              (16,"Hidden Node与RTS/CTS鲁棒性","Stage 0已验证基础物理趋势；学习模型的鲁棒性评估尚未完成。随机PHY错误尚未启用，不能推断模型对非碰撞Retry的判断能力。")]:
        lines += ["",f"## {number} {title}","",body]
    lines += ["","## 17 结论","","当前已建立基于真实ns-3的模块化工程，并完成当前协议范围的正确性验证。核心控制性能差异以第11节实测表为准；quick预算不足以回答正式收敛、最佳latent维度、最大决策周期或渐进部署收益。不得把原交底书性能数字作为本次结果，也不得只选最好seed。","",
              "## 18 当前不足","","- 尚未完成五训练seed正式实验及后续消融。","- 报告为真实成功上行机会附着的逻辑标签，未序列化为标准协议字段；FP32 equivalent overhead仅是浮点实现工程参考，不代表最终协议bit开销或已扣除空口容量。",
              "- Profile采用同步配置API，未建模真实Beacon广播延迟。","- 当前仅802.11a普通DATA/ACK，未验证HE聚合路径、随机PHY误码或真实设备。","- 截止时间队列中尚未完成的包单独留在队列中，不视为已失败；仅成功包构成时延分布，过载时应结合drop、queue一起解读。","- 现有任务没有bit量化、真实芯片部署或Encoder更新的空口模型分发。"]
    path=ROOT/"reports/实验报告.md"; path.write_text("\n".join(lines)+"\n",encoding="utf-8")
    # The archived copy has a different directory depth; preserve usable figure/reference links.
    archived="\n".join(lines)+"\n"
    archived=archived.replace("../results/figures/","../../figures/")
    for reference in ("EDCA_Profile参数表.md","EDCA仿真正确性检查.md"):
        archived=archived.replace(f"]({reference})",f"](../../../reports/{reference})")
    (root/"实验报告.md").write_text(archived,encoding="utf-8")
    print(path)


if __name__=="__main__": main()
