# Paper-ready framework figure

This directory contains the reproducible framework figure for the current
`candidate_full40_multiac_v6` TCN-DDQN/EDCA system.  The layout follows a
cyclic closed-loop visual schema: distributed STA observation on the left,
centralized AP-side decision making on the right, and the real ns-3.44 MAC at
the bottom.

## Outputs

- `fig_framework_en.pdf` / `fig_framework_zh.pdf`: preferred vector files for
  LaTeX and Word manuscripts.
- `fig_framework_en.svg` / `fig_framework_zh.svg`: editable vector versions.
- `fig_framework_en.png` / `fig_framework_zh.png`: 600 dpi review versions.

Regenerate all outputs from the project root with:

```powershell
.\.venv\Scripts\python.exe paper_figures\make_framework_figure.py
```

## Suggested captions

English:

> Closed-loop architecture of the proposed asynchronous age-aware TCN-DDQN
> controller. Each STA samples per-AC MAC statistics and encodes the causal
> history with a shared TCN. Reports become visible at AP 0 only after a
> successful ACK. AP 0 aggregates the latest valid reports and their ages,
> selects one of 865 constrained EDCA actions, and applies it only to BSS 0;
> the other six co-channel BSSs remain uncontrolled.

Chinese:

> 异步年龄感知 TCN-DDQN 控制器的闭环架构。各 STA 对每个 AC 的 MAC 状态进行采样，
> 并通过共享因果 TCN 压缩历史窗口；报告仅在对应 MPDU 成功确认后对 AP 0 可见。
> AP 0 聚合最新有效报告及其年龄，从 865 个受约束 EDCA 动作中选择联合动作，且仅控制
> BSS 0，其余 6 个同信道 BSS 始终保持默认 EDCA。

## Configuration facts represented in the figure

- 40 STAs and four access categories in the controlled BSS.
- A 10-sample by 6-feature history for every STA-AC pair.
- 8-dimensional TCN latent vectors and a 299-dimensional DDQN input.
- Five action heads of sizes 17/3/2/3/3.
- 16 EDCA profiles times 54 CW tuples, plus one native-default fallback:
  865 executable actions.
- A 1 s control period over a real ns-3.44 802.11ax MAC.

