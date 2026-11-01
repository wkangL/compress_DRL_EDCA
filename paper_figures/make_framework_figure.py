"""Draw the algorithm framework figure (paper Fig. 3-1) in an IEEE-journal style.

Usage:
    python paper_figures/make_framework_figure.py            # both languages
    python paper_figures/make_framework_figure.py --lang zh

Outputs (vector PDF/SVG + 600 dpi PNG) into paper_figures/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- canvas ----
W, H = 100.0, 71.5
FIG_W_IN = 7.0                      # IEEE double-column text width
FIG_H_IN = FIG_W_IN * H / W

INK = "#111111"
GREY = "#5A5A5A"
RA = r"$\Rightarrow$"               # rightwards double arrow, drawn by mathtext

THEME = {
    "sta": dict(fill="#E9F0F6", edge="#2F6187", panel="#A9C2D4"),
    "ap": dict(fill="#FAF1E7", edge="#9C6A37", panel="#DCC0A0"),
    "env": dict(fill="#EFEFEF", edge="#4A4A4A", panel="#4A4A4A"),
}

TEXT = {
    "zh": dict(
        sta_panel="STA 侧：分布式观测（40 STA × 4 AC，编码器权重共享）",
        ap_panel="AP 0 侧：集中决策（全网唯一 RL 决策节点）",
        b1_t="MAC 状态采样（每 100 ms）",
        b1_b="队列字节 / MAC 服务时延 / 重传\n"
             "最终失败 / 丢弃 ＋ 时延有效位\n"
             + RA + "  $x\\in\\mathbb{R}^{10\\times 6}$",
        b2_t="共享因果 TCN 编码",
        b2_b="3 个残差块（d = 1, 2, 4；C = 32）\n"
             "因果卷积，绝不使用未来采样\n"
             "时间维池化 " + RA + " $z\\in\\mathbb{R}^{8}$",
        b3_t="异步报告生成  G → R → X",
        b3_b="G 发送前快照窗口\n"
             "R 仅在该 MPDU 真被 ACK 后生效\n"
             "X 撤销未完成；报告 $(z,\\,t_{\\mathrm{gen}})$",
        link_t="上行报告",
        link_m="$(z,\\,t_{\\mathrm{gen}})$",
        link_b="仅随成功 ACK 到达",
        b4_t="报告缓存 ＋ 编码器版本校验",
        b4_b="只保留最新 $t_{\\mathrm{gen}}$；拒绝跨版本混用；无报告 STA 置 $v_i=0$\n"
             "报告年龄 $a_i(t)=t-t_{\\mathrm{gen}}$，随决策时刻增长",
        b5_t="按 AC 集合聚合  " + RA + "  299 维决策输入",
        b5_b="注意力池化 32 ＋ 逐维最大 32 ＋ 年龄均值/P95/覆盖率 3\n"
             + RA + " 每 AC 67 维（4 AC 共 268）；＋ AP 本地 10 ＋ 上一动作 21",
        b6_t="分支式 Dueling Double DQN",
        b6_b="共享主干 256 → 128；$V(s)$ ＋ 5 个中心化优势分支（17/3/2/3/3）\n"
             "仅 $(\\beta_1,\\beta_2)$ 联合选择；Default 回退将标量归零",
        b7_t="联合约束选择  " + RA + "  联合动作 $a_t$",
        b7_b="16 个 Profile × 54 组 CW 参数 ＋ 1 个原生 Default 回退 = 865 个动作\n"
             "作用域：仅 BSS 0 的 4 个 AC（AP 0 ＋ 40 STA）",
        env_t="ns-3.44 真实 802.11ax MAC 执行（BSS 0：AP 0 ＋ 40 STA，决策周期 1 s）",
        env_b="Default EDCA 竞争 · A-MPDU ＋ Block ACK · 重传上限 10；"
              "逐 MPDU 账本输出时延 / 丢包 / 重传 / 准点交付\n"
              "其余 6 个同信道 BSS 全程保持 Default EDCA ＋ 原生 BEB，作为不可控环境",
        next_epoch="下一决策时刻  $t \\leftarrow t+1\\,\\mathrm{s}$",
        act_label="动作 $a_t$ 按 1 s 周期下发",
    ),
    "en": dict(
        sta_panel="STA side: distributed observation (40 STAs × 4 ACs)",
        ap_panel="AP 0: centralized decision (the only RL agent)",
        b1_t="MAC state sampling (100 ms)",
        b1_b="queue bytes / MAC service delay / retries\n"
             "final failures / drops + delay-valid bit\n"
             + RA + "  $x\\in\\mathbb{R}^{10\\times 6}$",
        b2_t="Shared causal TCN encoder",
        b2_b="3 residual blocks (d = 1, 2, 4; C = 32)\n"
             "causal: never uses future samples\n"
             "temporal pooling " + RA + " $z\\in\\mathbb{R}^{8}$",
        b3_t="Async report  G → R → X",
        b3_b="G snapshots the window before TX\n"
             "R releases it only after a real ACK\n"
             "X cancels " + RA + " report $(z,\\,t_{\\mathrm{gen}})$",
        link_t="uplink report",
        link_m="$(z,\\,t_{\\mathrm{gen}})$",
        link_b="only after ACK",
        b4_t="Report cache + encoder-version check",
        b4_b="keep the newest $t_{\\mathrm{gen}}$; reject mixed versions; "
             "$v_i=0$ if absent\n"
             "report age $a_i(t)=t-t_{\\mathrm{gen}}$ grows between decisions",
        b5_t="Per-AC set aggregation  " + RA + "  299-dim input",
        b5_b="attention pool 32 + per-dim max 32 + age mean/P95/coverage 3\n"
             + RA + " 67 per AC (268 total); + 10 AP-local + 21 last-action features",
        b6_t="Branching dueling double DQN",
        b6_b="shared trunk 256 → 128; $V(s)$ + five centered heads (17/3/2/3/3)\n"
             "joint $(\\beta_1,\\beta_2)$ choice; fallback canonicalizes all scalars to zero",
        b7_t="Constrained joint selection  " + RA + "  action $a_t$",
        b7_b="16 profiles × 54 CW tuples + 1 native-default fallback = 865 actions\n"
             "scope: the 4 ACs of BSS 0 only (AP 0 + 40 STAs)",
        env_t="ns-3.44 real 802.11ax MAC (BSS 0: AP 0 + 40 STAs, 1 s decision period)",
        env_b="default EDCA contention · A-MPDU + Block ACK · retry limit 10; "
              "a per-MPDU ledger yields delay / loss / retransmission / on-time delivery\n"
              "the other six co-channel BSSs keep default EDCA + native BEB as an "
              "uncontrollable environment",
        next_epoch="next decision  $t \\leftarrow t+1\\,\\mathrm{s}$",
        act_label="action $a_t$ applied every 1 s",
    ),
}


def rounded(ax, x, y, w, h, fill, edge, lw=1.0, z=3, rs=1.1):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={rs}",
            linewidth=lw, edgecolor=edge, facecolor=fill, zorder=z,
        )
    )


def panel(ax, x, y, w, h, color):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0,rounding_size=2.0",
            linewidth=0.9, edgecolor=color, facecolor="none",
            linestyle=(0, (5, 3)), zorder=1,
        )
    )


_FITS: list[tuple[object, float, str]] = []
_BADGES: list[tuple[object, float]] = []


def card(ax, x, y, w, h, title, body, theme, badge=None, ts=7.6, bs=6.4):
    rounded(ax, x, y, w, h, THEME[theme]["fill"], THEME[theme]["edge"])
    title_artist = ax.text(x + w / 2, y + h - 2.15, title, ha="center", va="center",
                           fontsize=ts, fontweight="bold", color=INK, zorder=5)
    body_artist = ax.text(x + w / 2, y + h - 4.30, body, ha="center", va="top",
                          fontsize=bs, color=INK, zorder=5, linespacing=1.52)
    _FITS.append((title_artist, w - 3.4, f"card{badge} title"))
    _FITS.append((body_artist, w - 2.0, f"card{badge} body"))
    if badge is not None:
        _BADGES.append((title_artist, x + 4.0))
    if badge is not None:
        ax.add_patch(Circle((x + 2.6, y + h - 2.15), 1.25,
                            facecolor=THEME[theme]["edge"], edgecolor="none", zorder=6))
        ax.text(x + 2.6, y + h - 2.15, str(badge), ha="center", va="center",
                fontsize=6.2, fontweight="bold", color="white", zorder=7)


def arrow(ax, p0, p1, lw=1.0, color=INK, z=4):
    ax.add_patch(
        FancyArrowPatch(
            p0, p1, arrowstyle="-|>", mutation_scale=9, linewidth=lw,
            color=color, linestyle="-", shrinkA=0, shrinkB=0, zorder=z,
            joinstyle="miter", capstyle="butt",
        )
    )


def stack_hint(ax, x, y, w, h, edge, dx=0.55, dy=0.55, n=2):
    """Ghost outlines behind a box: 'this block is replicated'."""
    for i in range(n, 0, -1):
        ax.add_patch(
            FancyBboxPatch(
                (x - dx * i, y + dy * i), w, h,
                boxstyle="round,pad=0,rounding_size=1.1",
                linewidth=0.7, edgecolor=edge, facecolor="white",
                alpha=0.65, zorder=2,
            )
        )


def build(lang: str, out_stem: Path) -> None:
    t = TEXT[lang]
    _FITS.clear()
    _BADGES.clear()
    # Noto Sans SC keeps Chinese labels compact and readable after two-column
    # reduction; Times New Roman matches the surrounding IEEE manuscript.
    family = "Noto Sans SC" if lang == "zh" else "Times New Roman"
    plt.rcParams.update({
        "font.family": family,
        "font.size": 7,
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---------------------------------------------------------- geometry ----
    sta_x, sta_w = 3.0, 29.0
    ap_x, ap_w = 49.0, 46.0
    top, bot = 67.2, 14.5
    sta_gap, ap_gap = 6.0, 3.0
    sta_h = (top - bot - 2 * sta_gap) / 3
    ap_h = (top - bot - 3 * ap_gap) / 4

    y1 = bot
    y2 = y1 + sta_h + sta_gap
    y3 = y2 + sta_h + sta_gap
    ay1 = bot
    ay2 = ay1 + ap_h + ap_gap
    ay3 = ay2 + ap_h + ap_gap
    ay4 = ay3 + ap_h + ap_gap

    # ------------------------------------------------------------ panels ----
    panel(ax, 1.5, 13.2, 33.5, 56.0, THEME["sta"]["panel"])
    panel(ax, 46.5, 13.2, 50.0, 56.0, THEME["ap"]["panel"])
    sta_title = ax.text(3.0, 70.5, t["sta_panel"], ha="left", va="center",
                        fontsize=7.4, fontweight="bold", color=THEME["sta"]["edge"])
    ap_title = ax.text(48.0, 70.5, t["ap_panel"], ha="left", va="center",
                       fontsize=7.4, fontweight="bold", color=THEME["ap"]["edge"])
    _FITS.append((sta_title, 42.0, "STA panel title"))
    _FITS.append((ap_title, 48.0, "AP panel title"))

    # ------------------------------------------------------- STA 1-2-3 ------
    stack_hint(ax, sta_x, y1, sta_w, sta_h, THEME["sta"]["edge"])
    card(ax, sta_x, y1, sta_w, sta_h, t["b1_t"], t["b1_b"], "sta", 1)
    card(ax, sta_x, y2, sta_w, sta_h, t["b2_t"], t["b2_b"], "sta", 2)
    card(ax, sta_x, y3, sta_w, sta_h, t["b3_t"], t["b3_b"], "sta", 3)

    arrow(ax, (sta_x + sta_w / 2, y1 + sta_h), (sta_x + sta_w / 2, y2))
    arrow(ax, (sta_x + sta_w / 2, y2 + sta_h), (sta_x + sta_w / 2, y3))

    # ------------------------------------------------------- AP 4-5-6-7 ----
    card(ax, ap_x, ay4, ap_w, ap_h, t["b4_t"], t["b4_b"], "ap", 4)
    card(ax, ap_x, ay3, ap_w, ap_h, t["b5_t"], t["b5_b"], "ap", 5)
    card(ax, ap_x, ay2, ap_w, ap_h, t["b6_t"], t["b6_b"], "ap", 6)
    card(ax, ap_x, ay1, ap_w, ap_h, t["b7_t"], t["b7_b"], "ap", 7)

    cx_ap = ap_x + ap_w / 2
    arrow(ax, (cx_ap, ay4), (cx_ap, ay3 + ap_h))
    arrow(ax, (cx_ap, ay3), (cx_ap, ay2 + ap_h))
    arrow(ax, (cx_ap, ay2), (cx_ap, ay1 + ap_h))

    # ----------------------------------------------- STA -> AP report link --
    y_mid = y3 + sta_h - 3.7
    arrow(ax, (sta_x + sta_w, y_mid), (ap_x, ay4 + ap_h - 3.7), lw=1.1,
          color=THEME["ap"]["edge"])
    ax.text(40.7, y_mid + 2.2, t["link_t"], ha="center", va="bottom",
            fontsize=6.6, fontweight="bold", color=THEME["ap"]["edge"])
    ax.text(40.7, y_mid - 0.3, t["link_m"], ha="center", va="top", fontsize=6.6)
    link_b = ax.text(40.7, y_mid - 3.1, t["link_b"], ha="center", va="top",
                     fontsize=5.9, color=GREY)
    _FITS.append((link_b, 14.6, "link note"))

    # --------------------------------------------------------- environment --
    env_y, env_h = 1.2, 9.6
    rounded(ax, 1.5, env_y, 97.0, env_h, THEME["env"]["fill"],
            THEME["env"]["edge"], lw=1.1)
    ax.text(50.0, env_y + env_h - 2.4, t["env_t"], ha="center", va="center",
            fontsize=7.4, fontweight="bold", color=INK, zorder=5)
    ax.text(50.0, env_y + env_h - 4.9, t["env_b"], ha="center", va="top",
            fontsize=6.2, color=INK, zorder=5, linespacing=1.5)
    ax.add_patch(Circle((4.05, env_y + env_h - 2.4), 1.25,
                        facecolor=THEME["env"]["edge"], edgecolor="none", zorder=6))
    ax.text(4.05, env_y + env_h - 2.4, "8", ha="center", va="center",
            fontsize=6.2, fontweight="bold", color="white", zorder=7)

    # action down, state up
    arrow(ax, (cx_ap, ay1), (cx_ap, env_y + env_h), lw=1.1)
    ax.text(cx_ap + 2.4, 11.85, t["act_label"],
            ha="left", va="center", fontsize=5.9, color=GREY)

    ax_up = 7.0
    arrow(ax, (ax_up, env_y + env_h), (ax_up, y1), lw=1.1)
    ax.text(ax_up + 1.6, 11.85, t["next_epoch"], ha="left",
            va="center", fontsize=6.1, color=GREY, style="italic")

    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out_stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out_stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.01)

    # ------------------------------------------------ text-fit verification --
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    problems = 0
    for artist, available, tag in _FITS:
        box = artist.get_window_extent(renderer)
        (x0, _), (x1, _) = inv.transform([[box.x0, box.y0], [box.x1, box.y1]])
        width = abs(x1 - x0)
        if width > available:
            problems += 1
            head = artist.get_text().splitlines()[0][:46]
            print(f"  [overflow] {tag}: {width:.1f} > {available:.1f} u :: {head}")
    for artist, left_limit in _BADGES:
        box = artist.get_window_extent(renderer)
        (x0, _), (x1, _) = inv.transform([[box.x0, box.y0], [box.x1, box.y1]])
        if min(x0, x1) < left_limit:
            problems += 1
            head = artist.get_text()[:46]
            print(f"  [badge overlap] title starts at {min(x0, x1):.1f} "
                  f"< {left_limit:.1f} u :: {head}")
    print(f"  text fit: {'OK' if problems == 0 else str(problems) + ' overflow(s)'}")
    plt.close(fig)
    for ext in ("pdf", "svg", "png"):
        p = out_stem.with_suffix("." + ext)
        print(f"{p}  ({p.stat().st_size / 1024:.0f} kB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["zh", "en", "both"], default="both")
    args = ap.parse_args()
    langs = ["zh", "en"] if args.lang == "both" else [args.lang]
    for lang in langs:
        build(lang, HERE / f"fig_framework_{lang}")


if __name__ == "__main__":
    main()
