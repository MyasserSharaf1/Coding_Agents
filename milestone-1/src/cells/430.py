import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch

# Slots 1-4 in fixed order (validated for adjacent pairs); a test-agent system is also hatched.
_SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SYS_COLOR = {s: _SLOTS[i % len(_SLOTS)] for i, s in enumerate(SYSTEMS)}
SYS_MARKER = {s: ("D" if s in TEST_AGENT_SYSTEMS else ["o", "o", "s", "^"][i % 4])
              for i, s in enumerate(SYSTEMS)}
# Roles get their own validated set so a role is never mistaken for a system.
ROLE_COLOR = {"generator": "#2a78d6", "test_writer": "#4a3aa7",
              "supervisor": "#e87ba4", "repair": "#008300"}
ROLE_LABEL = {"generator": "code agent - first draft", "test_writer": "test agent",
              "supervisor": "supervisor", "repair": "code agent - repair"}
C_GOOD, C_CRITICAL = "#0ca30c", "#d03b3b"      # reserved status colours
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
SYS_HATCH = {s: ("///" if s in TEST_AGENT_SYSTEMS else None) for s in SYSTEMS}   # secondary encoding

mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "hatch.color": "white",
})

SYS_HANDLES = [Patch(facecolor=SYS_COLOR[s], hatch=SYS_HATCH[s], edgecolor="white",
                     label=SYSTEM_LABEL[s]) for s in SYSTEMS]
N_SYS = len(SYSTEMS)
BAR_W = 0.8 / N_SYS
BENCH_TICKS = ["%s\n(n = %d)" % (M1_RESULTS[t]["label"], M1_RESULTS[t]["df"].task_id.nunique())
               for t in BENCH_TAGS]


def mget(tag, system, col):
    r = M1_METRICS[(M1_METRICS.tag == tag) & (M1_METRICS.system == system)]
    return float(r[col].iloc[0]) if len(r) else float("nan")


def label_bars(ax, bars, fmt="{:.1f}", values=None, inside=False):
    top = ax.get_ylim()[1]
    for i, b in enumerate(bars):
        h = b.get_height() if values is None else values[i]
        if h != h:
            continue
        y_top = b.get_y() + b.get_height()
        if inside and b.get_height() > 0.12 * top:     # at the base: clear of the error bar
            ax.text(b.get_x() + b.get_width() / 2, b.get_y() + 0.025 * top, fmt.format(h),
                    ha="center", va="bottom", fontsize=8, color="white", fontweight="bold")
        else:
            ax.text(b.get_x() + b.get_width() / 2, y_top + 0.015 * top, fmt.format(h),
                    ha="center", va="bottom", fontsize=8, color=INK_2)


def grouped(ax, col, title, fmt="{:.1f}", ylabel=None, ci=False):
    x = np.arange(len(BENCH_TAGS))
    tops = []
    for k, s in enumerate(SYSTEMS):
        off = (k - (N_SYS - 1) / 2) * BAR_W
        vals = [mget(t, s, col) for t in BENCH_TAGS]
        kw = {}
        if ci:
            lo = [v - mget(t, s, "ci_lo") for v, t in zip(vals, BENCH_TAGS)]
            hi = [mget(t, s, "ci_hi") - v for v, t in zip(vals, BENCH_TAGS)]
            kw = dict(yerr=[lo, hi], error_kw=dict(ecolor=INK_2, elinewidth=1, capsize=3))
            tops += [mget(t, s, "ci_hi") for t in BENCH_TAGS]
        bars = ax.bar(x + off, vals, width=BAR_W * 0.92, color=SYS_COLOR[s],
                      hatch=SYS_HATCH[s], edgecolor="white", linewidth=0.6, **kw)
        tops += vals
        bars._m1_vals = vals
    finite = [v for v in tops if v == v]
    ax.set_ylim(0, (max(finite) if finite else 1) * 1.18)
    for c in ax.containers:
        if hasattr(c, "_m1_vals"):
            label_bars(ax, c, fmt, c._m1_vals if ci else None, inside=ci)
    ax.set_xticks(x); ax.set_xticklabels(BENCH_TICKS)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_title(title, color=INK, loc="left")
    ax.grid(axis="x", visible=False)
    return ax



def header(fig, title, subtitle, handles=None, x=0.05, ncol=None, fontsize=None):
    """Title, subtitle and legend stacked top-left, spaced in inches so they never collide."""
    h = fig.get_size_inches()[1]
    fig.suptitle(title, x=x, ha="left", va="bottom", fontsize=12.5, color=INK, y=1 + 0.78 / h)
    fig.text(x, 1 + 0.53 / h, subtitle, ha="left", fontsize=9, color=INK_2)
    if handles:
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(x - 0.006, 1 + 0.45 / h),
                   ncol=ncol or len(handles), fontsize=fontsize or 9)


def save(fig, name):
    fig.savefig(os.path.join(M1_OUTPUT_DIR, name))
    plt.show()

print("Chart style ready. Benchmarks:", BENCH_LABELS, "| systems:", SYSTEMS)
