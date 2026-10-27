"""Explainer figure: how rfsurrogates splits the solver time across contributors.

Top left: the bottleneck (solver time for one family on one desktop vs 10 contributors).
Top right: open models and open data vs closed commercial models.
Middle: the contribution process, steps 1 to 4.
Bottom: the loop (more contributors, more records, better models, more device families).

Text follows ASD-STE100 Simplified Technical English. Numbers come from README.md,
CONTRIBUTING.md and docs/OPEN_PROJECT.md. Style follows the filter-fm figures
(make_fem_vs_fm.py): DejaVu Sans, the same colours, rounded boxes on a pixel canvas.

    python tools/make_process_figure.py   -> docs/img/contribute_process.png / .pdf  (1600 x 1000 px)
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "img")
STEM = "contribute_process"

# ---------------------------------------------------------------- numbers (from the repo docs)
SOLVE_MIN = 11          # one coarse Palace solve on a desktop, mean (CONTRIBUTING.md)
N_FAMILY1 = 261         # Palace solves for family 1 (README.md)
N_PEOPLE = 10           # illustration: 10 contributors share one family
HOURS_ONE = round(N_FAMILY1 * SOLVE_MIN / 60)                    # about 48 h
PER_PERSON = round(N_FAMILY1 / N_PEOPLE)                         # about 26 solves
HOURS_EACH = round(PER_PERSON * SOLVE_MIN / 60)                  # about 5 h

# ---------------------------------------------------------------- style (same as the filter-fm figures)
BLUE = "#2b6cb0"
BLUE_FILL = "#eaf2fb"
ORANGE = "#e07a3a"
ORANGE_FILL = "#fdf0e8"
GREEN = "#2f855a"
GREEN_FILL = "#e9f5ee"
AMBER = "#b7791f"
AMBER_FILL = "#fdf6e3"
RED = "#c53030"
RED_FILL = "#fbecec"
GREY = "#6b7280"
GREY_FILL = "#f3f4f6"
INK = "#1f2937"
BODY = "#4b5563"
RULE = "#d1d5db"
MONO = "DejaVu Sans Mono"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "figure.facecolor": "white",
})

BODY_SIZE = 13
HEAD_SIZE = 17


class Canvas:
    """Pixel-coordinate drawing surface (y grows downward)."""

    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(h, 0)
        self.ax.axis("off")

    def text(self, x, y, s, size=BODY_SIZE, color=BODY, weight="normal", ha="left", va="top", **kw):
        kw.setdefault("linespacing", 1.35)
        return self.ax.text(x, y, s, fontsize=size, color=color, weight=weight, ha=ha, va=va, **kw)

    def rbox(self, x, y, w, h, edge, fill, lw=1.4, r=10):
        p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", linewidth=lw,
                           edgecolor=edge, facecolor=fill)
        self.ax.add_patch(p)
        return p

    def arrow(self, x1, y1, x2, y2, color=GREY, lw=1.6, ms=16):
        self.ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms, linewidth=lw,
                                          color=color, shrinkA=0, shrinkB=0))

    def line(self, x1, y1, x2, y2, color=RULE, lw=1.0):
        self.ax.plot([x1, x2], [y1, y2], color=color, lw=lw, solid_capstyle="butt")

    def width(self, artist):
        """Width of a text artist in canvas pixels (1 canvas px = 1 display px at dpi 100)."""
        return artist.get_window_extent(renderer=self.fig.canvas.get_renderer()).width

    def chip(self, x, y, label, edge, fill, size=12, h=28, pad=10, weight="bold", color=None, r=6, lw=1.4, **kw):
        """Rounded label chip sized to its text. Returns the right edge."""
        t = self.text(x + pad, y + h / 2, label, size=size, color=color or edge, weight=weight,
                      va="center_baseline", zorder=3, **kw)
        w = self.width(t) + 2 * pad
        self.rbox(x, y, w, h, edge, fill, lw=lw, r=r).set_zorder(2)
        return x + w

    def save(self):
        os.makedirs(OUT, exist_ok=True)
        for ext in ("png", "pdf"):
            self.fig.savefig(os.path.join(OUT, f"{STEM}.{ext}"), dpi=100, facecolor="white")
        plt.close(self.fig)


# ---------------------------------------------------------------- blocks
def header(c: Canvas, m, W):
    c.text(m, 28, "rfsurrogates: share the solver time, share the models", size=27, color=INK, weight="bold")
    c.text(m, 80, "A surrogate model needs a few hundred full-wave solves. Contributors share the solves. "
                  "The models and the data are open.", size=14, color=BODY)
    c.line(m, 122, m + W, 122)


def bottleneck(c: Canvas, x, y, w):
    c.text(x, y, "Solver time is the bottleneck", size=HEAD_SIZE, color=INK, weight="bold")
    c.text(x, y + 32, f"One full-wave solve: about {SOLVE_MIN} min on a desktop.   One surrogate query: 4.5 ms.")
    bx0, bx1 = x + 250, x + w - 10          # 0 h ... HOURS_ONE h
    sx = (bx1 - bx0) / HOURS_ONE
    # one desktop
    yr = y + 72
    c.text(x, yr - 2, "One desktop", color=INK, weight="bold")
    c.text(x, yr + 20, f"{N_FAMILY1} solves x {SOLVE_MIN} min", size=12.5)
    c.ax.add_patch(Rectangle((bx0, yr), HOURS_ONE * sx, 38, facecolor=BLUE, edgecolor="none"))
    c.text(bx1 - 14, yr + 19, f"about {HOURS_ONE} h, about 2 days", size=14, color="white", weight="bold",
           ha="right", va="center_baseline")
    # ten contributors
    yr = y + 130
    c.text(x, yr - 2, f"{N_PEOPLE} contributors", color=INK, weight="bold")
    c.text(x, yr + 20, f"about {PER_PERSON} solves each", size=12.5)
    bh, bg = 4.4, 2.4
    for i in range(N_PEOPLE):
        c.ax.add_patch(Rectangle((bx0, yr + i * (bh + bg)), HOURS_EACH * sx, bh, facecolor=ORANGE,
                                 edgecolor="none"))
    c.text(bx0 + HOURS_EACH * sx + 16, yr + 5 * (bh + bg), f"about {HOURS_EACH} h each, at the same time",
           size=14, color=ORANGE, weight="bold", va="center_baseline")
    # time axis
    ya = yr + N_PEOPLE * (bh + bg) + 8
    c.line(bx0, ya, bx1, ya, color=GREY, lw=1.0)
    for hh in range(0, HOURS_ONE + 1, 12):
        c.line(bx0 + hh * sx, ya, bx0 + hh * sx, ya + 5, color=GREY, lw=1.0)
        c.text(bx0 + hh * sx, ya + 8, f"{hh} h", size=11.5, color=GREY, ha="center")
    return ya + 30


def open_panel(c: Canvas, x, y, w):
    c.text(x, y, "Open models, open data", size=HEAD_SIZE, color=INK, weight="bold")
    c.text(x, y + 32, "An open alternative to closed commercial models.")
    c0, c1, c2 = x + 12, x + 190, x + 386
    rh = 30
    yy = y + 62
    c.text(c1, yy + rh / 2, "rfsurrogates", color=ORANGE, weight="bold", va="center_baseline")
    c.text(c2, yy + rh / 2, "Closed models", color=GREY, weight="bold", va="center_baseline")
    c.line(x, yy + rh, x + w, yy + rh, color=INK, lw=1.2)
    yy += rh
    rows = [("Code, weights", "open, Apache-2.0", "closed"),
            ("Data", "open, CC-BY-4.0", "not published"),
            ("Who adds data", "anyone", "the owner")]
    for i, (k, a, b) in enumerate(rows):
        if i % 2 == 0:
            c.ax.add_patch(Rectangle((x, yy), w, rh, facecolor=GREY_FILL, edgecolor="none"))
        c.text(c0, yy + rh / 2, k, color=INK, weight="bold", va="center_baseline")
        c.text(c1, yy + rh / 2, a, va="center_baseline")
        c.text(c2, yy + rh / 2, b, va="center_baseline")
        yy += rh
    c.line(x, yy, x + w, yy, color=RULE, lw=1.0)
    yy += 12
    c.text(x, yy, "Family 1: 261 Palace solves. Held-out passband |S21| error:\n"
                  "0.52 dB for the trained model, 2.86 dB for the textbook prior alone.", size=12.5)
    return yy + 44


def step_head(c: Canvas, x, y, num, title, sub, edge):
    c.rbox(x, y, 34, 34, edge, edge, r=17)
    c.text(x + 17, y + 17, num, size=15, color="white", weight="bold", ha="center", va="center_baseline")
    c.text(x + 46, y - 2, title, size=15, color=edge, weight="bold")
    c.text(x + 46, y + 22, sub, size=12.5, color=GREY)


def lines(c: Canvas, x, y, items, size=BODY_SIZE, step=22, color=BODY):
    for s in items:
        c.text(x, y, s, size=size, color=color)
        y += step * (s.count("\n") + 1)
    return y


def gate(c: Canvas, x, y, label):
    return c.chip(x, y, label, AMBER, AMBER_FILL, size=12.5, h=28)


def process(c: Canvas, m, y, W, h):
    w1, w2, w3, w4 = 568, 272, 304, 276
    g = (W - w1 - w2 - w3 - w4) / 3
    x1 = m
    x2 = x1 + w1 + g
    x3 = x2 + w2 + g
    x4 = x3 + w3 + g
    for x, w, edge, fill in ((x1, w1, BLUE, BLUE_FILL), (x2, w2, GREY, GREY_FILL), (x3, w3, GREEN, GREEN_FILL),
                             (x4, w4, ORANGE, ORANGE_FILL)):
        c.rbox(x, y, w, h, edge, fill)
    for xa, xb in ((x1 + w1, x2), (x2 + w2, x3), (x3 + w3, x4)):
        c.arrow(xa + 3, y + 35, xb - 3, y + 35, color=GREY, lw=2.0, ms=18)
    p = 18
    # 1 contributor ------------------------------------------------------
    step_head(c, x1 + p, y + p, "1", "Contributor solves designs", "on their own machine, to their own budget", BLUE)
    yy = y + p + 50
    c.rbox(x1 + p, yy, w1 - 2 * p, 54, INK, INK, r=6)
    c.text(x1 + p + 12, yy + 8, "python tools/contribute.py go \\\n    --family edge_coupled_bpf --n 20",
           size=12, color="white", family=MONO, linespacing=1.3)
    yy += 66
    lx = x1 + p + 128
    c.text(x1 + p, yy, "First run", color=INK, weight="bold")
    yy = lines(c, lx, yy, ["Draws designs from the parameter box.", "Opens a claim issue on GitHub."])
    yy += 2
    gate(c, lx, yy, "Claim gate: the maintainer accepts")
    yy += 42
    c.text(x1 + p, yy, "Second run", color=INK, weight="bold")
    yy = lines(c, lx, yy, ["Builds each case."])
    yy += 2
    gate(c, lx, yy, "Geometry gate: rejects a wrong layout")
    yy += 38
    yy = lines(c, lx, yy, ["Solves each case on this machine.", "Checks each record.", "Opens one pull request."])
    yy += 6
    c.line(x1 + p, yy, x1 + w1 - p, yy, color=RULE)
    yy += 12
    c.text(x1 + p, yy, "Solver", color=INK, weight="bold")
    lines(c, lx, yy, ["Palace: runs automatically.",
                      "HFSS, CST or openEMS: the tool exports\n"
                      "the geometry. You solve it. The tool\n"
                      "reads the Touchstone file back."])
    # 2 CI ----------------------------------------------------------------
    step_head(c, x2 + p, y + p, "2", "CI verifier", "on the pull request", GREY)
    yy = lines(c, x2 + p, y + p + 58, ["GitHub Actions runs", "the checker again", "on every record."])
    yy += 12
    for lab, e, f in (("PASS", GREEN, GREEN_FILL), ("NEEDS REVIEW", AMBER, AMBER_FILL), ("FAIL", RED, RED_FILL)):
        c.chip(x2 + p, yy, lab, e, f, size=13, h=32, pad=12)
        yy += 44
    # 3 merge -------------------------------------------------------------
    step_head(c, x3 + p, y + p, "3", "Maintainer merges", "records are pooled", GREEN)
    yy = lines(c, x3 + p, y + p + 58, ["The records go to the", "Hugging Face dataset"])
    yy += 4
    c.chip(x3 + p, yy, "rfsurrogates/samples", GREEN, "white", size=12, h=30, weight="normal", color=INK,
           family=MONO)
    yy += 42
    lines(c, x3 + p, yy, ["and to the data index.", "Each record names", "its contributor."])
    # 4 retrain -----------------------------------------------------------
    step_head(c, x4 + p, y + p, "4", "Retrain, publish", "open weights", ORANGE)
    yy = lines(c, x4 + p, y + p + 58, ["Retrain when a family", "has about 25 % more", "records."])
    yy += 8
    yy = lines(c, x4 + p, yy, ["New weights go to", "Hugging Face."])
    yy += 12
    c.text(x4 + p, yy, "Anyone can use them:", color=INK, weight="bold")
    yy += 32
    for lab in ("browser demo", "inverse design", "download"):
        c.chip(x4 + p, yy, lab, ORANGE, "white", size=13, h=32, weight="normal", color=INK)
        yy += 42
    return (x1, w1), (x4, w4)


def loop(c: Canvas, y_top, s1, s4):
    """Return path from step 4 to step 1, with the growth cycle written on it."""
    (x1, w1), (x4, w4) = s1, s4
    ya = y_top + 36
    xr = x4 + w4 / 2
    xl = x1 + 70
    c.line(xr, y_top, xr, ya, color=ORANGE, lw=2.2)
    c.line(xr, ya, xl, ya, color=ORANGE, lw=2.2)
    c.arrow(xl, ya, xl, y_top + 2, color=ORANGE, lw=2.2, ms=18)
    labels = ["more contributors", "more records", "better models", "more device families"]
    gap = 46
    # measure first, then place the chain at the centre of the return line
    probe = [c.text(0, 0, s, size=14, weight="bold") for s in labels]
    widths = [c.width(t) + 32 for t in probe]
    for t in probe:
        t.remove()
    total = sum(widths) + gap * (len(labels) - 1)
    x = (xl + xr) / 2 - total / 2
    for i, (s, wd) in enumerate(zip(labels, widths)):
        c.chip(x, ya - 18, s, ORANGE, "white", size=14, h=36, pad=16, color=INK, r=18, lw=1.8)
        if i < len(labels) - 1:
            c.arrow(x + wd + 6, ya, x + wd + gap - 6, ya, color=ORANGE, lw=2.2, ms=16)
        x += wd + gap


def footer(c: Canvas, m, W, H):
    c.line(m, H - 44, m + W, H - 44)
    c.text(m, H - 32, "Simulation only. Solve times: Palace, coarse mesh, one desktop. "
                      "Code and weights Apache-2.0, data CC-BY-4.0.", size=11.5, color=GREY)
    c.text(m + W, H - 32, "Ben Hardy · rfsurrogates", size=11.5, color=GREY, ha="right")


def main():
    Wpx, Hpx = 1600, 1000
    c = Canvas(Wpx, Hpx)
    m, W = 48, Wpx - 2 * 48
    header(c, m, W)
    y0 = 140
    yb = bottleneck(c, m, y0, 820)
    yo = open_panel(c, m + 900, y0, W - 900)
    y = max(yb, yo) + 12
    c.line(m, y, m + W, y)
    y += 16
    h = Hpx - 44 - 66 - y
    s1, s4 = process(c, m, y, W, h)
    loop(c, y + h, s1, s4)
    footer(c, m, W, Hpx)
    c.save()


if __name__ == "__main__":
    main()
