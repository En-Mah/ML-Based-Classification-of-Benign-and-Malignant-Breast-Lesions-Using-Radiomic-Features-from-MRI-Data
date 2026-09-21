from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D


NAVY = "#163A63"
BLUE = "#2C5F92"
GREEN = "#2E7D4F"
ORANGE = "#D95F02"
LIGHT_BLUE = "#F6F9FC"
LIGHT_GREEN = "#F5FAF6"
LIGHT_ORANGE = "#FFF8F3"
TEXT = "#17202A"
MUTED = "#5A6772"
WHITE = "#FFFFFF"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def rounded_box(ax, x, y, w, h, text="", edge=NAVY, face=WHITE,
                lw=1.0, fontsize=8.5, weight="normal", text_color=TEXT,
                linestyle="-", radius=0.010):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        linewidth=lw, edgecolor=edge, facecolor=face, linestyle=linestyle,
    )
    ax.add_patch(patch)
    if text:
        ax.text(x + w/2, y + h/2, text,
                ha="center", va="center", fontsize=fontsize,
                fontweight=weight, color=text_color, linespacing=1.16)
    return patch


def arrow(ax, x1, y1, x2, y2, color=NAVY, lw=1.1,
          mutation_scale=10, linestyle="-"):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>",
        mutation_scale=mutation_scale, linewidth=lw,
        color=color, linestyle=linestyle, shrinkA=0, shrinkB=0,
    )
    ax.add_patch(arr)
    return arr


def panel_label(ax, x, y, label):
    ax.text(x, y, label, ha="left", va="bottom",
            fontsize=12, fontweight="bold", color=NAVY)


def panel_title(ax, x, y, w, title, subtitle=None):
    ax.text(x + w/2, y, title, ha="center", va="top",
            fontsize=10.2, fontweight="bold", color=TEXT, linespacing=1.14)
    if subtitle:
        ax.text(x + w/2, y - 0.038, subtitle,
                ha="center", va="top", fontsize=8.0,
                color=BLUE, fontweight="bold")


def small_person(ax, cx, cy, scale=1.0, color=ORANGE):
    head = plt.Circle((cx, cy + 0.012*scale), 0.006*scale, color=color)
    ax.add_patch(head)
    ax.plot([cx, cx], [cy + 0.005*scale, cy - 0.010*scale],
            color=color, lw=2.0*scale, solid_capstyle="round")



fig, ax = plt.subplots(figsize=(16, 8.7))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

panel_y = 0.08
panel_h = 0.80
x_a, w_a = 0.025, 0.115
x_b, w_b = 0.160, 0.135
x_c, w_c = 0.315, 0.270
x_d, w_d = 0.605, 0.160
x_e, w_e = 0.785, 0.190

for x, w in [(x_a,w_a),(x_b,w_b),(x_c,w_c),(x_d,w_d),(x_e,w_e)]:
    rounded_box(ax, x, panel_y, w, panel_h, edge=NAVY, face=WHITE, lw=1.1)

for x, lab in [(x_a,"(a)"),(x_b,"(b)"),(x_c,"(c)"),(x_d,"(d)"),(x_e,"(e)")]:
    panel_label(ax, x+0.004, panel_y+panel_h+0.025, lab)

mid_y = panel_y + panel_h/2
for x1, x2 in [
    (x_a+w_a+0.006, x_b-0.008),
    (x_b+w_b+0.006, x_c-0.008),
    (x_c+w_c+0.006, x_d-0.008),
    (x_d+w_d+0.006, x_e-0.008),
]:
    arrow(ax, x1, mid_y, x2, mid_y, lw=1.5, mutation_scale=14)

# -----------------------------------------------------------------------------
# (a) Inputs
# -----------------------------------------------------------------------------
panel_title(ax, x_a, panel_y+panel_h-0.025, w_a, "Input feature\nsources")
items_a = ["ADC", "Pre", "Post1", "Post2", "T2"]
y0 = panel_y + panel_h - 0.17
box_h = 0.070
step = 0.115
for i, name in enumerate(items_a):
    rounded_box(ax, x_a+0.020, y0-i*step, w_a-0.040, box_h,
                text=name, edge=BLUE, face=LIGHT_BLUE,
                lw=0.9, fontsize=9.4, weight="bold")

# -----------------------------------------------------------------------------
# (b) Combinations
# -----------------------------------------------------------------------------
panel_title(ax, x_b, panel_y+panel_h-0.025, w_b, "31 dataset\ncombinations")
items_b = [
    "Single-source\n(n=5)",
    "Pairwise\n(n=10)",
    "Triple-source\n(n=10)",
    "Quadruple-source\n(n=5)",
    "All-five\n(n=1)",
]
for i, label in enumerate(items_b):
    rounded_box(ax, x_b+0.015, y0-i*step, w_b-0.030, box_h,
                text=label, edge=BLUE, face=LIGHT_BLUE,
                lw=0.9, fontsize=8.1)

# -----------------------------------------------------------------------------
# (c) Patient-aware nested CV
# -----------------------------------------------------------------------------
panel_title(ax, x_c, panel_y+panel_h-0.020, w_c,
            "Patient-aware nested\ncross-validation")

outer_x = x_c + 0.020
outer_y = panel_y + 0.455
outer_w = w_c - 0.040
outer_h = 0.235
rounded_box(ax, outer_x, outer_y, outer_w, outer_h,
            edge=BLUE, face=WHITE, lw=1.0)
ax.text(outer_x+outer_w/2, outer_y+outer_h-0.030,
        "Outer CV (5 folds)", ha="center", va="center",
        fontsize=9.0, fontweight="bold", color=BLUE)

fold_gap = 0.006
fold_w = (outer_w - 0.030 - 4*fold_gap)/5
fold_y = outer_y + 0.090
fold_h = 0.062
fold_xs = []
for i in range(5):
    fx = outer_x + 0.015 + i*(fold_w+fold_gap)
    fold_xs.append(fx)
    is_test = (i == 4)
    rounded_box(ax, fx, fold_y, fold_w, fold_h,
                text=f"Fold {i+1}" + ("\n(TEST)" if is_test else ""),
                edge=ORANGE if is_test else NAVY,
                face=LIGHT_ORANGE if is_test else WHITE,
                lw=1.0, fontsize=7.3,
                text_color=ORANGE if is_test else TEXT,
                linestyle="--" if is_test else "-", radius=0.005)

train_left = fold_xs[0]
train_right = fold_xs[3] + fold_w
br_y = fold_y - 0.018
ax.plot([train_left, train_right], [br_y, br_y], color=BLUE, lw=1.0)
ax.plot([train_left, train_left], [br_y-0.008, br_y+0.008], color=BLUE, lw=1.0)
ax.plot([train_right, train_right], [br_y-0.008, br_y+0.008], color=BLUE, lw=1.0)
ax.text((train_left+train_right)/2, br_y-0.025,
        "Outer-training (4 folds)", ha="center", va="center",
        fontsize=7.4, color=BLUE)

inner_x = outer_x
inner_y = panel_y + 0.115
inner_w = outer_w * 0.62
inner_h = 0.285
rounded_box(ax, inner_x, inner_y, inner_w, inner_h,
            edge=GREEN, face=LIGHT_GREEN, lw=1.0, linestyle="--")
ax.text(inner_x+inner_w/2, inner_y+inner_h-0.040,
        "Inner CV (3 folds)\n(on outer-training only)",
        ha="center", va="center", fontsize=8.5,
        fontweight="bold", color=GREEN, linespacing=1.15)

igap = 0.010
iw = (inner_w - 0.040 - 2*igap)/3
iy = inner_y + 0.125
ih = 0.055
for i in range(3):
    rounded_box(ax, inner_x+0.020+i*(iw+igap), iy, iw, ih,
                text=f"Fold {i+1}", edge=GREEN, face=WHITE,
                lw=0.9, fontsize=7.1, radius=0.005)

ax.text(inner_x+inner_w*0.34, inner_y+0.080,
        "Inner-training\n(2 folds)", ha="center", va="center",
        fontsize=7.0, color=MUTED)
ax.text(inner_x+inner_w*0.78, inner_y+0.080,
        "Inner-val\n(1 fold)", ha="center", va="center",
        fontsize=7.0, color=MUTED)
ax.text(inner_x+inner_w/2, inner_y+0.025,
        "Model selection and tuning", ha="center", va="center",
        fontsize=7.5, color=GREEN, fontweight="bold")

# Test branch
test_x = inner_x + inner_w + 0.020
test_y = inner_y + 0.030
test_w = outer_w - inner_w - 0.020
test_h = inner_h - 0.030
rounded_box(ax, test_x, test_y, test_w, test_h,
            edge=ORANGE, face=LIGHT_ORANGE, lw=1.0, linestyle="--")
ax.text(test_x+test_w/2, test_y+test_h-0.042,
        "Outer-test\n(held-out fold)", ha="center", va="center",
        fontsize=8.1, color=ORANGE, fontweight="bold")
for j in [-1,0,1]:
    small_person(ax, test_x+test_w/2+j*0.018, test_y+0.120, scale=0.8)
ax.text(test_x+test_w/2, test_y+0.055,
        "No fitting or\noptimisation", ha="center", va="center",
        fontsize=7.6, color=ORANGE, fontweight="bold")

# Training/test paths
arrow(ax, outer_x+outer_w*0.36, outer_y,
      inner_x+inner_w*0.45, inner_y+inner_h,
      color=GREEN, lw=1.0, mutation_scale=9)
arrow(ax, fold_xs[4]+fold_w/2, fold_y,
      test_x+test_w/2, test_y+test_h,
      color=ORANGE, lw=1.0, mutation_scale=9, linestyle="--")

# Legend
legend_y1 = panel_y + 0.055
legend_y2 = panel_y + 0.028
ax.add_line(Line2D([x_c+0.035, x_c+0.075], [legend_y1,legend_y1], color=GREEN, lw=1.4))
ax.text(x_c+0.082, legend_y1, "Training: fitting/optimisation allowed",
        ha="left", va="center", fontsize=6.7, color=MUTED)
ax.add_line(Line2D([x_c+0.035, x_c+0.075], [legend_y2,legend_y2], color=ORANGE, lw=1.4, linestyle="--"))
ax.text(x_c+0.082, legend_y2, "Test: no fitting or optimisation",
        ha="left", va="center", fontsize=6.7, color=MUTED)

# -----------------------------------------------------------------------------
# (d) Training pipeline
# -----------------------------------------------------------------------------
panel_title(ax, x_d, panel_y+panel_h-0.025, w_d,
            "Training pipeline", subtitle="training partition only")
steps = [
    "Median imputation",
    "Variance filtering",
    "Feature selection",
    "Auto-k",
    "Standardisation",
    "Optional SMOTE",
    "Optuna hyperparameter\noptimisation",
]
box_x = x_d + 0.018
box_w = w_d - 0.036
box_h_d = 0.064
start_y = panel_y + panel_h - 0.175
step_d = 0.090
for i, label in enumerate(steps):
    y = start_y - i*step_d
    rounded_box(ax, box_x, y, box_w, box_h_d,
                text=label, edge=BLUE, face=LIGHT_BLUE,
                lw=0.9, fontsize=7.7 if i < 6 else 7.1)
    if i < len(steps)-1:
        arrow(ax, box_x+box_w/2, y,
              box_x+box_w/2, y-(step_d-box_h_d)+0.003,
              color=NAVY, lw=0.9, mutation_scale=8)

# -----------------------------------------------------------------------------
# (e) Evaluation/output
# -----------------------------------------------------------------------------
panel_title(ax, x_e, panel_y+panel_h-0.025, w_e,
            "Evaluation and\noutput collection")

eval_x = x_e + 0.025
eval_w = w_e - 0.050
eval_start = panel_y + panel_h - 0.185

rounded_box(ax, eval_x, eval_start, eval_w, 0.105,
            text="Outer-test patients\n(held-out fold)",
            edge=ORANGE, face=LIGHT_ORANGE, lw=1.0,
            fontsize=8.1, weight="bold", text_color=ORANGE)

flow = [
    ("Trained model", 0.060),
    ("Out-of-fold\npredictions", 0.078),
]
prev_bottom = eval_start
curr_y = eval_start - 0.090
for label, h in flow:
    rounded_box(ax, eval_x, curr_y, eval_w, h,
                text=label, edge=BLUE, face=LIGHT_BLUE,
                lw=0.9, fontsize=8.2, weight="bold", text_color=NAVY)
    arrow(ax, eval_x+eval_w/2, prev_bottom,
          eval_x+eval_w/2, curr_y+h,
          color=NAVY, lw=0.9, mutation_scale=8)
    prev_bottom = curr_y
    curr_y -= h + 0.055

metrics_y = panel_y + 0.115
metrics_h = 0.260
rounded_box(ax, eval_x, metrics_y, eval_w, metrics_h,
            edge=BLUE, face=WHITE, lw=0.9)
ax.text(eval_x+eval_w/2, metrics_y+metrics_h-0.040,
        "Performance assessment", ha="center", va="center",
        fontsize=8.2, color=NAVY, fontweight="bold")
metrics = ["ROC-AUC", "PR-AUC", "Accuracy", "Sensitivity", "Specificity"]
for i, m in enumerate(metrics):
    ax.text(eval_x+eval_w/2, metrics_y+metrics_h-0.090-i*0.034,
            m, ha="center", va="center", fontsize=7.6, color=TEXT)
arrow(ax, eval_x+eval_w/2, prev_bottom,
      eval_x+eval_w/2, metrics_y+metrics_h,
      color=NAVY, lw=0.9, mutation_scale=8)


outdir = Path("figures")
outdir.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png", "svg"):
    p = outdir / f"computational_workflow.{ext}"
    fig.savefig(p, dpi=600 if ext == "png" else None,
                bbox_inches="tight", facecolor="white")
    print(f"Saved: {p}")
plt.close(fig)
