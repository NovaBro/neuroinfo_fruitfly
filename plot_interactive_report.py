"""Self-contained interactive report (plotly) for the pruning study, built
from build_figure_data.py's tables.

Interactivity earns its place in three ways the static figures cannot match:
hovering a point gives the exact n / best k / all four metrics; the k-curve
explorer lets the reader check that the conclusion holds at every k rather
than trusting one; and the 3D skeleton viewer shows how pruning changes the
arbor in depth, which the 2D projection necessarily hides.

Writes figures/results_report.html with plotly.js embedded, so it opens
offline and can be shared as a single file.
"""

import functools
from pathlib import Path

print = functools.partial(print, flush=True)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

RESULTS_DIR = Path.cwd() / "data" / "MANC" / "results"
DATA_DIR = RESULTS_DIR / "figure_data"
FIG_DIR = RESULTS_DIR / "figures"

METHODS = ["NBLAST", "Geodesic GW", "Euclidean GW", "UGW"]
SEVERITY_ORDER = ["none", "mild", "moderate", "tertiary", "aggressive"]
# Match plot_final_figures.py: colorblind-safe per method, sequential per severity.
METHOD_COLORS = dict(zip(METHODS, px.colors.qualitative.Safe))
SEVERITY_COLORS = dict(zip(SEVERITY_ORDER, px.colors.sequential.Blues[3:8]))
MAX_NODES_3D = 3000


def best_k(tidy):
    idx = tidy.groupby(["definition", "severity", "method"])["mcc"].idxmax()
    return tidy.loc[idx].copy()


def fig_headline(tidy):
    peak = best_k(tidy)
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                        subplot_titles=("Strahler-order pruning", "Centrifugal-order pruning"))
    for col, definition in enumerate(["strahler", "centrifugal"], start=1):
        sub = peak[peak["definition"].isin([definition, "baseline"])]
        for method in METHODS:
            rows = sub[sub["method"] == method]
            if rows.empty:
                continue
            rows = rows.set_index("severity").reindex(
                [s for s in SEVERITY_ORDER if s in set(rows["severity"])]).reset_index()
            fig.add_trace(go.Scatter(
                x=rows["severity"], y=rows["mcc"], name=method, legendgroup=method,
                showlegend=(col == 1), mode="lines+markers",
                line=dict(color=METHOD_COLORS[method], width=3),
                marker=dict(size=10),
                customdata=np.stack([rows["k"], rows["accuracy"],
                                     rows["balanced_accuracy"], rows["macro_f1"]], axis=-1),
                hovertemplate=("<b>%{fullData.name}</b><br>severity=%{x}<br>"
                               "MCC=%{y:.3f} at k=%{customdata[0]}<br>"
                               "accuracy=%{customdata[1]:.3f}<br>"
                               "balanced acc=%{customdata[2]:.3f}<br>"
                               "macro F1=%{customdata[3]:.3f}<extra></extra>"),
            ), row=1, col=col)
    fig.update_yaxes(title_text="MCC at best k", row=1, col=1)
    fig.update_layout(height=520, template="plotly_white",
                      title="Classification performance vs pruning severity")
    return fig


def fig_k_explorer(tidy):
    """Dropdown by method; one line per severity. Makes 'k=10 was the worst
    end of the curve' something the reader can check rather than take on faith."""
    fig = go.Figure()
    trace_method = []
    for method in METHODS:
        for definition in ["baseline", "strahler"]:
            sub = tidy[(tidy["method"] == method) & (tidy["definition"] == definition)]
            for severity in SEVERITY_ORDER:
                rows = sub[sub["severity"] == severity].sort_values("k")
                if rows.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=rows["k"], y=rows["mcc"], name=severity, mode="lines+markers",
                    line=dict(color=SEVERITY_COLORS[severity], width=2.5),
                    visible=(method == METHODS[0]),
                    hovertemplate=f"{method} / {severity}<br>k=%{{x}}<br>MCC=%{{y:.3f}}<extra></extra>",
                ))
                trace_method.append(method)

    buttons = [dict(label=m, method="update",
                    args=[{"visible": [tm == m for tm in trace_method]},
                          {"title": f"MCC vs k - {m}"}])
               for m in METHODS]
    fig.update_layout(
        updatemenus=[dict(buttons=buttons, direction="down", x=0, xanchor="left",
                          y=1.15, yanchor="top")],
        height=520, template="plotly_white",
        title=f"MCC vs k - {METHODS[0]}",
        xaxis_title="k (KNN neighbours)", yaxis_title="MCC",
    )
    return fig


def fig_skeletons_3d(skel):
    """One trace per pruning level, dropdown over exemplar neurons. Levels are
    legend-toggleable, which makes the nesting obvious in a way a static
    overlay cannot."""
    body_ids = sorted(skel["body_id"].unique())
    conditions = ["none"] + [f"strahler_{s}" for s in SEVERITY_ORDER[1:]]

    fig = go.Figure()
    trace_body = []
    for body_id in body_ids:
        for severity, condition in zip(SEVERITY_ORDER, conditions):
            rows = skel[(skel["body_id"] == body_id) & (skel["condition"] == condition)]
            if rows.empty:
                continue
            if len(rows) > MAX_NODES_3D:
                rows = rows.sample(MAX_NODES_3D, random_state=0)
            fig.add_trace(go.Scatter3d(
                x=rows["x"], y=rows["y"], z=rows["z"], mode="markers",
                name=severity, marker=dict(size=1.5, color=SEVERITY_COLORS[severity]),
                visible=(body_id == body_ids[0]),
                hovertemplate=f"{body_id} / {severity}<extra></extra>",
            ))
            trace_body.append(body_id)

    buttons = [dict(label=str(b), method="update",
                    args=[{"visible": [tb == b for tb in trace_body]},
                          {"title": f"Pruning levels in 3D - neuron {b}"}])
               for b in body_ids]
    fig.update_layout(
        updatemenus=[dict(buttons=buttons, direction="down", x=0, xanchor="left",
                          y=1.1, yanchor="top")],
        height=700, template="plotly_white",
        title=f"Pruning levels in 3D - neuron {body_ids[0]}",
        scene=dict(aspectmode="data"),
    )
    return fig


def summary_table(tidy):
    peak = best_k(tidy)
    cols = ["definition", "severity", "method", "k", "mcc",
            "accuracy", "balanced_accuracy", "macro_f1"]
    table = peak[cols].sort_values(["definition", "severity", "method"])
    return table.to_html(index=False, float_format=lambda v: f"{v:.3f}",
                         border=0, classes="summary")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    tidy = pd.read_csv(DATA_DIR / "performance_tidy.csv")
    skel = pd.read_csv(DATA_DIR / "skeleton_examples.csv.gz")

    blocks = []
    for i, (title, fig) in enumerate([
        ("Performance vs pruning severity", fig_headline(tidy)),
        ("How the choice of k affects every method", fig_k_explorer(tidy)),
        ("What Strahler pruning removes, in 3D", fig_skeletons_3d(skel)),
    ]):
        # plotly.js inlined once in the first block keeps the file offline-usable.
        blocks.append(f"<h2>{title}</h2>" + fig.to_html(
            full_html=False, include_plotlyjs=("inline" if i == 0 else False)))

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>MANC pruning study - results</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 1200px; color: #222; }}
 h1 {{ margin-bottom: .2rem; }} h2 {{ margin-top: 2.5rem; }}
 .lede {{ color: #555; }}
 table.summary {{ border-collapse: collapse; font-size: 14px; }}
 table.summary th, table.summary td {{ padding: 4px 10px; border-bottom: 1px solid #eee; text-align: right; }}
 table.summary th {{ text-align: right; border-bottom: 2px solid #ccc; }}
</style></head><body>
<h1>MANC hemilineage classification: the effect of branch pruning</h1>
<p class="lede">NBLAST outperforms all GW variants to begin with, and Strahler-order
pruning widens that gap further - it raises NBLAST while collapsing Geodesic GW and
UGW. Centrifugal-order pruning does not: it lowers NBLAST instead, so the benefit is
specific to removing peripheral twigs rather than to pruning in general. Every point
is reported at that method's own best k.</p>
{''.join(blocks)}
<h2>Summary (each method at its best k)</h2>
{summary_table(tidy)}
</body></html>"""

    out = FIG_DIR / "results_report.html"
    out.write_text(html)
    print(f"Saved {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
