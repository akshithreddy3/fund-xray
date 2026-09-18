"""Chart theming shared by every Plotly figure in app.py.

Colors are the validated default palette (see the dataviz skill /
`references/palette.md`): an 8-slot categorical order with adjacent
CVD Delta-E >= 8 and normal-vision Delta-E >= 15 in light mode, a
single-hue blue sequential ramp, and a fixed status palette (good /
warning / critical) reserved for state, never reused as a plain
series color.

Colors are assigned **by entity identity, in a fixed order, everywhere
in the app** -- Mkt-RF is always the same blue whether it appears in
the static-exposure bar chart, the time-varying overlay, or the
regime table, and "stressed" is always the same red. A filter that
changes which series are shown must never repaint the survivors with
different colors, so colors are looked up by name, never assigned by
list position at render time.
"""

from __future__ import annotations

# Categorical palette, slots 1-8, fixed order (dataviz skill default).
CATEGORICAL = {
    1: "#2a78d6",  # blue
    2: "#eb6834",  # orange
    3: "#1baf7a",  # aqua
    4: "#eda100",  # yellow
    5: "#e87ba4",  # magenta
    6: "#008300",  # green
    7: "#4a3aa7",  # violet
    8: "#e34948",  # red
}

# One fixed color per factor, by identity, reused across every tab.
FACTOR_COLORS = {
    "const": CATEGORICAL[7],
    "Mkt-RF": CATEGORICAL[1],
    "SMB": CATEGORICAL[2],
    "HML": CATEGORICAL[3],
    "RMW": CATEGORICAL[4],
    "CMA": CATEGORICAL[5],
    "Mom": CATEGORICAL[6],
}

# Two-series comparisons (constrained vs. unconstrained; rolling vs.
# Kalman) are colored by ESTIMATOR identity, not by factor, since that's
# the entity varying across the two series in those charts.
ESTIMATOR_COLORS = {
    "constrained": CATEGORICAL[1],
    "unconstrained": CATEGORICAL[2],
    "rolling": CATEGORICAL[1],
    "kalman": CATEGORICAL[2],
}

# Status palette: reserved for regime STATE, never reused as a plain
# series color. calm/elevated/stressed is a genuine state indicator,
# not an arbitrary category, so this is exactly what these colors are for.
REGIME_COLORS = {
    "calm": "#0ca30c",  # good
    "elevated": "#fab219",  # warning
    "stressed": "#d03b3b",  # critical
}

SEQUENTIAL_BLUE = "#2a78d6"

# Light-mode chart chrome (dataviz skill default surfaces/ink).
CHART_SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

PLOTLY_LAYOUT_DEFAULTS = dict(
    plot_bgcolor=CHART_SURFACE,
    paper_bgcolor=CHART_SURFACE,
    font=dict(color=PRIMARY_INK, family="system-ui, -apple-system, 'Segoe UI', sans-serif"),
    # Below the plot, not above: a legend placed above (a common Plotly
    # recipe) collides with the chart title unless the title is removed,
    # which we don't want to give up.
    legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0),
    margin=dict(l=10, r=10, t=40, b=10),
    hovermode="x unified",
)

AXIS_DEFAULTS = dict(
    gridcolor=GRIDLINE,
    linecolor=BASELINE,
    tickfont=dict(color=MUTED_INK),
)


def apply_theme(fig):
    """Apply shared layout/axis defaults to a Plotly figure in place."""
    fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
    fig.update_xaxes(**AXIS_DEFAULTS)
    fig.update_yaxes(**AXIS_DEFAULTS)
    return fig
