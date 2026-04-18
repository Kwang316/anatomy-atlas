"""
viewer.py — Standalone Anatomy Atlas Viewer (no 3D Slicer required).

Loads the synthetic phantom segmentation and renders an interactive axial
slice viewer with organ info panel, system filters, and quiz mode.

Usage:
    python viewer.py [--phantom ./phantom]
    # Open http://localhost:8050
"""
import argparse
import json
import random
import numpy as np
import SimpleITK as sitk
import plotly.graph_objects as go
from pathlib import Path
from dash import Dash, dcc, html, Input, Output, State, ctx, ALL

_HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_atlas(phantom_dir: Path):
    img     = sitk.ReadImage(str(phantom_dir / "segmentation.nii.gz"))
    volume  = sitk.GetArrayFromImage(img)   # shape: (Z, Y, X)
    spacing = img.GetSpacing()              # (sx, sy, sz) mm
    vox_ml  = spacing[0] * spacing[1] * spacing[2] / 1000.0

    with open(_HERE / "AnatomyAtlas" / "Resources" / "anatomy_data.json") as f:
        db = json.load(f)
    organs  = db["organs"]
    systems = db["systems"]

    # Pre-compute phantom volumes from the label map
    organ_volumes = {}
    for key, data in organs.items():
        voxels = int((volume == data["label"]).sum())
        organ_volumes[key] = round(voxels * vox_ml, 0)

    return volume, organs, systems, organ_volumes


def label_to_rgb_map(organs: dict) -> dict:
    """Return {label_id: [r, g, b]} 0-255, plus label 0 = background."""
    m = {0: [15, 23, 42]}
    for data in organs.values():
        c = data["color"]
        m[data["label"]] = [int(c[0]*255), int(c[1]*255), int(c[2]*255)]
    return m


def make_slice_image(volume, z, l2rgb, highlight_label=None, visible_labels=None):
    """Return an RGB array (H, W, 3) for axial slice z."""
    sl  = volume[z]                                    # (Y, X)
    rgb = np.full((*sl.shape, 3), fill_value=0, dtype=np.uint8)
    rgb[:, :] = l2rgb[0]                              # background

    for label_id, color in l2rgb.items():
        if label_id == 0:
            continue
        if visible_labels is not None and label_id not in visible_labels:
            continue
        mask = sl == label_id
        if not mask.any():
            continue
        if highlight_label is not None and label_id != highlight_label:
            rgb[mask] = [max(0, c // 6) for c in color]
        else:
            rgb[mask] = color

    return rgb


def make_slice_fig(volume, z, l2rgb, highlight_label=None, visible_labels=None,
                   title_suffix=""):
    rgb = make_slice_image(volume, z, l2rgb, highlight_label, visible_labels)
    fig = go.Figure(go.Image(z=rgb))
    fig.update_layout(
        title=dict(
            text=f"Axial slice {z}{title_suffix}",
            font=dict(color="#94a3b8", size=11),
            x=0.01, xanchor="left",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="#1e293b",
        plot_bgcolor="#1e293b",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                   scaleanchor="x"),
    )
    return fig


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

BG     = "#0f172a"
CARD   = "#1e293b"
BORDER = "#334155"
TEXT   = "#cbd5e1"
BLUE   = "#0ea5e9"
MUTED  = "#64748b"
PURPLE = "#7c3aed"
GREEN  = "#22c55e"
RED    = "#ef4444"


def _label(text, **style):
    base = {"color": MUTED, "fontSize": "9px", "letterSpacing": "2px",
            "margin": "0 0 4px", "textTransform": "uppercase"}
    base.update(style)
    return html.P(text, style=base)


def _card(**style):
    base = {"backgroundColor": CARD, "border": f"1px solid {BORDER}",
            "borderRadius": "12px", "padding": "16px"}
    base.update(style)
    return base


def organ_list_items(organs, systems, system_filter="all", selected_key=None):
    items = []
    for key, data in organs.items():
        if system_filter != "all" and data.get("system") != system_filter:
            continue
        c   = data["color"]
        bg  = f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})"
        dot = html.Span(style={
            "display": "inline-block", "width": "10px", "height": "10px",
            "borderRadius": "50%", "backgroundColor": bg,
            "marginRight": "8px", "flexShrink": "0",
        })
        is_sel = key == selected_key
        items.append(html.Div(
            id={"type": "organ-row", "key": key},
            n_clicks=0,
            children=[dot, html.Span(data["display_name"],
                                     style={"fontSize": "13px"})],
            style={
                "display": "flex", "alignItems": "center",
                "padding": "6px 10px", "borderRadius": "6px",
                "cursor": "pointer",
                "color": "white" if is_sel else TEXT,
                "backgroundColor": f"{BLUE}30" if is_sel else "transparent",
            },
        ))
    return items


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

def build_app(volume, organs, systems, organ_volumes):
    l2rgb  = label_to_rgb_map(organs)
    n_slices = volume.shape[0]
    mid_z    = n_slices // 2

    key_to_label = {k: v["label"] for k, v in organs.items()}

    app = Dash(__name__, title="Anatomy Atlas")

    app.layout = html.Div(
        style={"backgroundColor": BG, "minHeight": "100vh",
               "fontFamily": "Inter, system-ui, sans-serif",
               "color": TEXT, "padding": "24px 32px",
               "boxSizing": "border-box"},
        children=[
            # Hidden stores for app state
            dcc.Store(id="selected-organ", data=None),
            dcc.Store(id="active-system",  data="all"),
            dcc.Store(id="quiz-state",     data={
                "active": False, "organ": None,
                "correct": 0, "total": 0,
            }),

            # Header
            html.Div([
                html.P("MEDICAL IMAGING PORTFOLIO", style={
                    "color": BLUE, "fontSize": "10px",
                    "letterSpacing": "3px", "margin": "0 0 4px",
                }),
                html.H1("Anatomy Atlas", style={
                    "color": "white", "fontSize": "26px",
                    "margin": "0 0 2px", "fontWeight": "700",
                }),
                html.P("Interactive multi-organ segmentation — synthetic phantom",
                       style={"color": MUTED, "fontSize": "12px",
                              "margin": "0 0 20px"}),
            ]),

            # System filter pills
            html.Div(id="system-filters", style={
                "display": "flex", "gap": "8px",
                "flexWrap": "wrap", "marginBottom": "20px",
            }, children=[
                html.Button(
                    "All",
                    id={"type": "sys-btn", "system": "all"},
                    n_clicks=0,
                    style={"background": f"{BLUE}20", "border": f"1px solid {BLUE}60",
                           "borderRadius": "20px", "color": BLUE,
                           "padding": "4px 14px", "fontSize": "11px",
                           "cursor": "pointer", "fontFamily": "inherit"},
                ),
            ] + [
                html.Button(
                    f"{v.get('icon','')} {v['label']}".strip(),
                    id={"type": "sys-btn", "system": k},
                    n_clicks=0,
                    style={"background": "transparent",
                           "border": f"1px solid {v['color']}60",
                           "borderRadius": "20px", "color": v["color"],
                           "padding": "4px 14px", "fontSize": "11px",
                           "cursor": "pointer", "fontFamily": "inherit"},
                )
                for k, v in systems.items()
            ]),

            # 3-column main layout
            html.Div(style={
                "display": "grid",
                "gridTemplateColumns": "200px 1fr 300px",
                "gap": "16px", "alignItems": "start",
            }, children=[

                # ── Left: organ list ────────────────────────────────────
                html.Div(style={**_card(padding="12px 6px"),
                                "maxHeight": "calc(100vh - 180px)",
                                "overflowY": "auto"}, children=[
                    _label("Structures", padding="0 8px 8px"),
                    html.Div(id="organ-list",
                             children=organ_list_items(organs, systems)),
                    html.Hr(style={"border": f"1px solid {BORDER}",
                                   "margin": "12px 4px"}),
                    html.Button("🎯  Start Quiz", id="quiz-btn", n_clicks=0,
                                style={"width": "100%", "padding": "8px",
                                       "background": PURPLE, "color": "white",
                                       "border": "none", "borderRadius": "6px",
                                       "cursor": "pointer", "fontWeight": "bold",
                                       "fontSize": "12px",
                                       "fontFamily": "inherit"}),
                    html.Div(id="quiz-input-area",
                             style={"display": "none",
                                    "padding": "8px 4px 0"},
                             children=[
                                 html.P(id="quiz-prompt", style={
                                     "color": TEXT, "fontSize": "11px",
                                     "margin": "0 0 6px",
                                 }),
                                 dcc.Input(
                                     id="quiz-answer", type="text",
                                     placeholder="Name this structure…",
                                     debounce=False, n_submit=0,
                                     style={"width": "100%",
                                            "background": BG,
                                            "border": f"1px solid {BORDER}",
                                            "color": "white",
                                            "padding": "6px 8px",
                                            "borderRadius": "6px",
                                            "fontFamily": "inherit",
                                            "boxSizing": "border-box",
                                            "fontSize": "12px"},
                                 ),
                                 html.Button(
                                     "Check", id="quiz-submit", n_clicks=0,
                                     style={"width": "100%", "marginTop": "6px",
                                            "padding": "6px",
                                            "background": BLUE, "color": "white",
                                            "border": "none",
                                            "borderRadius": "6px",
                                            "cursor": "pointer",
                                            "fontFamily": "inherit",
                                            "fontSize": "12px"},
                                 ),
                                 html.P(id="quiz-result", style={
                                     "fontSize": "12px", "fontWeight": "bold",
                                     "padding": "6px 0 0", "margin": "0",
                                 }),
                             ]),
                ]),

                # ── Centre: slice viewer ────────────────────────────────
                html.Div([
                    html.Div(style=_card(padding="8px"), children=[
                        dcc.Graph(
                            id="slice-graph",
                            figure=make_slice_fig(volume, mid_z, l2rgb),
                            config={"displayModeBar": False},
                            style={"height": "480px"},
                        ),
                    ]),
                    html.Div(style={"padding": "10px 4px 0"}, children=[
                        _label("Axial slice — superior ↑  inferior ↓"),
                        dcc.Slider(
                            id="slice-slider",
                            min=0, max=n_slices - 1,
                            value=mid_z, step=1,
                            marks={
                                0:           {"label": "0 (inf)",
                                              "style": {"color": MUTED,
                                                        "fontSize": "10px"}},
                                mid_z:       {"label": str(mid_z),
                                              "style": {"color": MUTED,
                                                        "fontSize": "10px"}},
                                n_slices-1:  {"label": f"{n_slices-1} (sup)",
                                              "style": {"color": MUTED,
                                                        "fontSize": "10px"}},
                            },
                            tooltip={"placement": "bottom",
                                     "always_visible": False},
                        ),
                    ]),
                ]),

                # ── Right: info panel ───────────────────────────────────
                html.Div(style=_card(), children=[
                    html.P(id="info-name",
                           style={"color": "white", "fontSize": "17px",
                                  "fontWeight": "bold",
                                  "margin": "0 0 3px",
                                  "wordBreak": "break-word"},
                           children="Click a structure"),
                    html.P(id="info-system",
                           style={"color": MUTED, "fontSize": "11px",
                                  "margin": "0 0 16px"}),
                    html.Div(style={
                        "display": "grid",
                        "gridTemplateColumns": "1fr 1fr",
                        "gap": "12px", "marginBottom": "16px",
                    }, children=[
                        html.Div([
                            _label("Volume"),
                            html.P(id="info-volume",
                                   style={"color": BLUE, "fontSize": "20px",
                                          "fontWeight": "bold",
                                          "margin": "0"},
                                   children="—"),
                        ]),
                        html.Div([
                            _label("Normal range"),
                            html.P(id="info-range",
                                   style={"color": TEXT, "fontSize": "12px",
                                          "margin": "0"},
                                   children="—"),
                        ]),
                    ]),
                    _label("Clinical notes"),
                    html.Div(id="info-notes", style={
                        "background": BG,
                        "border": f"1px solid {BORDER}",
                        "borderRadius": "6px",
                        "padding": "10px",
                        "fontSize": "12px",
                        "color": TEXT,
                        "lineHeight": "1.65",
                        "minHeight": "80px",
                        "marginBottom": "12px",
                    }),
                    html.Div(id="info-fact", style={
                        "background": "#1e3a5f",
                        "borderLeft": f"3px solid {BLUE}",
                        "borderRadius": "0 6px 6px 0",
                        "padding": "8px 10px",
                        "fontSize": "11px",
                        "color": "#7dd3fc",
                        "lineHeight": "1.5",
                        "minHeight": "20px",
                    }),
                ]),

            ]),
        ],
    )

    # ── Callbacks ────────────────────────────────────────────────────────────

    @app.callback(
        Output("active-system", "data"),
        Input({"type": "sys-btn", "system": ALL}, "n_clicks"),
        State({"type": "sys-btn", "system": ALL}, "id"),
        prevent_initial_call=True,
    )
    def update_system(n_clicks_list, ids):
        triggered = ctx.triggered_id
        if triggered is None:
            return "all"
        return triggered["system"]

    @app.callback(
        Output("selected-organ", "data"),
        Input({"type": "organ-row", "key": ALL}, "n_clicks"),
        State({"type": "organ-row", "key": ALL}, "id"),
        State("quiz-state", "data"),
        prevent_initial_call=True,
    )
    def select_organ(n_clicks_list, ids, quiz):
        triggered = ctx.triggered_id
        if triggered is None or quiz.get("active"):
            return None
        return triggered["key"]

    @app.callback(
        Output("organ-list", "children"),
        Input("active-system", "data"),
        Input("selected-organ", "data"),
        Input("quiz-state", "data"),
    )
    def refresh_organ_list(system, selected, quiz):
        sel = quiz.get("organ") if quiz.get("active") else selected
        return organ_list_items(organs, systems, system, sel)

    @app.callback(
        Output("slice-graph", "figure"),
        Input("slice-slider", "value"),
        Input("selected-organ", "data"),
        Input("active-system", "data"),
        Input("quiz-state", "data"),
    )
    def update_slice(z, selected, system, quiz):
        if quiz.get("active") and quiz.get("organ"):
            hl = key_to_label.get(quiz["organ"])
        elif selected:
            hl = key_to_label.get(selected)
        else:
            hl = None

        if system != "all":
            vis = {organs[k]["label"]
                   for k, v in organs.items()
                   if v.get("system") == system}
        else:
            vis = None

        suffix = ""
        if hl:
            name = organs.get(
                quiz.get("organ") if quiz.get("active") else selected,
                {}
            ).get("display_name", "")
            suffix = f" — {name}"
        return make_slice_fig(volume, z, l2rgb, hl, vis, suffix)

    @app.callback(
        Output("info-name",   "children"),
        Output("info-system", "children"),
        Output("info-volume", "children"),
        Output("info-range",  "children"),
        Output("info-notes",  "children"),
        Output("info-fact",   "children"),
        Input("selected-organ", "data"),
        Input("quiz-state", "data"),
    )
    def update_info(selected, quiz):
        key = None
        if quiz.get("active") and quiz.get("organ") and not quiz.get("revealing"):
            return "❓ Name the highlighted structure", "", "—", "—", "", ""
        if quiz.get("active") and quiz.get("organ"):
            key = quiz["organ"]
        elif selected:
            key = selected

        if not key or key not in organs:
            return "Click a structure", "", "—", "—", "", ""

        data    = organs[key]
        sys_key = data.get("system", "")
        sys_v   = systems.get(sys_key, {})
        sys_str = f"{sys_v.get('icon','')} {sys_v.get('label','')}".strip()

        vol_ml  = organ_volumes.get(key)
        vol_str = f"{vol_ml:.0f} mL" if vol_ml else "—"

        vr = data.get("volume_range_ml")
        rng_str = f"{vr[0]}–{vr[1]} mL" if vr else "—"

        fact = data.get("fun_fact", "")
        fact_el = f"💡 {fact}" if fact else ""

        return (data["display_name"], sys_str, vol_str, rng_str,
                data.get("clinical_notes", ""), fact_el)

    # Quiz callbacks

    @app.callback(
        Output("quiz-state",      "data"),
        Output("quiz-btn",        "children"),
        Output("quiz-input-area", "style"),
        Output("quiz-prompt",     "children"),
        Output("quiz-result",     "children"),
        Output("quiz-result",     "style"),
        Output("quiz-answer",     "value"),
        Input("quiz-btn",    "n_clicks"),
        Input("quiz-submit", "n_clicks"),
        Input("quiz-answer", "n_submit"),
        State("quiz-state",  "data"),
        State("quiz-answer", "value"),
        State("active-system", "data"),
        prevent_initial_call=True,
    )
    def handle_quiz(btn_clicks, submit_clicks, n_submit, quiz, answer, system):
        triggered = ctx.triggered_id

        if triggered == "quiz-btn":
            active = not quiz.get("active", False)
            if active:
                organ_key = _random_organ(organs, system)
                new_quiz = {
                    "active": True, "organ": organ_key,
                    "correct": 0, "total": 0, "revealing": False,
                }
                prompt = f"Name the highlighted structure ({_count_for_system(organs, system)} organs in this system)"
                return (new_quiz, "🎯  Stop Quiz",
                        {"display": "block", "padding": "8px 4px 0"},
                        prompt, "", {}, "")
            else:
                return ({"active": False, "organ": None,
                         "correct": 0, "total": 0, "revealing": False},
                        "🎯  Start Quiz",
                        {"display": "none"}, "", "", {}, "")

        if triggered in ("quiz-submit", "quiz-answer") and quiz.get("active"):
            ans     = (answer or "").strip().lower()
            organ   = quiz.get("organ")
            correct = organs[organ]["display_name"].lower() if organ else ""
            is_right = _quiz_match(ans, correct)

            new_total   = quiz["total"] + 1
            new_correct = quiz["correct"] + (1 if is_right else 0)
            score_str   = f"({new_correct}/{new_total})"

            if is_right:
                result_text  = f"✅ Correct! — {organs[organ]['display_name']} {score_str}"
                result_style = {"color": GREEN, "fontSize": "12px",
                                "fontWeight": "bold", "padding": "6px 0 0",
                                "margin": "0"}
            else:
                result_text  = f"❌ It was: {organs[organ]['display_name']} {score_str}"
                result_style = {"color": RED, "fontSize": "12px",
                                "fontWeight": "bold", "padding": "6px 0 0",
                                "margin": "0"}

            next_organ = _random_organ(organs, system)
            new_quiz = {
                "active": True, "organ": next_organ,
                "correct": new_correct, "total": new_total,
                "revealing": False,
            }
            prompt = f"Name the highlighted structure"
            return (new_quiz, "🎯  Stop Quiz",
                    {"display": "block", "padding": "8px 4px 0"},
                    prompt, result_text, result_style, "")

        return quiz, "🎯  Start Quiz", {"display": "none"}, "", "", {}, ""

    return app


# ---------------------------------------------------------------------------
# Quiz helpers
# ---------------------------------------------------------------------------

def _quiz_match(answer: str, correct: str) -> bool:
    if not answer:
        return False
    a = answer.strip().lower()
    c = correct.strip().lower()
    if a == c or c in a:
        return True
    SKIP = {"left", "right", "upper", "lower"}
    words = sorted(c.split(), key=len, reverse=True)
    key_word = next((w for w in words if len(w) >= 4 and w not in SKIP), None)
    return bool(key_word and key_word in a)


def _random_organ(organs, system="all"):
    keys = [k for k, v in organs.items()
            if system == "all" or v.get("system") == system]
    return random.choice(keys) if keys else list(organs.keys())[0]


def _count_for_system(organs, system):
    if system == "all":
        return len(organs)
    return sum(1 for v in organs.values() if v.get("system") == system)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phantom", default="./phantom",
                        help="Directory containing segmentation.nii.gz")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()

    phantom_dir = Path(args.phantom)
    if not (phantom_dir / "segmentation.nii.gz").exists():
        print(f"Segmentation not found in {phantom_dir}. Generating…")
        import subprocess, sys
        subprocess.run([sys.executable, "generate_phantom.py",
                        "--out", str(phantom_dir)], check=True)

    print("Loading atlas data…")
    volume, organs, systems, organ_volumes = load_atlas(phantom_dir)
    print(f"  Volume shape: {volume.shape}  "
          f"({volume.shape[0]} axial slices, "
          f"{len(organs)} organs)")

    app = build_app(volume, organs, systems, organ_volumes)
    print(f"\nAnatomy Atlas running at http://localhost:{args.port}\n")
    app.run(debug=False, port=args.port)
