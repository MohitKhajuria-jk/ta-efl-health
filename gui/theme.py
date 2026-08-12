"""Light/dark theming for the TA-EFL Health desktop app.

One palette dictionary per theme drives both the Qt stylesheet and the
matplotlib figures, so switching themes restyles the whole window including
the charts. Purely presentational — no application logic lives here.
"""
from __future__ import annotations

PALETTES: dict[str, dict[str, str]] = {
    "light": {
        "bg": "#ffffff",
        "surface": "#f6f8fc",
        "surface_alt": "#ffffff",
        "row_alt": "#f8fafc",
        "border": "#e3e8f0",
        "text": "#0f1b33",
        "text_muted": "#64748b",
        "text_soft": "#475569",
        "accent": "#1a56db",
        "accent_hover": "#1e40af",
        "accent_fg": "#ffffff",
        "danger": "#e0483e",
        "success": "#1a9e5c",
        "warn_bg": "#fdf6e3",
        "warn_border": "#f0e2b6",
        "warn_text": "#8a5a00",
        "nav_bg": "#0f1b33",
        "nav_fg": "#cbd5e1",
        "risk_high_bg": "#fdf3f2",
        "risk_high_border": "#f3cdc9",
        "risk_low_bg": "#f1faf4",
        "risk_low_border": "#c7e8d5",
        "header_bg": "#eef2f9",
        "chart_bg": "#ffffff",
        "chart_fg": "#0f1b33",
        "chart_grid": "#c9d2e0",
        "disabled": "#94a3b8",
    },
    "dark": {
        "bg": "#0f172a",
        "surface": "#1a2337",
        "surface_alt": "#141d2f",
        "row_alt": "#182134",
        "border": "#2c3852",
        "text": "#e8edf7",
        "text_muted": "#94a3b8",
        "text_soft": "#b6c2d4",
        "accent": "#4f83f1",
        "accent_hover": "#3b6fdd",
        "accent_fg": "#ffffff",
        "danger": "#f2695e",
        "success": "#3fc07f",
        "warn_bg": "#33290f",
        "warn_border": "#5c4a1c",
        "warn_text": "#e8c56b",
        "nav_bg": "#0b1222",
        "nav_fg": "#cbd5e1",
        "risk_high_bg": "#3a1f1d",
        "risk_high_border": "#7a3a34",
        "risk_low_bg": "#12332a",
        "risk_low_border": "#2f6b52",
        "header_bg": "#1e2942",
        "chart_bg": "#141d2f",
        "chart_fg": "#e8edf7",
        "chart_grid": "#38445e",
        "disabled": "#64748b",
    },
}

_current = "light"


def available() -> list[str]:
    return list(PALETTES)


def name() -> str:
    return _current


def set_theme(theme: str) -> str:
    global _current
    _current = theme if theme in PALETTES else "light"
    return _current


def toggle() -> str:
    return set_theme("dark" if _current == "light" else "light")


def T() -> dict[str, str]:
    """The active palette."""
    return PALETTES[_current]


# ---------------------------------------------------------------------------
# Qt stylesheet
# ---------------------------------------------------------------------------
def stylesheet() -> str:
    c = T()
    return f"""
    QWidget {{
        background: {c['bg']};
        color: {c['text']};
        font-size: 13px;
    }}
    QLabel {{ background: transparent; }}
    QLabel#muted  {{ color: {c['text_muted']}; font-size: 12px; }}
    QLabel#soft   {{ color: {c['text_soft']};  font-size: 12px; }}
    QLabel#h1     {{ color: {c['text']}; font-size: 20px; font-weight: 700; }}
    QLabel#h2     {{ color: {c['text']}; font-size: 15px; font-weight: 700;
                     padding-top: 8px; }}
    QLabel#h3     {{ color: {c['text_soft']}; font-size: 13px; font-weight: 700;
                     padding-top: 8px; }}
    QLabel#metric {{ color: {c['accent']}; font-size: 22px; font-weight: 600; }}
    QLabel#body   {{ color: {c['text']}; font-size: 13px; }}
    QLabel#dangerHead {{ color: {c['danger']}; font-size: 12px; font-weight: 600; }}
    QLabel#accentHead {{ color: {c['accent']}; font-size: 12px; font-weight: 600; }}
    QLabel#warn {{
        color: {c['warn_text']}; background: {c['warn_bg']};
        border: 1px solid {c['warn_border']}; border-radius: 6px;
        padding: 8px; font-size: 12px;
    }}

    QFrame#card {{
        background: {c['surface']};
        border: 1px solid {c['border']};
        border-radius: 8px;
    }}
    QFrame#panel {{
        background: {c['surface_alt']};
        border: 1px solid {c['border']};
        border-radius: 6px;
    }}
    QFrame#placeholder {{
        background: {c['surface']};
        border: 1px dashed {c['border']};
        border-radius: 8px;
    }}
    QFrame#card QLabel, QFrame#panel QLabel, QFrame#placeholder QLabel {{
        border: none; background: transparent;
    }}

    QPushButton {{
        background: {c['surface']};
        color: {c['text']};
        border: 1px solid {c['border']};
        border-radius: 6px;
        padding: 7px 14px;
    }}
    QPushButton:hover  {{ border-color: {c['accent']}; }}
    QPushButton:disabled {{ color: {c['disabled']}; border-color: {c['border']}; }}
    QPushButton#primary {{
        background: {c['accent']}; color: {c['accent_fg']};
        border: none; font-weight: 600; padding: 8px 18px;
    }}
    QPushButton#primary:hover    {{ background: {c['accent_hover']}; }}
    QPushButton#primary:disabled {{ background: {c['disabled']}; }}

    QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QTextEdit {{
        background: {c['surface_alt']};
        color: {c['text']};
        border: 1px solid {c['border']};
        border-radius: 6px;
        padding: 4px 6px;
        selection-background-color: {c['accent']};
        selection-color: {c['accent_fg']};
    }}
    QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {{
        border-color: {c['accent']};
    }}
    QComboBox QAbstractItemView {{
        background: {c['surface_alt']};
        color: {c['text']};
        border: 1px solid {c['border']};
        selection-background-color: {c['accent']};
        selection-color: {c['accent_fg']};
    }}

    QTableWidget, QTableView {{
        background: {c['surface_alt']};
        alternate-background-color: {c['row_alt']};
        color: {c['text']};
        gridline-color: {c['border']};
        border: 1px solid {c['border']};
        border-radius: 6px;
    }}
    QTableWidget::item:selected, QTableView::item:selected {{
        background: {c['accent']}; color: {c['accent_fg']};
    }}
    QHeaderView::section {{
        background: {c['header_bg']};
        color: {c['text_soft']};
        border: none;
        border-bottom: 1px solid {c['border']};
        padding: 6px;
        font-weight: 600;
    }}

    QProgressBar {{
        background: {c['surface']};
        border: 1px solid {c['border']};
        border-radius: 6px;
        text-align: center;
        color: {c['text']};
    }}
    QProgressBar::chunk {{ background: {c['accent']}; border-radius: 5px; }}

    QListWidget#nav {{
        background: {c['nav_bg']};
        color: {c['nav_fg']};
        border: none;
        font-size: 13px;
    }}
    QListWidget#nav::item {{ padding: 11px; }}
    QListWidget#nav::item:selected {{
        background: {c['accent']}; color: {c['accent_fg']};
    }}
    QListWidget#nav::item:hover {{ background: {c['surface']}; }}

    QScrollArea {{ border: none; background: transparent; }}
    QScrollBar:vertical {{
        background: transparent; width: 10px; margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: {c['border']}; border-radius: 5px; min-height: 30px;
    }}
    QScrollBar::handle:vertical:hover {{ background: {c['accent']}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
    QToolTip {{
        background: {c['surface']}; color: {c['text']};
        border: 1px solid {c['border']}; padding: 4px;
    }}
    """


# ---------------------------------------------------------------------------
# Matplotlib
# ---------------------------------------------------------------------------
def style_axes(fig, *axes) -> None:
    """Apply the active palette to a figure and its axes."""
    c = T()
    fig.patch.set_facecolor(c["chart_bg"])
    for ax in (axes or fig.axes):
        ax.set_facecolor(c["chart_bg"])
        for spine in ax.spines.values():
            spine.set_color(c["chart_grid"])
        ax.tick_params(colors=c["chart_fg"], labelcolor=c["chart_fg"])
        ax.xaxis.label.set_color(c["chart_fg"])
        ax.yaxis.label.set_color(c["chart_fg"])
        ax.title.set_color(c["chart_fg"])
        ax.grid(color=c["chart_grid"], alpha=0.35)
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(c["chart_bg"])
            legend.get_frame().set_edgecolor(c["chart_grid"])
            for text in legend.get_texts():
                text.set_color(c["chart_fg"])


def restyle(fig) -> None:
    """Re-theme a figure that has already been drawn (used on theme switch)."""
    style_axes(fig)
