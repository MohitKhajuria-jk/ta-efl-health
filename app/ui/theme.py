"""Colours and the global Qt stylesheet (matched to the design reference)."""
from __future__ import annotations

NAVY = "#0B1D4A"
NAVY_2 = "#102A63"
BLUE = "#1D4ED8"
BLUE_HOVER = "#1E40AF"
BLUE_LIGHT = "#EFF4FF"
TITLE = "#0B2A6F"
CARD_TITLE = "#1E40AF"
TEXT = "#1F2937"
MUTED = "#6B7280"
BORDER = "#E2E8F0"
BG = "#F4F6FB"
GREEN = "#16A34A"
GREEN_HOVER = "#15803D"
RED = "#DC2626"
ORANGE = "#D97706"
PURPLE = "#7C3AED"

TINTS = {  # metric-card backgrounds and icon colours (dashboard row)
    "blue": ("#EEF4FF", "#1D4ED8"),
    "green": ("#EDFBF1", "#16A34A"),
    "purple": ("#F4EFFF", "#7C3AED"),
    "red": ("#FFF0F1", "#E11D48"),
    "slate": ("#F1F5F9", "#334155"),
}

FONT_FAMILY = "Segoe UI"

STYLESHEET = f"""
* {{ font-family: "{FONT_FAMILY}", "Segoe UI", Arial, sans-serif; font-size: 10pt; color: {TEXT}; }}
QMainWindow, #Workspace, QStackedWidget#Pages {{ background: {BG}; }}
QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}

/* ---------- sidebar ---------- */
#Sidebar {{ background: {NAVY}; }}
#Sidebar QLabel {{ color: #E5ECFF; }}
#SidebarBrand {{ color: white; font-size: 11pt; font-weight: 600; }}
#NavButton {{
    color: #DCE5FF; background: transparent; border: none; border-radius: 6px;
    text-align: left; padding: 9px 12px; font-size: 10pt;
}}
#NavButton:hover {{ background: {NAVY_2}; }}
#NavButton:checked {{ background: {BLUE}; color: white; font-weight: 600; }}
#SidebarFooter QCheckBox {{ color: #DCE5FF; font-size: 9pt; }}
#SidebarFooter QPushButton {{
    color: #DCE5FF; background: {NAVY_2}; border: 1px solid #23408A; border-radius: 5px;
    padding: 5px 8px; font-size: 9pt;
}}
#SidebarFooter QPushButton:hover {{ background: #18357A; }}

/* ---------- headers ---------- */
#PageTitle {{ color: {TITLE}; font-size: 19pt; font-weight: 700; }}
#PageSubtitle {{ color: #374151; font-size: 10pt; }}
#CardTitle {{ color: {CARD_TITLE}; font-size: 11pt; font-weight: 700; }}
#Muted {{ color: {MUTED}; font-size: 9pt; }}

/* ---------- cards ---------- */
#Card {{ background: white; border: 1px solid {BORDER}; border-radius: 8px; }}
#MetricCard {{ border: 1px solid {BORDER}; border-radius: 8px; }}
#MetricValue {{ font-size: 20pt; font-weight: 700; color: #0F172A; }}
#MetricCaption {{ color: #334155; font-size: 9.5pt; }}

/* ---------- banners ---------- */
#BannerReference {{ background: #FFF7E6; border: 1px solid #F59E0B; border-radius: 6px; }}
#BannerReference QLabel {{ color: #92400E; font-weight: 600; }}
#BannerInfo {{ background: {BLUE_LIGHT}; border: 1px solid #BFD3FF; border-radius: 6px; }}
#BannerInfo QLabel {{ color: #1E3A8A; }}
#BannerWarning {{ background: #FFFBEB; border: 1px solid #FCD34D; border-radius: 6px; }}
#BannerWarning QLabel {{ color: #92400E; }}
#BannerError {{ background: #FEF2F2; border: 1px solid #FCA5A5; border-radius: 6px; }}
#BannerError QLabel {{ color: #991B1B; }}
#BannerSuccess {{ background: #F0FDF4; border: 1px solid #86EFAC; border-radius: 6px; }}
#BannerSuccess QLabel {{ color: #166534; }}

/* ---------- inputs ---------- */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextBrowser {{
    background: white; border: 1px solid #CBD5E1; border-radius: 5px; padding: 5px 8px;
    selection-background-color: {BLUE};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border: 1px solid {BLUE}; }}
QLineEdit[invalid="true"], QComboBox[invalid="true"] {{ border: 1px solid {RED}; background: #FFF5F5; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QPlainTextEdit#Log {{ font-family: Consolas, "Courier New", monospace; font-size: 9pt; }}
QCheckBox {{ spacing: 8px; }}

/* ---------- buttons ---------- */
QPushButton {{
    background: white; border: 1px solid #CBD5E1; border-radius: 6px; padding: 7px 14px;
}}
QPushButton:hover {{ background: #F1F5F9; }}
QPushButton:disabled {{ color: #94A3B8; background: #F1F5F9; border-color: #E2E8F0; }}
QPushButton#Primary {{ background: {BLUE}; color: white; border: none; font-weight: 600; }}
QPushButton#Primary:hover {{ background: {BLUE_HOVER}; }}
QPushButton#Primary:disabled {{ background: #93A8E8; color: #EEF2FF; }}
QPushButton#Success {{ background: {GREEN}; color: white; border: none; font-weight: 600; }}
QPushButton#Success:hover {{ background: {GREEN_HOVER}; }}
QPushButton#Success:disabled {{ background: #9BD3AE; color: #F0FDF4; }}
QPushButton#Link {{ background: transparent; border: none; color: {BLUE}; padding: 2px; text-align: left; }}
QPushButton#Link:hover {{ text-decoration: underline; }}
QPushButton#Danger {{ background: white; color: {RED}; border: 1px solid #FCA5A5; }}

/* ---------- tabs (segmented, as in the design) ---------- */
QTabWidget::pane {{ border: none; top: 6px; }}
QTabBar::tab {{
    background: #EEF1F6; color: #334155; padding: 7px 18px; margin-right: 2px;
    border: 1px solid {BORDER}; border-bottom: none; border-top-left-radius: 5px;
    border-top-right-radius: 5px; min-width: 110px;
}}
QTabBar::tab:selected {{ background: {BLUE}; color: white; font-weight: 600; border-color: {BLUE}; }}
QTabBar::tab:hover:!selected {{ background: #E2E8F0; }}

/* ---------- tables ---------- */
QTableWidget {{
    background: white; border: 1px solid {BORDER}; border-radius: 6px;
    gridline-color: #EEF2F7; alternate-background-color: #FAFBFD;
}}
QTableWidget::item {{ padding: 4px; }}
QTableWidget::item:selected {{ background: #DBEAFE; color: {TEXT}; }}
QHeaderView::section {{
    background: #F8FAFC; color: #0F172A; font-weight: 600; padding: 6px 6px;
    border: none; border-bottom: 1px solid {BORDER};
}}
QProgressBar {{
    background: #E5EAF3; border: none; border-radius: 5px; height: 10px; text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{ background: {BLUE}; border-radius: 5px; }}
QToolTip {{ background: #111827; color: white; border: none; padding: 4px 6px; }}
"""

# Surfaces that Qt would otherwise paint from the (possibly dark) system palette.
STYLESHEET += f"""
QWidget#Workspace, QTabWidget > QWidget, QStackedWidget > QWidget {{ background: {BG}; }}
QDialog, QMessageBox {{ background: white; }}
QMessageBox QLabel, QDialog QLabel {{ color: {TEXT}; }}
QComboBox QAbstractItemView {{
    background: white; color: {TEXT}; border: 1px solid #CBD5E1;
    selection-background-color: #DBEAFE; selection-color: {TEXT}; outline: none;
}}
QMenu {{ background: white; color: {TEXT}; border: 1px solid #CBD5E1; }}
QMenu::item:selected {{ background: #DBEAFE; }}
QGroupBox {{ background: white; border: 1px solid {BORDER}; border-radius: 6px; margin-top: 6px; padding: 8px; }}
#LoginPane {{ background: white; }}
#ResultBox[tone="high"] {{ background: #FDECEC; border-radius: 8px; }}
#ResultBox[tone="low"] {{ background: #ECFDF3; border-radius: 8px; }}
#ResultBox[tone="empty"], #ResultBox[tone="reference"] {{ background: #F1F5F9; border-radius: 8px; }}
#TipBox {{ background: #F0F6FF; border: 1px solid #C7DAFF; border-radius: 8px; }}
#ReportPaper {{ background: white; border: 1px solid {BORDER}; border-radius: 4px; }}
QScrollBar:vertical {{ background: transparent; width: 10px; }}
QScrollBar::handle:vertical {{ background: #CBD5E1; border-radius: 5px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
"""


def light_palette():
    """An explicit light palette so Windows dark mode cannot make text unreadable."""
    from PyQt6.QtGui import QColor, QPalette

    pal = QPalette()
    roles = QPalette.ColorRole
    colors = {
        roles.Window: BG, roles.WindowText: TEXT, roles.Base: "#FFFFFF",
        roles.AlternateBase: "#FAFBFD", roles.Text: TEXT, roles.Button: "#FFFFFF",
        roles.ButtonText: TEXT, roles.BrightText: "#FFFFFF", roles.ToolTipBase: "#111827",
        roles.ToolTipText: "#FFFFFF", roles.Highlight: "#DBEAFE", roles.HighlightedText: TEXT,
        roles.Link: BLUE, roles.PlaceholderText: "#94A3B8", roles.Light: "#FFFFFF",
        roles.Midlight: "#F1F5F9", roles.Mid: "#CBD5E1", roles.Dark: "#94A3B8",
        roles.Shadow: "#64748B",
    }
    for role, c in colors.items():
        pal.setColor(role, QColor(c))
    disabled = QPalette.ColorGroup.Disabled
    for role in (roles.WindowText, roles.Text, roles.ButtonText):
        pal.setColor(disabled, role, QColor("#94A3B8"))
    return pal


def apply_theme(app) -> None:
    """Apply style, light palette, font and stylesheet to the QApplication."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QFont

    app.setStyle("Fusion")
    try:  # Qt >= 6.8: stop following the Windows dark/light setting
        app.styleHints().setColorScheme(Qt.ColorScheme.Light)
    except AttributeError:
        pass
    app.setPalette(light_palette())
    app.setFont(QFont(FONT_FAMILY, 10))
    app.setStyleSheet(STYLESHEET)
