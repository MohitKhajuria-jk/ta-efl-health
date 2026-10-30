"""Icons rendered from the Windows 11 'Segoe Fluent Icons' font (or MDL2 on Windows 10).

If neither icon font is installed (e.g. Linux/macOS), a plain Unicode symbol is
used instead, so the application never depends on bundled image files.
"""
from __future__ import annotations

from functools import lru_cache

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QFontDatabase, QFontMetrics, QIcon, QPainter, QPixmap, QTextOption

# name: (Segoe Fluent/MDL2 code point, fallback unicode symbol)
GLYPHS: dict[str, tuple[int, str]] = {
    "dashboard": (0xE80F, "⌂"),
    "clients": (0xE716, "☷"),
    "training": (0xE713, "⚙"),
    "prediction": (0xE95E, "♥"),
    "explain": (0xEA80, "✦"),
    "results": (0xE9D2, "▤"),
    "trust": (0xEA18, "⛨"),
    "settings": (0xE713, "⚙"),
    "reports": (0xE8A5, "▦"),
    "about": (0xE946, "ⓘ"),
    "user": (0xE77B, "☺"),
    "lock": (0xE72E, "⚿"),
    "eye": (0xE890, "◉"),
    "search": (0xE721, "⌕"),
    "refresh": (0xE72C, "↻"),
    "play": (0xE768, "▶"),
    "stop": (0xE71A, "■"),
    "heart": (0xEB52, "♥"),
    "database": (0xE8F1, "▥"),
    "chart": (0xE9D2, "▤"),
    "shield": (0xEA18, "⛨"),
    "people": (0xE716, "☷"),
    "sync": (0xE895, "⟳"),
    "up": (0xE74A, "▲"),
    "down": (0xE74B, "▼"),
    "bulb": (0xEA80, "✦"),
    "check": (0xE73E, "✓"),
    "warning": (0xE7BA, "⚠"),
    "arrow": (0xE72A, "→"),
    "logout": (0xE7E8, "⏻"),
    "folder": (0xE8B7, "▣"),
    "brain": (0xE9D9, "✺"),
}

ICON_FONTS = ("Segoe Fluent Icons", "Segoe MDL2 Assets")


@lru_cache(maxsize=1)
def _icon_family() -> str | None:
    families = set(QFontDatabase.families())
    for fam in ICON_FONTS:
        if fam in families:
            return fam
    return None


def glyph(name: str) -> tuple[str, QFont]:
    """Return (text, font) for an icon name."""
    code, fallback = GLYPHS.get(name, (0, "•"))
    fam = _icon_family()
    if fam:
        font = QFont(fam)
        if QFontMetrics(font).inFontUcs4(code):
            return chr(code), font
    return fallback, QFont("Segoe UI Symbol")


@lru_cache(maxsize=256)
def _pixmap(name: str, color: str, size: int) -> QPixmap:
    text, font = glyph(name)
    font.setPixelSize(int(size * 0.8))
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    p.setFont(font)
    p.setPen(QColor(color))
    p.drawText(QRectF(0, 0, size, size), text, QTextOption(Qt.AlignmentFlag.AlignCenter))
    p.end()
    return pm


def icon(name: str, color: str = "#FFFFFF", size: int = 32) -> QIcon:
    return QIcon(_pixmap(name, color, size))


def pixmap(name: str, color: str, size: int) -> QPixmap:
    return _pixmap(name, color, size)
