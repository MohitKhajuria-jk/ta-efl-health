"""Vector shield-and-cross logo drawn with QPainter (no image assets needed)."""
from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QLinearGradient, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QWidget


class ShieldLogo(QWidget):
    def __init__(self, size: int = 64, heartbeat: bool = False) -> None:
        super().__init__()
        self._heartbeat = heartbeat
        self.setFixedSize(size, int(size * 1.12))

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt API
        w, h = float(self.width()), float(self.height())
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.moveTo(w * 0.5, h * 0.02)
        path.lineTo(w * 0.94, h * 0.17)
        path.cubicTo(w * 0.94, h * 0.60, w * 0.78, h * 0.84, w * 0.5, h * 0.98)
        path.cubicTo(w * 0.22, h * 0.84, w * 0.06, h * 0.60, w * 0.06, h * 0.17)
        path.closeSubpath()
        grad = QLinearGradient(QPointF(0, 0), QPointF(w, h))
        grad.setColorAt(0.0, QColor("#4F8BFF"))
        grad.setColorAt(1.0, QColor("#1D4ED8"))
        p.setPen(QPen(QBrush(QColor("#E0EAFF")), max(1.0, w * 0.035)))
        p.setBrush(QBrush(grad))
        p.drawPath(path)
        # white medical cross
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor("white")))
        cx, cy, arm, thick = w * 0.5, h * 0.47, w * 0.40, w * 0.15
        p.drawRoundedRect(QRectF(cx - thick / 2, cy - arm / 2, thick, arm), 2, 2)
        p.drawRoundedRect(QRectF(cx - arm / 2, cy - thick / 2, arm, thick), 2, 2)
        if self._heartbeat:
            pen = QPen(QBrush(QColor(255, 255, 255, 110)), max(1.0, w * 0.025))
            p.setPen(pen)
            line = QPainterPath()
            y0 = h * 0.47
            line.moveTo(-w * 0.9, y0)
            for dx, dy in ((-0.6, 0), (-0.5, -0.18), (-0.42, 0.15), (-0.34, 0)):
                line.lineTo(w * (0.5 + dx), y0 + h * dy)
            p.drawPath(line)
        p.end()
