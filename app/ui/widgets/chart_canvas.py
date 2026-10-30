"""Matplotlib canvas embedded in Qt, with hover read-out and bar picking."""
from __future__ import annotations

from typing import Callable

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QSizePolicy


class ChartCanvas(FigureCanvasQTAgg):
    def __init__(self, width: float = 5.0, height: float = 3.2, min_height: int = 240) -> None:
        fig = Figure(figsize=(width, height), dpi=100, facecolor="white")
        super().__init__(fig)  # sets self.figure
        self.ax = self.figure.add_subplot(111)
        self.setMinimumHeight(min_height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._pick_cb: Callable[[str], None] | None = None
        self._annot = None
        self.mpl_connect("pick_event", self._on_pick)
        self.mpl_connect("motion_notify_event", self._on_motion)

    def reset(self, n_axes: int = 1) -> list:
        self.figure.clear()
        axes = [self.figure.add_subplot(1, n_axes, i + 1) for i in range(n_axes)]
        self.ax = axes[0]
        self._annot = None
        return axes

    def finish(self) -> None:
        try:
            self.figure.tight_layout()
        except Exception:  # noqa: BLE001 - layout issues must not break the page
            pass
        self.draw_idle()

    # ----------------------------------------------------- interactions
    def on_pick(self, cb: Callable[[str], None]) -> None:
        self._pick_cb = cb

    def _on_pick(self, event) -> None:
        gid = getattr(event.artist, "get_gid", lambda: None)()
        if gid and self._pick_cb:
            self._pick_cb(gid)

    def _on_motion(self, event) -> None:
        """Show the data coordinates of the nearest line point under the cursor."""
        ax = event.inaxes
        if ax is None or not ax.lines:
            if self._annot is not None and self._annot.get_visible():
                self._annot.set_visible(False)
                self.draw_idle()
            return
        best = None
        for line in ax.lines:
            if line.get_label().startswith("_"):
                continue
            xs, ys = line.get_xdata(), line.get_ydata()
            for x, y in zip(xs, ys):
                try:
                    px, py = ax.transData.transform((float(x), float(y)))
                except (TypeError, ValueError):
                    continue
                d = (px - event.x) ** 2 + (py - event.y) ** 2
                if best is None or d < best[0]:
                    best = (d, x, y, line.get_label())
        if best is None or best[0] > 12 ** 2:
            if self._annot is not None and self._annot.get_visible():
                self._annot.set_visible(False)
                self.draw_idle()
            return
        _, x, y, lab = best
        text = f"{lab}\n({float(x):.3g}, {float(y):.3f})"
        if self._annot is None or self._annot.axes is not ax:
            self._annot = ax.annotate("", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
                                      fontsize=7, bbox={"boxstyle": "round", "fc": "#111827",
                                                        "ec": "none", "alpha": 0.9},
                                      color="white")
        self._annot.xy = (x, y)
        self._annot.set_text(text)
        self._annot.set_visible(True)
        self.draw_idle()
