"""Loading indicator widgets with modern animations.

Provides spinners and progress indicators with smooth animations for processing states.
"""

from __future__ import annotations

from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets

from themes import get_theme_manager


class SpinnerRing(QtWidgets.QWidget):
    """Rotating ring spinner with gradient arc."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        size: int = 40,
        line_width: int = 4,
    ):
        """Initialize spinner.

        Args:
            parent: Parent widget.
            size: Spinner diameter.
            line_width: Arc line width.
        """
        super().__init__(parent)

        self._size = size
        self._line_width = line_width
        self._rotation = 0.0

        self.setFixedSize(size, size)

        # Rotation animation
        self._rotation_anim = QtCore.QPropertyAnimation(self, b"rotation")
        self._rotation_anim.setDuration(1000)
        self._rotation_anim.setStartValue(0.0)
        self._rotation_anim.setEndValue(360.0)
        self._rotation_anim.setLoopCount(-1)
        self._rotation_anim.setEasingCurve(QtCore.QEasingCurve.Type.Linear)

    def getRotation(self) -> float:
        return self._rotation

    def setRotation(self, rotation: float) -> None:
        self._rotation = rotation
        self.update()

    rotation = QtCore.pyqtProperty(float, getRotation, setRotation)  # type: ignore[attr-defined]

    def start(self) -> None:
        """Start spinning."""
        self._rotation_anim.start()

    def stop(self) -> None:
        """Stop spinning."""
        self._rotation_anim.stop()

    def showEvent(self, a0: QtGui.QShowEvent | None) -> None:  # noqa: N803
        super().showEvent(a0)
        self.start()

    def hideEvent(self, a0: QtGui.QHideEvent | None) -> None:  # noqa: N803
        super().hideEvent(a0)
        self.stop()

    def paintEvent(self, a0: QtGui.QPaintEvent | None) -> None:  # noqa: N803
        """Draw rotating arc."""
        theme = get_theme_manager()

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        # Calculate arc rect with padding for line width
        padding = self._line_width // 2 + 1
        arc_rect = QtCore.QRectF(
            padding, padding,
            self._size - 2 * padding,
            self._size - 2 * padding,
        )

        # Create gradient pen
        gradient = QtGui.QConicalGradient(
            arc_rect.center().x(),
            arc_rect.center().y(),
            -self._rotation,
        )
        primary_color = QtGui.QColor(theme.colors.primary)
        transparent = QtGui.QColor(primary_color)
        transparent.setAlpha(0)

        gradient.setColorAt(0.0, primary_color)
        gradient.setColorAt(0.7, primary_color)
        gradient.setColorAt(1.0, transparent)

        pen = QtGui.QPen(QtGui.QBrush(gradient), self._line_width)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)

        # Draw arc (270 degrees, starting from rotation angle)
        painter.drawArc(arc_rect, int(-self._rotation * 16), int(270 * 16))


class ProcessingOverlay(QtWidgets.QWidget):
    """Semi-transparent overlay with centered loading indicator."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        message: str = "Processing...",
    ):
        """Initialize processing overlay.

        Args:
            parent: Parent widget to overlay.
            message: Message to display.
        """
        super().__init__(parent)

        self._message = message
        self._opacity = 0.0

        # Make widget fill parent
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)

        # Layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        # Spinner
        self._spinner = SpinnerRing(self, size=48, line_width=4)
        layout.addWidget(self._spinner, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)

        # Message label
        self._label = QtWidgets.QLabel(message, self)
        self._label.setStyleSheet(
            "color: #E8EAED; font-size: 12pt; background: transparent;"
        )
        self._label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._label)

        # Fade animation
        self._opacity_effect = QtWidgets.QGraphicsOpacityEffect(self)
        self._opacity_effect.setOpacity(0.0)
        self.setGraphicsEffect(self._opacity_effect)

        self._fade_anim = QtCore.QPropertyAnimation(self._opacity_effect, b"opacity")
        self._fade_anim.setDuration(300)
        self._fade_anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)

        self.hide()

    def setMessage(self, message: str) -> None:
        """Update the message text."""
        self._message = message
        self._label.setText(message)

    def show_overlay(self) -> None:
        """Show overlay with fade-in animation."""
        self.show()
        self.raise_()
        self._spinner.start()

        self._fade_anim.stop()
        self._fade_anim.setStartValue(0.0)
        self._fade_anim.setEndValue(1.0)
        self._fade_anim.start()

    def hide_overlay(self) -> None:
        """Hide overlay with fade-out animation."""
        self._fade_anim.stop()
        self._fade_anim.setStartValue(self._opacity_effect.opacity())
        self._fade_anim.setEndValue(0.0)
        self._fade_anim.finished.connect(self._on_fade_out_finished)
        self._fade_anim.start()

    def _on_fade_out_finished(self) -> None:
        """Handle fade-out completion."""
        self.hide()
        self._spinner.stop()
        try:
            self._fade_anim.finished.disconnect(self._on_fade_out_finished)
        except TypeError:
            pass

    def paintEvent(self, a0: QtGui.QPaintEvent | None) -> None:  # noqa: N803
        """Draw semi-transparent background."""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        # Semi-transparent dark background
        painter.setBrush(QtGui.QColor(0, 0, 0, 150))
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawRect(self.rect())

    def resizeEvent(self, a0: QtGui.QResizeEvent | None) -> None:  # noqa: N803
        """Ensure overlay fills parent."""
        super().resizeEvent(a0)
        if self.parent():
            parent_widget = self.parent()
            if isinstance(parent_widget, QtWidgets.QWidget):
                self.setGeometry(parent_widget.rect())


class RecordingPulse(QtWidgets.QWidget):
    """Pulsing red dot for recording indicator."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, size: int = 12):
        """Initialize recording pulse.

        Args:
            parent: Parent widget.
            size: Dot diameter.
        """
        super().__init__(parent)

        self._size = size
        self._pulse_scale = 1.0

        self.setFixedSize(size + 8, size + 8)  # Extra space for pulse

        # Pulse animation
        self._pulse_anim = QtCore.QPropertyAnimation(self, b"pulseScale")
        self._pulse_anim.setDuration(800)
        self._pulse_anim.setStartValue(1.0)
        self._pulse_anim.setKeyValueAt(0.5, 1.3)
        self._pulse_anim.setEndValue(1.0)
        self._pulse_anim.setLoopCount(-1)
        self._pulse_anim.setEasingCurve(QtCore.QEasingCurve.Type.InOutQuad)

    def getPulseScale(self) -> float:
        return self._pulse_scale

    def setPulseScale(self, scale: float) -> None:
        self._pulse_scale = scale
        self.update()

    pulseScale = QtCore.pyqtProperty(float, getPulseScale, setPulseScale)  # type: ignore[attr-defined]

    def start(self) -> None:
        self._pulse_anim.start()

    def stop(self) -> None:
        self._pulse_anim.stop()
        self._pulse_scale = 1.0
        self.update()

    def paintEvent(self, a0: QtGui.QPaintEvent | None) -> None:  # noqa: N803
        """Draw pulsing red dot."""
        theme = get_theme_manager()

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        center = QtCore.QPointF(self.width() / 2, self.height() / 2)
        base_radius = self._size / 2

        # Draw outer pulse (faded)
        if self._pulse_scale > 1.0:
            outer_radius = base_radius * self._pulse_scale
            outer_opacity = 1.0 - (self._pulse_scale - 1.0) / 0.3
            outer_color = QtGui.QColor(theme.colors.error)
            outer_color.setAlphaF(max(0, outer_opacity * 0.5))

            painter.setBrush(outer_color)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.drawEllipse(center, outer_radius, outer_radius)

        # Draw solid center dot
        painter.setBrush(QtGui.QColor(theme.colors.error))
        painter.drawEllipse(center, base_radius, base_radius)
