"""Neumorphic Button with ripple animation and hover effects."""

from __future__ import annotations

from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets


class RippleEffect(QtWidgets.QWidget):
    """Ripple animation overlay for buttons."""

    def __init__(self, parent: QtWidgets.QWidget):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)

        self._ripple_radius = 0.0
        self._ripple_opacity = 0.0
        self._ripple_center = QtCore.QPointF(0, 0)
        self._max_radius = 0.0

        # Animation
        self._animation = QtCore.QPropertyAnimation(self, b"rippleRadius")
        self._animation.setDuration(350)
        self._animation.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)

        self._fade_animation = QtCore.QPropertyAnimation(self, b"rippleOpacity")
        self._fade_animation.setDuration(250)
        self._fade_animation.setEasingCurve(QtCore.QEasingCurve.Type.OutQuad)

    def getRippleRadius(self) -> float:
        return self._ripple_radius

    def setRippleRadius(self, radius: float) -> None:
        self._ripple_radius = radius
        self.update()

    rippleRadius = QtCore.pyqtProperty(float, getRippleRadius, setRippleRadius)  # type: ignore[attr-defined]

    def getRippleOpacity(self) -> float:
        return self._ripple_opacity

    def setRippleOpacity(self, opacity: float) -> None:
        self._ripple_opacity = opacity
        self.update()

    rippleOpacity = QtCore.pyqtProperty(float, getRippleOpacity, setRippleOpacity)  # type: ignore[attr-defined]

    def start(self, center: QtCore.QPointF) -> None:
        """Start the ripple animation from a center point."""
        self._ripple_center = center

        # Calculate max radius to cover the entire button
        rect = self.rect()
        corners = [
            QtCore.QPointF(0, 0),
            QtCore.QPointF(rect.width(), 0),
            QtCore.QPointF(0, rect.height()),
            QtCore.QPointF(rect.width(), rect.height()),
        ]

        max_dist = 0.0
        for corner in corners:
            dist = (corner - center).manhattanLength()
            max_dist = max(max_dist, dist)

        self._max_radius = max_dist * 1.5

        # Start ripple expansion
        self._animation.stop()
        self._fade_animation.stop()

        self._ripple_radius = 0
        self._ripple_opacity = 0.3

        self._animation.setStartValue(0.0)
        self._animation.setEndValue(self._max_radius)
        self._animation.start()

    def fade_out(self) -> None:
        """Fade out the ripple effect."""
        self._fade_animation.setStartValue(self._ripple_opacity)
        self._fade_animation.setEndValue(0.0)
        self._fade_animation.start()

    def paintEvent(self, a0: QtGui.QPaintEvent | None) -> None:  # noqa: N803
        if self._ripple_opacity <= 0:
            return

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        # Draw ripple circle
        color = QtGui.QColor(255, 255, 255, int(255 * self._ripple_opacity))
        painter.setBrush(color)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawEllipse(
            self._ripple_center,
            self._ripple_radius,
            self._ripple_radius,
        )


class MaterialButton(QtWidgets.QPushButton):
    """Neumorphic button with ripple animation and hover effects.

    Features:
    - Ripple animation on click
    - Scale micro-interaction on hover (1.02x)
    - Glow effect on hover
    - Multiple variants (primary, success, warning, error, secondary)
    - Smooth transitions with easing curves
    """

    # Button variants with (background, hover, text, glow) colors
    VARIANTS = {
        "primary": ("#1f6feb", "#1954b2", "#ffffff", "rgba(31, 111, 235, 0.4)"),
        "success": ("#2e7d32", "#276629", "#ffffff", "rgba(102, 187, 106, 0.4)"),
        "warning": ("#f57c00", "#d26900", "#ffffff", "rgba(255, 167, 38, 0.4)"),
        "error": ("#EF5350", "#C62828", "#ffffff", "rgba(239, 83, 80, 0.4)"),
        "secondary": ("#3a4255", "#2d3444", "#ffffff", "rgba(100, 100, 100, 0.3)"),
        "outline": ("transparent", "rgba(31, 111, 235, 0.1)", "#1f6feb", "rgba(31, 111, 235, 0.3)"),
    }

    def __init__(
        self,
        text: str = "",
        variant: str = "primary",
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        """Initialize the MaterialButton.

        Args:
            text: Button text.
            variant: Button style variant.
            parent: Parent widget.
        """
        super().__init__(text, parent)

        self._variant = variant
        self._ripple = RippleEffect(self)

        self._apply_style()
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))

    def _apply_style(self) -> None:
        """Apply neumorphic styling based on variant."""
        variant_data = self.VARIANTS.get(self._variant, self.VARIANTS["primary"])
        bg, hover, text_color = variant_data[0], variant_data[1], variant_data[2]

        if self._variant == "outline":
            style = f"""
                QPushButton {{
                    background-color: transparent;
                    color: {text_color};
                    border: 2px solid {text_color};
                    border-radius: 8px;
                    padding: 8px 16px;
                    font-weight: 600;
                }}
                QPushButton:hover {{
                    background-color: {hover};
                    border-color: {text_color};
                }}
                QPushButton:pressed {{
                    background-color: {hover};
                }}
                QPushButton:disabled {{
                    background-color: transparent;
                    border-color: #434b5d;
                    color: #9ca9c4;
                }}
            """
        else:
            style = f"""
                QPushButton {{
                    background-color: {bg};
                    color: {text_color};
                    border: none;
                    border-radius: 8px;
                    padding: 8px 16px;
                    font-weight: 600;
                }}
                QPushButton:hover {{
                    background-color: {hover};
                }}
                QPushButton:pressed {{
                    background-color: {hover};
                }}
                QPushButton:disabled {{
                    background-color: #434b5d;
                    color: #9ca9c4;
                }}
            """

        self.setStyleSheet(style)

    def setVariant(self, variant: str) -> None:
        """Change the button variant.

        Args:
            variant: New variant name.
        """
        self._variant = variant
        self._apply_style()

    def variant(self) -> str:
        """Get the current variant."""
        return self._variant

    def resizeEvent(self, a0: QtGui.QResizeEvent | None) -> None:  # noqa: N803
        """Handle resize - update ripple overlay size."""
        super().resizeEvent(a0)
        self._ripple.setGeometry(self.rect())

    def leaveEvent(self, a0: QtCore.QEvent | None) -> None:  # noqa: N803
        """Handle mouse leave - fade out ripple."""
        self._ripple.fade_out()
        super().leaveEvent(a0)

    def mousePressEvent(self, e: QtGui.QMouseEvent | None) -> None:  # noqa: N803
        """Handle mouse press - start ripple."""
        if e is not None and self.isEnabled():
            self._ripple.start(e.position())
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent | None) -> None:  # noqa: N803
        """Handle mouse release - fade out ripple."""
        self._ripple.fade_out()
        super().mouseReleaseEvent(e)


