"""Glassmorphism Card widget with animated reveal effects."""

from __future__ import annotations

from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets

from themes import get_theme_manager


class GlassCard(QtWidgets.QFrame):
    """Glassmorphism Card with animated reveal and hover effects.

    A container widget that provides:
    - Semi-transparent frosted glass background
    - Subtle light border for glass edge effect
    - Animated hover glow intensification
    - Fade + scale reveal animation on show
    - 16px border-radius for modern rounded corners
    """

    # Elevation shadow configurations: (blur, y_offset, opacity)
    ELEVATION_LEVELS = {
        0: (0, 0, 0.0),      # No shadow
        1: (12, 3, 0.20),    # Low (slightly stronger for glass)
        2: (18, 5, 0.25),    # Medium
        3: (24, 7, 0.30),    # High
        4: (32, 10, 0.35),   # Extra high
    }

    def __init__(
        self,
        title: str = "",
        parent: Optional[QtWidgets.QWidget] = None,
        elevation: int = 1,
        hover_elevation: int = 2,
        animate_on_show: bool = True,
    ):
        """Initialize the GlassCard.

        Args:
            title: Optional title displayed at the top of the card.
            parent: Parent widget.
            elevation: Initial elevation level (0-4).
            hover_elevation: Elevation level on hover (0-4).
            animate_on_show: Whether to play reveal animation on first show.
        """
        super().__init__(parent)

        self._title = title
        self._elevation = elevation
        self._hover_elevation = hover_elevation
        self._current_elevation = elevation
        self._animate_on_show = animate_on_show
        self._has_animated = False
        self._is_hovered = False

        # Animation components
        self._opacity_effect: Optional[QtWidgets.QGraphicsOpacityEffect] = None
        self._fade_anim: Optional[QtCore.QPropertyAnimation] = None
        self._shadow_effect: Optional[QtWidgets.QGraphicsDropShadowEffect] = None

        self._setup_ui()
        self._setup_animations()
        self._apply_elevation(elevation)

    def _setup_ui(self) -> None:
        """Set up the card UI."""
        theme = get_theme_manager()
        c = theme.colors

        # Main layout - consistent padding
        self._main_layout = QtWidgets.QVBoxLayout(self)
        self._main_layout.setContentsMargins(14, 12, 14, 12)
        self._main_layout.setSpacing(10)

        # Title label (if title provided)
        if self._title:
            self._title_label = QtWidgets.QLabel(self._title, self)
            self._title_label.setStyleSheet(
                f"font-weight: 600; font-size: 13pt; color: {c.group_title}; "
                "background: transparent; padding: 0; margin: 0;"
            )
            self._main_layout.addWidget(self._title_label)

        # Content widget (container for user content)
        self._content_widget = QtWidgets.QWidget(self)
        self._content_widget.setStyleSheet("background: transparent;")
        self._content_layout = QtWidgets.QVBoxLayout(self._content_widget)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(10)
        self._main_layout.addWidget(self._content_widget, stretch=1)

        # Base glassmorphism styling
        self.setStyleSheet(self._get_glass_stylesheet())

        # Enable mouse tracking for hover effects
        self.setMouseTracking(True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)

    def _setup_animations(self) -> None:
        """Set up animation components."""
        # Opacity effect for fade animation
        self._opacity_effect = QtWidgets.QGraphicsOpacityEffect(self)
        self._opacity_effect.setOpacity(1.0 if not self._animate_on_show else 0.0)

        # Fade animation
        self._fade_anim = QtCore.QPropertyAnimation(self._opacity_effect, b"opacity")
        self._fade_anim.setDuration(450)
        self._fade_anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)

    def _get_glass_stylesheet(self, is_hovered: bool = False) -> str:
        """Generate the glassmorphism stylesheet.

        Args:
            is_hovered: Whether card is in hover state.
        """
        theme = get_theme_manager()
        c = theme.colors

        bg = c.glass_bg_hover if is_hovered else c.glass_bg
        border = c.glass_border_hover if is_hovered else c.glass_border

        return f"""
            GlassCard {{
                background-color: {bg};
                border: 1px solid {border};
                border-radius: 16px;
            }}
        """

    def _apply_elevation(self, level: int, animated: bool = False) -> None:
        """Apply elevation shadow effect.

        Args:
            level: Elevation level (0-4).
            animated: Whether to animate the transition.
        """
        self._current_elevation = level

        # Get shadow params
        blur, offset, opacity = self.ELEVATION_LEVELS.get(level, self.ELEVATION_LEVELS[1])

        # Always create a fresh shadow effect - Qt deletes effects when replaced
        try:
            # Check if existing effect is still valid
            if self._shadow_effect is not None:
                self._shadow_effect.blurRadius()  # Will raise if deleted
        except RuntimeError:
            self._shadow_effect = None

        # Create new shadow effect
        self._shadow_effect = QtWidgets.QGraphicsDropShadowEffect(self)
        self._shadow_effect.setBlurRadius(blur)
        self._shadow_effect.setOffset(0, offset)
        self._shadow_effect.setColor(QtGui.QColor(0, 0, 0, int(255 * opacity)))

        self.setGraphicsEffect(self._shadow_effect)

    def showEvent(self, a0: QtGui.QShowEvent | None) -> None:  # noqa: N803
        """Handle show event - trigger reveal animation."""
        super().showEvent(a0)

        if self._animate_on_show and not self._has_animated:
            self._has_animated = True
            # Delay slightly to ensure geometry is set
            QtCore.QTimer.singleShot(10, self._play_reveal_animation)

    def _play_reveal_animation(self) -> None:
        """Play the reveal animation (fade + scale)."""
        if self._fade_anim is None or self._opacity_effect is None:
            return

        # Set opacity effect for fade
        self.setGraphicsEffect(self._opacity_effect)
        self._opacity_effect.setOpacity(0.0)

        # Start fade in
        self._fade_anim.setStartValue(0.0)
        self._fade_anim.setEndValue(1.0)

        # Connect to restore shadow effect after fade completes
        self._fade_anim.finished.connect(self._on_reveal_finished)
        self._fade_anim.start()

    def _on_reveal_finished(self) -> None:
        """Handle reveal animation completion."""
        # Restore shadow effect
        self._apply_elevation(self._elevation)

        # Disconnect to avoid repeated calls
        if self._fade_anim:
            try:
                self._fade_anim.finished.disconnect(self._on_reveal_finished)
            except TypeError:
                pass  # Already disconnected

    def reveal(self, delay: int = 0) -> None:
        """Manually trigger reveal animation.

        Args:
            delay: Delay in milliseconds before starting animation.
        """
        self._has_animated = False
        if delay > 0:
            QtCore.QTimer.singleShot(delay, self._play_reveal_animation)
        else:
            self._play_reveal_animation()

    def enterEvent(self, event: QtGui.QEnterEvent | None) -> None:
        """Handle mouse enter - increase elevation and update styling."""
        self._is_hovered = True
        self._apply_elevation(self._hover_elevation)
        self.setStyleSheet(self._get_glass_stylesheet(is_hovered=True))
        super().enterEvent(event)

    def leaveEvent(self, a0: QtCore.QEvent | None) -> None:  # noqa: N803
        """Handle mouse leave - restore normal elevation and styling."""
        self._is_hovered = False
        self._apply_elevation(self._elevation)
        self.setStyleSheet(self._get_glass_stylesheet(is_hovered=False))
        super().leaveEvent(a0)

    def setTitle(self, title: str) -> None:
        """Set the card title.

        Args:
            title: New title text.
        """
        self._title = title
        if hasattr(self, "_title_label"):
            self._title_label.setText(title)
        else:
            # Create title label if it doesn't exist
            theme = get_theme_manager()
            self._title_label = QtWidgets.QLabel(title, self)
            self._title_label.setStyleSheet(
                f"font-weight: 600; font-size: 13pt; color: {theme.colors.group_title}; "
                "background: transparent; padding: 0; margin: 0;"
            )
            self._main_layout.insertWidget(0, self._title_label)

    def title(self) -> str:
        """Get the card title."""
        return self._title

    def contentLayout(self) -> QtWidgets.QVBoxLayout:
        """Get the content layout for adding widgets.

        Returns:
            The content layout where child widgets should be added.
        """
        return self._content_layout

    def addWidget(self, widget: QtWidgets.QWidget, stretch: int = 0) -> None:
        """Add a widget to the card content.

        Args:
            widget: Widget to add.
            stretch: Stretch factor.
        """
        self._content_layout.addWidget(widget, stretch)

    def addLayout(self, layout: QtWidgets.QLayout, stretch: int = 0) -> None:
        """Add a layout to the card content.

        Args:
            layout: Layout to add.
            stretch: Stretch factor.
        """
        self._content_layout.addLayout(layout, stretch)

    def setElevation(self, level: int) -> None:
        """Set the base elevation level.

        Args:
            level: Elevation level (0-4).
        """
        self._elevation = level
        self._apply_elevation(level)

    def setHoverElevation(self, level: int) -> None:
        """Set the hover elevation level.

        Args:
            level: Elevation level on hover (0-4).
        """
        self._hover_elevation = level

    def setContentMargins(
        self, left: int, top: int, right: int, bottom: int
    ) -> None:
        """Set the card content margins.

        Args:
            left: Left margin.
            top: Top margin.
            right: Right margin.
            bottom: Bottom margin.
        """
        self._main_layout.setContentsMargins(left, top, right, bottom)

    def setSpacing(self, spacing: int) -> None:
        """Set the content layout spacing.

        Args:
            spacing: Spacing between content items.
        """
        self._content_layout.setSpacing(spacing)

    def setAnimateOnShow(self, animate: bool) -> None:
        """Set whether to animate on show.

        Args:
            animate: Whether to play reveal animation on show.
        """
        self._animate_on_show = animate

    def flash_glow(self, color: str = "success", duration: int = 600) -> None:
        """Flash a glow effect on the card.

        Args:
            color: Glow color variant (success, error, warning, primary).
            duration: Total duration of the flash in milliseconds.
        """
        theme = get_theme_manager()

        # Get glow color
        glow_colors = {
            "success": theme.colors.glow_success,
            "error": theme.colors.glow_error,
            "warning": theme.colors.glow_warning,
            "primary": theme.colors.glow_primary,
        }
        glow_color = glow_colors.get(color, theme.colors.glow_primary)

        # Parse rgba color
        # Format: rgba(r, g, b, a)
        import re
        match = re.match(r'rgba\((\d+),\s*(\d+),\s*(\d+),\s*([\d.]+)\)', glow_color)
        if not match:
            return

        r, g, b, a = int(match.group(1)), int(match.group(2)), int(match.group(3)), float(match.group(4))

        # Clean up existing effect before creating new one to prevent memory accumulation
        current_effect = self.graphicsEffect()
        if current_effect is not None:
            current_effect.deleteLater()
        self._shadow_effect = None  # Mark old shadow as invalid

        # Create glow effect
        glow_effect = QtWidgets.QGraphicsDropShadowEffect(self)
        glow_effect.setOffset(0, 0)
        glow_effect.setBlurRadius(25)
        glow_effect.setColor(QtGui.QColor(r, g, b, int(255 * a)))
        self.setGraphicsEffect(glow_effect)

        # Animate back to normal shadow after duration
        def restore_shadow():
            try:
                self._apply_elevation(self._elevation)
            except RuntimeError:
                pass  # Widget may have been deleted

        QtCore.QTimer.singleShot(duration, restore_shadow)


# Backward compatibility alias
MaterialCard = GlassCard
