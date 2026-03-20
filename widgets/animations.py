"""Animation utilities for modern UI effects.

Provides reusable animation classes for fade, slide, scale, and pulse effects.
All animations use QPropertyAnimation for hardware-accelerated smoothness.
"""

from __future__ import annotations

from typing import Callable, Optional, List

from PyQt6 import QtCore, QtWidgets

try:
    from PyQt6 import sip  # type: ignore[attr-defined]
except ImportError:
    import sip  # type: ignore[import-not-found]


class FadeEffect:
    """Fade animation wrapper using QGraphicsOpacityEffect."""

    def __init__(self, widget: QtWidgets.QWidget):
        """Initialize fade effect for a widget.

        Args:
            widget: Widget to apply fade effect to.
        """
        self._widget = widget
        self._effect: Optional[QtWidgets.QGraphicsOpacityEffect] = None
        self._animation: Optional[QtCore.QPropertyAnimation] = None
        self._setup_effect()

    def _setup_effect(self) -> None:
        """Create or recreate the opacity effect and animation."""
        self._effect = QtWidgets.QGraphicsOpacityEffect(self._widget)
        self._effect.setOpacity(1.0)
        self._widget.setGraphicsEffect(self._effect)

        self._animation = QtCore.QPropertyAnimation(self._effect, b"opacity")
        self._animation.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)

    def _ensure_effect_valid(self) -> None:
        """Ensure the effect hasn't been deleted, recreate if necessary."""
        if self._effect is None or sip.isdeleted(self._effect):
            self._setup_effect()

    def _require_effect(self) -> tuple[QtWidgets.QGraphicsOpacityEffect, QtCore.QPropertyAnimation]:
        self._ensure_effect_valid()
        assert self._effect is not None
        assert self._animation is not None
        return self._effect, self._animation

    def fade_in(
        self,
        duration: int = 400,
        start_opacity: float = 0.0,
        end_opacity: float = 1.0,
        on_finished: Optional[Callable] = None,
    ) -> None:
        """Fade the widget in.

        Args:
            duration: Animation duration in milliseconds.
            start_opacity: Starting opacity (0.0-1.0).
            end_opacity: Ending opacity (0.0-1.0).
            on_finished: Optional callback when animation completes.
        """
        effect, animation = self._require_effect()
        animation.stop()
        effect.setOpacity(start_opacity)
        animation.setDuration(duration)
        animation.setStartValue(start_opacity)
        animation.setEndValue(end_opacity)

        if on_finished:
            animation.finished.connect(on_finished)

        animation.start()

    def fade_out(
        self,
        duration: int = 300,
        on_finished: Optional[Callable] = None,
    ) -> None:
        """Fade the widget out.

        Args:
            duration: Animation duration in milliseconds.
            on_finished: Optional callback when animation completes.
        """
        effect, animation = self._require_effect()
        animation.stop()
        animation.setDuration(duration)
        animation.setStartValue(effect.opacity())
        animation.setEndValue(0.0)

        if on_finished:
            animation.finished.connect(on_finished)

        animation.start()

    def set_opacity(self, opacity: float) -> None:
        """Set opacity immediately without animation."""
        effect, _ = self._require_effect()
        effect.setOpacity(opacity)

    @property
    def opacity(self) -> float:
        """Get current opacity."""
        effect, _ = self._require_effect()
        return effect.opacity()


class ScaleAnimation:
    """Scale animation using geometry transformation."""

    def __init__(self, widget: QtWidgets.QWidget):
        """Initialize scale animation for a widget.

        Args:
            widget: Widget to animate.
        """
        self._widget = widget
        self._animation = QtCore.QPropertyAnimation(widget, b"geometry")
        self._animation.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        self._original_geometry: Optional[QtCore.QRect] = None

    def scale_in(
        self,
        duration: int = 400,
        from_scale: float = 0.95,
        on_finished: Optional[Callable] = None,
    ) -> None:
        """Scale widget from smaller size to full size.

        Args:
            duration: Animation duration in milliseconds.
            from_scale: Starting scale factor (0.0-1.0).
            on_finished: Optional callback when animation completes.
        """
        self._animation.stop()

        # Store original geometry
        self._original_geometry = self._widget.geometry()
        target = self._original_geometry

        # Calculate scaled starting geometry (centered)
        width_diff = int(target.width() * (1 - from_scale))
        height_diff = int(target.height() * (1 - from_scale))

        start_rect = QtCore.QRect(
            target.x() + width_diff // 2,
            target.y() + height_diff // 2,
            int(target.width() * from_scale),
            int(target.height() * from_scale),
        )

        self._animation.setDuration(duration)
        self._animation.setStartValue(start_rect)
        self._animation.setEndValue(target)

        if on_finished:
            self._animation.finished.connect(on_finished)

        self._animation.start()

    def scale_bounce(
        self,
        duration: int = 300,
        scale: float = 1.05,
        on_finished: Optional[Callable] = None,
    ) -> None:
        """Quick scale bounce effect (scale up then back to normal).

        Args:
            duration: Total animation duration in milliseconds.
            scale: Peak scale factor.
            on_finished: Optional callback when animation completes.
        """
        self._animation.stop()

        original = self._widget.geometry()

        # Calculate scaled geometry (centered)
        width_diff = int(original.width() * (scale - 1))
        height_diff = int(original.height() * (scale - 1))

        scaled_rect = QtCore.QRect(
            original.x() - width_diff // 2,
            original.y() - height_diff // 2,
            int(original.width() * scale),
            int(original.height() * scale),
        )

        # Use elastic curve for bounce effect
        self._animation.setEasingCurve(QtCore.QEasingCurve.Type.OutElastic)
        self._animation.setDuration(duration)
        self._animation.setStartValue(scaled_rect)
        self._animation.setEndValue(original)

        if on_finished:
            self._animation.finished.connect(on_finished)

        self._animation.start()


class RevealAnimation:
    """Combined fade + scale reveal animation."""

    def __init__(self, widget: QtWidgets.QWidget):
        """Initialize reveal animation.

        Args:
            widget: Widget to reveal.
        """
        self._widget = widget
        self._fade = FadeEffect(widget)
        self._scale = ScaleAnimation(widget)

    def reveal(
        self,
        duration: int = 450,
        from_scale: float = 0.95,
        delay: int = 0,
        on_finished: Optional[Callable] = None,
    ) -> None:
        """Reveal widget with combined fade and scale.

        Args:
            duration: Animation duration in milliseconds.
            from_scale: Starting scale factor.
            delay: Delay before starting animation.
            on_finished: Optional callback when animation completes.
        """
        # Set initial state
        self._fade.set_opacity(0.0)

        def start_animation():
            self._fade.fade_in(duration=duration)
            self._scale.scale_in(duration=duration, from_scale=from_scale)

            if on_finished:
                # Connect to scale animation finished (both should finish together)
                self._scale._animation.finished.connect(on_finished)

        if delay > 0:
            QtCore.QTimer.singleShot(delay, start_animation)
        else:
            start_animation()


class AnimationManager:
    """Central coordinator for staggered and sequential animations."""

    def __init__(self):
        """Initialize animation manager."""
        self._animations: List[RevealAnimation] = []

    def staggered_reveal(
        self,
        widgets: List[QtWidgets.QWidget],
        duration: int = 450,
        stagger_delay: int = 80,
        from_scale: float = 0.95,
        on_all_finished: Optional[Callable] = None,
    ) -> None:
        """Reveal multiple widgets with staggered timing.

        Args:
            widgets: List of widgets to reveal.
            duration: Duration for each widget's animation.
            stagger_delay: Delay between each widget's start.
            from_scale: Starting scale factor for each widget.
            on_all_finished: Callback when all animations complete.
        """
        self._animations = []

        for i, widget in enumerate(widgets):
            reveal = RevealAnimation(widget)
            self._animations.append(reveal)

            delay = i * stagger_delay
            is_last = i == len(widgets) - 1

            reveal.reveal(
                duration=duration,
                from_scale=from_scale,
                delay=delay,
                on_finished=on_all_finished if is_last else None,
            )
