"""Responsive layout components for YouTube Transcriber."""

from __future__ import annotations

from enum import Enum, auto
from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets


class Breakpoint(Enum):
    """Screen size breakpoints for responsive design."""

    COMPACT = auto()    # < 800px width
    MEDIUM = auto()     # 800-1280px
    EXPANDED = auto()   # 1280-1920px
    LARGE = auto()      # > 1920px

    @classmethod
    def from_width(cls, width: int) -> "Breakpoint":
        """Determine breakpoint from window width.

        Args:
            width: Window width in pixels.

        Returns:
            Appropriate breakpoint for the width.
        """
        if width < 800:
            return cls.COMPACT
        elif width < 1280:
            return cls.MEDIUM
        elif width < 1920:
            return cls.EXPANDED
        else:
            return cls.LARGE


class ResponsiveSplitter(QtWidgets.QSplitter):
    """A splitter that adjusts ratios based on window size.

    Automatically adjusts panel sizes when the window is resized
    to maintain optimal layout at different screen sizes.
    """

    # Splitter ratios for each breakpoint
    BREAKPOINT_RATIOS = {
        Breakpoint.COMPACT: [0.30, 0.10, 0.60],    # Minimize preview/progress
        Breakpoint.MEDIUM: [0.35, 0.15, 0.50],     # Balanced
        Breakpoint.EXPANDED: [0.40, 0.20, 0.40],   # Full features
        Breakpoint.LARGE: [0.35, 0.15, 0.50],      # More transcript space
    }

    breakpointChanged = QtCore.pyqtSignal(Breakpoint)

    def __init__(
        self,
        orientation: QtCore.Qt.Orientation = QtCore.Qt.Orientation.Vertical,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        """Initialize the ResponsiveSplitter.

        Args:
            orientation: Splitter orientation.
            parent: Parent widget.
        """
        super().__init__(orientation, parent)

        self._current_breakpoint: Optional[Breakpoint] = None
        self._user_adjusted = False
        self._last_user_sizes: Optional[list[int]] = None

        # Default splitter settings
        self.setChildrenCollapsible(False)
        self.setHandleWidth(6)
        self.setStyleSheet("QSplitter::handle { background-color: #2c3646; }")

        # Connect to splitter moved signal to detect user adjustments
        self.splitterMoved.connect(self._on_user_adjusted)

    def _on_user_adjusted(self) -> None:
        """Handle user manually adjusting the splitter."""
        self._user_adjusted = True
        self._last_user_sizes = self.sizes()

    def resizeEvent(self, a0: QtGui.QResizeEvent | None) -> None:  # noqa: N803
        """Handle resize - adjust splitter ratios if needed."""
        super().resizeEvent(a0)

        if a0 is None:
            return

        # Get window width for breakpoint calculation
        window = self.window()
        if window:
            width = window.width()
            new_breakpoint = Breakpoint.from_width(width)

            if new_breakpoint != self._current_breakpoint:
                self._current_breakpoint = new_breakpoint

                # Only auto-adjust if user hasn't manually adjusted
                if not self._user_adjusted:
                    self._apply_breakpoint_ratios(new_breakpoint)

                self.breakpointChanged.emit(new_breakpoint)

    def _apply_breakpoint_ratios(self, breakpoint: Breakpoint) -> None:
        """Apply splitter ratios for the given breakpoint.

        Args:
            breakpoint: Target breakpoint.
        """
        ratios = self.BREAKPOINT_RATIOS.get(
            breakpoint, self.BREAKPOINT_RATIOS[Breakpoint.EXPANDED]
        )

        # Calculate sizes based on total available space
        if self.orientation() == QtCore.Qt.Orientation.Vertical:
            total_size = self.height()
        else:
            total_size = self.width()

        if total_size <= 0:
            return

        # Calculate new sizes
        num_widgets = self.count()
        if num_widgets == 0:
            return

        # Use ratios for the first N widgets
        new_sizes = []
        for i in range(num_widgets):
            if i < len(ratios):
                new_sizes.append(int(total_size * ratios[i]))
            else:
                # Distribute remaining space equally
                remaining_ratio = (1.0 - sum(ratios)) / (num_widgets - len(ratios))
                new_sizes.append(int(total_size * remaining_ratio))

        self.setSizes(new_sizes)

    def setDefaultSizes(self, ratios: list[float]) -> None:
        """Set default size ratios.

        Args:
            ratios: List of ratios (should sum to 1.0).
        """
        if self.orientation() == QtCore.Qt.Orientation.Vertical:
            total_size = self.height()
        else:
            total_size = self.width()

        if total_size <= 0:
            total_size = 700  # Fallback

        sizes = [int(total_size * r) for r in ratios]
        self.setSizes(sizes)

    def resetUserAdjustment(self) -> None:
        """Reset user adjustment flag to allow auto-sizing."""
        self._user_adjusted = False
        self._last_user_sizes = None

        if self._current_breakpoint:
            self._apply_breakpoint_ratios(self._current_breakpoint)

    def currentBreakpoint(self) -> Optional[Breakpoint]:
        """Get the current breakpoint."""
        return self._current_breakpoint
