"""Nord Minimal Theme for YouTube Transcriber application.

Clean, modern theme inspired by the Nord color palette.
Removes glassmorphism and complex effects for a cleaner look.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PyQt6 import QtGui, QtWidgets


@dataclass
class ThemeColors:
    """Nord-inspired color definitions for clean, minimal UI."""

    # Polar Night (backgrounds) - darkest to lightest
    background: str = "#2E3440"      # Main background
    surface: str = "#3B4252"         # Cards, panels
    surface_variant: str = "#434C5E" # Hover states
    card: str = "#3B4252"            # Card backgrounds

    # Snow Storm (text) - lightest to darkest
    text_primary: str = "#ECEFF4"    # Main text
    text_secondary: str = "#D8DEE9"  # Secondary text
    text_disabled: str = "#8892A5"   # Disabled text
    text_hint: str = "#8892A5"       # Hints and placeholders

    # Frost (accent colors)
    primary: str = "#88C0D0"         # Primary accent (teal)
    primary_hover: str = "#81A1C1"   # Primary hover

    # Aurora (semantic colors)
    success: str = "#A3BE8C"         # Green
    success_hover: str = "#8FAB7A"
    success_light: str = "#A3BE8C"

    warning: str = "#EBCB8B"         # Yellow
    warning_hover: str = "#D9B86C"

    error: str = "#BF616A"           # Red
    error_hover: str = "#A54E56"
    error_light: str = "#BF616A"

    info: str = "#5E81AC"            # Blue

    # Component-specific
    border: str = "#4C566A"          # Borders
    input_background: str = "#3B4252"
    selection: str = "#5E81AC"

    # Recording state
    recording: str = "#BF616A"
    recording_text: str = "#BF616A"

    # Output areas
    live_preview_bg: str = "#2E3440"
    live_preview_text: str = "#A3BE8C"  # Green text for live preview

    progress_bg: str = "#2E3440"
    progress_text: str = "#88C0D0"      # Teal text for progress

    transcript_bg: str = "#2E3440"
    transcript_text: str = "#ECEFF4"

    # Status bar
    status_bar_bg: str = "#2E3440"
    status_bar_border: str = "#4C566A"

    # Splitter
    splitter_handle: str = "#4C566A"

    # Tooltip
    tooltip_bg: str = "#3B4252"
    tooltip_border: str = "#4C566A"

    # Group box title
    group_title: str = "#88C0D0"

    # Simplified glass/shadow colors (minimal effects)
    glass_bg: str = "rgba(59, 66, 82, 0.95)"
    glass_bg_hover: str = "rgba(67, 76, 94, 0.95)"
    glass_border: str = "rgba(76, 86, 106, 0.5)"
    glass_border_hover: str = "rgba(76, 86, 106, 0.7)"
    glass_highlight: str = "rgba(136, 192, 208, 0.1)"
    glass_shadow: str = "rgba(0, 0, 0, 0.2)"

    # Glow effects (subtle)
    glow_primary: str = "rgba(136, 192, 208, 0.2)"
    glow_success: str = "rgba(163, 190, 140, 0.2)"
    glow_warning: str = "rgba(235, 203, 139, 0.2)"
    glow_error: str = "rgba(191, 97, 106, 0.2)"

    # Loading/skeleton
    skeleton_base: str = "#3B4252"
    skeleton_shimmer: str = "#434C5E"
    skeleton_highlight: str = "#4C566A"

    # Animation timing (in ms)
    anim_fast: int = 100
    anim_normal: int = 200
    anim_slow: int = 300
    anim_stagger: int = 50


# Pre-defined themes
DARK_THEME = ThemeColors()


class ThemeManager:
    """Manages application theming and stylesheet generation."""

    def __init__(self, theme: Optional[ThemeColors] = None):
        """Initialize theme manager.

        Args:
            theme: Theme colors to use. Defaults to dark theme.
        """
        self.colors = theme or DARK_THEME

    def apply_to_app(self, app: QtWidgets.QApplication) -> None:
        """Apply theme to the entire application.

        Args:
            app: QApplication instance to theme.
        """
        app.setStyle("Fusion")
        self._configure_palette(app)

    def _configure_palette(self, app: QtWidgets.QApplication) -> None:
        """Configure the application palette with theme colors."""
        c = self.colors

        palette = QtGui.QPalette()

        # Window
        palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(c.background))
        palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(c.text_primary))

        # Base (input backgrounds)
        palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(c.input_background))
        palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(c.surface_variant))

        # Text
        palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(c.text_primary))
        palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor(c.text_primary))
        palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtGui.QColor(c.text_primary))

        # Buttons
        palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(c.surface_variant))
        palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(c.text_primary))

        # Highlights
        palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(c.primary))
        palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor("#ffffff"))

        # Misc
        palette.setColor(QtGui.QPalette.ColorRole.BrightText, QtGui.QColor(c.error_light))

        app.setPalette(palette)

    def get_main_stylesheet(self) -> str:
        """Generate the main application stylesheet."""
        c = self.colors

        return f"""
            QWidget {{
                background-color: {c.background};
                color: {c.text_primary};
                font-family: "Segoe UI", "Inter", "Helvetica Neue", "Arial", sans-serif;
                font-size: 11pt;
            }}

            QGroupBox {{
                border: 1px solid {c.border};
                border-radius: 8px;
                margin-top: 16px;
                padding: 12px;
                background-color: {c.surface};
            }}

            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                font-weight: 600;
                color: {c.group_title};
                background-color: transparent;
            }}

            QLineEdit, QComboBox, QTextEdit, QPlainTextEdit {{
                background-color: {c.input_background};
                border: 1px solid {c.border};
                border-radius: 6px;
                padding: 8px;
                selection-background-color: {c.selection};
            }}

            QLineEdit:focus, QComboBox:focus, QTextEdit:focus {{
                border-color: {c.primary};
            }}

            QPlainTextEdit#ProgressOutput {{
                background-color: {c.progress_bg};
                color: {c.progress_text};
                font-family: "JetBrains Mono", "Fira Code", "Consolas", monospace;
                font-size: 10pt;
                border: 1px solid {c.border};
                border-radius: 6px;
            }}

            QTextEdit#LivePreview {{
                background-color: {c.live_preview_bg};
                color: {c.live_preview_text};
                font-family: "JetBrains Mono", "Fira Code", "Consolas", monospace;
                font-size: 11pt;
                border: 1px solid {c.border};
                border-radius: 6px;
            }}

            QTextEdit#TranscriptOutput {{
                background-color: {c.transcript_bg};
                color: {c.transcript_text};
                font-size: 11pt;
                border: 1px solid {c.border};
                border-radius: 6px;
            }}

            QStatusBar {{
                background-color: {c.status_bar_bg};
                border-top: 1px solid {c.status_bar_border};
                padding: 4px;
            }}

            QToolTip {{
                background-color: {c.tooltip_bg};
                color: {c.text_primary};
                border: 1px solid {c.tooltip_border};
                border-radius: 4px;
                padding: 4px 8px;
            }}

            QSplitter::handle {{
                background-color: {c.splitter_handle};
                margin: 2px;
            }}

            QSplitter::handle:horizontal {{
                width: 4px;
            }}

            QSplitter::handle:vertical {{
                height: 4px;
            }}

            QSpinBox {{
                background-color: {c.input_background};
                border: 1px solid {c.border};
                border-radius: 6px;
                padding: 6px;
            }}

            QCheckBox {{
                spacing: 8px;
            }}

            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 2px solid {c.border};
                background-color: {c.input_background};
            }}

            QCheckBox::indicator:checked {{
                background-color: {c.primary};
                border-color: {c.primary};
            }}

            QTabWidget::pane {{
                border: 1px solid {c.border};
                border-radius: 6px;
                background-color: {c.surface};
                padding: 8px;
            }}

            QTabBar::tab {{
                background-color: {c.surface};
                border: 1px solid {c.border};
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                padding: 8px 16px;
                margin-right: 2px;
            }}

            QTabBar::tab:selected {{
                background-color: {c.primary};
                color: white;
            }}

            QTabBar::tab:hover:!selected {{
                background-color: {c.surface_variant};
            }}

            QPushButton {{
                background-color: {c.surface_variant};
                color: {c.text_primary};
                border: 1px solid {c.border};
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 500;
            }}

            QPushButton:hover {{
                background-color: {c.primary};
                color: white;
                border-color: {c.primary};
            }}

            QPushButton:pressed {{
                background-color: {c.primary_hover};
            }}

            QPushButton:disabled {{
                background-color: {c.surface};
                color: {c.text_disabled};
                border-color: {c.border};
            }}

            QComboBox::drop-down {{
                border: none;
                padding-right: 8px;
            }}

            QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid {c.text_secondary};
                margin-right: 8px;
            }}

            QScrollBar:vertical {{
                background-color: {c.background};
                width: 10px;
                border-radius: 5px;
            }}

            QScrollBar::handle:vertical {{
                background-color: {c.border};
                border-radius: 5px;
                min-height: 30px;
            }}

            QScrollBar::handle:vertical:hover {{
                background-color: {c.primary};
            }}

            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}

            QScrollBar:horizontal {{
                background-color: {c.background};
                height: 10px;
                border-radius: 5px;
            }}

            QScrollBar::handle:horizontal {{
                background-color: {c.border};
                border-radius: 5px;
                min-width: 30px;
            }}

            QScrollBar::handle:horizontal:hover {{
                background-color: {c.primary};
            }}

            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0px;
            }}
        """

    def get_button_style(
        self,
        variant: str = "primary",
        background: Optional[str] = None,
        hover: Optional[str] = None
    ) -> str:
        """Generate button stylesheet.

        Args:
            variant: Button variant (primary, success, warning, error, secondary)
            background: Optional custom background color
            hover: Optional custom hover color

        Returns:
            QPushButton stylesheet string
        """
        c = self.colors

        # Default colors by variant
        variants = {
            "primary": (c.primary, c.primary_hover),
            "success": (c.success, c.success_hover),
            "warning": (c.warning, c.warning_hover),
            "error": (c.error, c.error_hover),
            "secondary": (c.surface_variant, c.surface),
        }

        bg, hv = variants.get(variant, variants["primary"])
        bg = background or bg
        hv = hover or hv
        text_color = "white" if variant != "secondary" else c.text_primary

        return f"""
            QPushButton {{
                background-color: {bg};
                color: {text_color};
                border: none;
                border-radius: 6px;
                padding: 10px 18px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {hv};
            }}
            QPushButton:pressed {{
                background-color: {hv};
            }}
            QPushButton:disabled {{
                background-color: {c.surface};
                color: {c.text_disabled};
            }}
        """

    def get_title_style(self) -> str:
        """Get style for main title label."""
        return f"font-size: 20px; font-weight: 600; color: {self.colors.primary};"

    def get_status_style(self, status_type: str = "success") -> str:
        """Get style for status labels.

        Args:
            status_type: Type of status (success, warning, error, info)
        """
        colors = {
            "success": self.colors.success,
            "warning": self.colors.warning,
            "error": self.colors.error,
            "info": self.colors.info,
        }
        color = colors.get(status_type, self.colors.success)
        return f"color: {color}; font-weight: 600;"

    def get_gpu_status_style(self, has_gpu: bool) -> str:
        """Get style for GPU status label."""
        color = self.colors.primary if has_gpu else self.colors.warning
        return f"color: {color}; font-weight: 500; padding: 0 10px;"

    def get_recording_status_style(self, is_recording: bool) -> str:
        """Get style for recording status label."""
        if is_recording:
            return f"color: {self.colors.recording}; font-size: 10pt; font-weight: 600;"
        return f"color: {self.colors.text_disabled}; font-size: 10pt;"

    def get_glass_card_style(self, is_hovered: bool = False) -> str:
        """Get card stylesheet (simplified, no glass effect).

        Args:
            is_hovered: Whether card is in hover state.

        Returns:
            QSS stylesheet for card.
        """
        c = self.colors
        bg = c.surface_variant if is_hovered else c.surface

        return f"""
            GlassCard {{
                background-color: {bg};
                border: 1px solid {c.border};
                border-radius: 8px;
            }}
        """

    def get_glow_color(self, variant: str) -> str:
        """Get glow color for a variant.

        Args:
            variant: Button/card variant.

        Returns:
            RGBA color string for glow effect.
        """
        c = self.colors
        glows = {
            "primary": c.glow_primary,
            "success": c.glow_success,
            "warning": c.glow_warning,
            "error": c.glow_error,
        }
        return glows.get(variant, c.glow_primary)


# Global theme manager instance
_theme_manager: Optional[ThemeManager] = None


def get_theme_manager() -> ThemeManager:
    """Get the global theme manager instance."""
    global _theme_manager
    if _theme_manager is None:
        _theme_manager = ThemeManager(DARK_THEME)
    return _theme_manager


def set_theme(theme: ThemeColors) -> None:
    """Set a new theme globally."""
    global _theme_manager
    _theme_manager = ThemeManager(theme)
