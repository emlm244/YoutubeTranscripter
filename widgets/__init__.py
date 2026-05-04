"""Modern UI widgets for the transcription app.

Provides glassmorphism cards, material buttons, and animated components.
"""

# Card widgets
from .material_card import GlassCard, MaterialCard  # MaterialCard is alias for GlassCard

# Button widgets
from .material_button import MaterialButton

# Layout widgets
from .responsive_layout import ResponsiveSplitter, Breakpoint

__all__ = [
    # Cards
    "GlassCard",
    "MaterialCard",
    # Buttons
    "MaterialButton",
    # Layout
    "ResponsiveSplitter",
    "Breakpoint",
]
