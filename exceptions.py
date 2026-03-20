"""Custom exceptions for YouTube Transcriber application.

Only includes exceptions that are actually raised in the codebase.
"""

from __future__ import annotations


class TranscriberError(Exception):
    """Base exception for all transcriber errors."""
    pass


class AudioDownloadError(TranscriberError):
    """Failed to download audio from YouTube."""
    pass


class FileValidationError(TranscriberError):
    """File validation failed (size, format, etc.)."""
    pass
