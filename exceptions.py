"""Custom exceptions for the transcription application.

Only includes exceptions that are actually raised in the codebase.
"""

from __future__ import annotations


class TranscriberError(Exception):
    """Base exception for all transcriber errors."""


class AudioDownloadError(TranscriberError):
    """Failed to download audio from YouTube."""


class FileValidationError(TranscriberError):
    """File validation failed (size, format, etc.)."""
