from __future__ import annotations

import pytest

from exceptions import AudioDownloadError, FileValidationError, TranscriberError


def test_exception_hierarchy():
    assert issubclass(TranscriberError, Exception)
    assert issubclass(AudioDownloadError, TranscriberError)
    assert issubclass(FileValidationError, TranscriberError)


def test_exception_instantiation_preserves_message():
    assert str(TranscriberError("generic failure")) == "generic failure"
    assert str(AudioDownloadError("download failed")) == "download failed"
    assert str(FileValidationError("bad file")) == "bad file"


def test_specific_exceptions_raise_and_catch_via_base_class():
    with pytest.raises(TranscriberError, match="download failed"):
        raise AudioDownloadError("download failed")

    with pytest.raises(TranscriberError, match="bad file"):
        raise FileValidationError("bad file")
