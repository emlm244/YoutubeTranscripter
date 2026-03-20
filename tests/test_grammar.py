"""Tests for grammar post-processor with mocked GECToR."""

from __future__ import annotations

import sys
import types

import grammar_postprocessor as gp

from config import GrammarConfig
from grammar_postprocessor import (
    GrammarPostProcessor,
    check_grammar_status,
    post_process_grammar,
)


class TestGrammarPostProcessor:
    """Test GrammarPostProcessor class."""

    def test_disabled_returns_original(self):
        config = GrammarConfig(enabled=False)
        processor = GrammarPostProcessor(config)
        text, enhanced = processor.process_text("hello world")
        assert text == "hello world"
        assert not enhanced

    def test_empty_text_returns_original(self):
        processor = GrammarPostProcessor()
        text, enhanced = processor.process_text("")
        assert text == ""
        assert not enhanced

    def test_status_when_disabled(self):
        config = GrammarConfig(enabled=False)
        processor = GrammarPostProcessor(config)
        assert processor.get_status() == "Disabled"


class TestPostProcessGrammar:
    """Test convenience function."""

    def test_with_disabled_config(self):
        config = GrammarConfig(enabled=False)
        text, segments, enhanced = post_process_grammar(
            text="hello", config=config
        )
        assert text == "hello"
        assert not enhanced

    def test_with_segments_disabled(self):
        config = GrammarConfig(enabled=False)
        segs = [{"start": 0, "end": 1, "text": "hello"}]
        text, result_segs, enhanced = post_process_grammar(
            text="hello", segments_data=segs, config=config
        )
        assert not enhanced
        assert len(result_segs) == 1


class TestCheckGrammarStatus:
    """Test check_grammar_status function."""

    def test_returns_tuple(self):
        available, status = check_grammar_status()
        assert isinstance(available, bool)
        assert isinstance(status, str)


class TestGrammarRegressions:
    """Regression tests for backend lifecycle and caching behavior."""

    def test_languagetool_cache_is_per_language(self, monkeypatch):
        created_languages = []

        class FakeLanguageTool:
            def __init__(self, language):
                created_languages.append(language)
                self.language = language

        fake_module = types.SimpleNamespace(
            LanguageTool=FakeLanguageTool,
            utils=types.SimpleNamespace(correct=lambda text, _matches: text),
        )
        monkeypatch.setitem(sys.modules, "language_tool_python", fake_module)
        monkeypatch.setattr(gp, "_lt_instances", {})
        monkeypatch.setattr(
            gp,
            "get_languagetool_runtime_status",
            lambda: gp.LanguageToolRuntimeStatus(True, "ready"),
        )

        tool_en_1 = gp._get_languagetool("en-US")
        tool_en_2 = gp._get_languagetool("en-US")
        tool_de = gp._get_languagetool("de-DE")

        assert tool_en_1 is tool_en_2
        assert tool_de is not tool_en_1
        assert created_languages == ["en-US", "de-DE"]

    def test_gector_initialize_retries_after_transient_failure(self, monkeypatch):
        class FakeCuda:
            @staticmethod
            def is_available():
                return False

        fake_torch = types.SimpleNamespace(cuda=FakeCuda())
        fake_transformers = types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda _: object())
        )
        fake_gector = types.SimpleNamespace(load_verb_dict=lambda _: ({}, {}))

        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
        monkeypatch.setitem(sys.modules, "gector", fake_gector)

        class FakeModel:
            def cpu(self):
                return self

            def cuda(self):
                return self

        calls = {"count": 0}

        def fake_load_gector_model(_model_id):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("transient model load error")
            return FakeModel()

        manager = gp._GECToRManager()
        monkeypatch.setattr(manager, "_ensure_verb_dict", lambda: gp.DATA_DIR / "verb-form-vocab.txt")
        monkeypatch.setattr(gp, "_load_gector_model", fake_load_gector_model)

        first = manager.initialize("fake/model")
        second = manager.initialize("fake/model")

        assert first is False
        assert second is True
        assert calls["count"] == 2
        assert manager.is_available() is True
        assert manager.get_error() is None
