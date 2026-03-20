"""Additional unit coverage for grammar_postprocessor behaviors."""

from __future__ import annotations

import builtins
import json
import sys
import types

import pytest

import grammar_postprocessor as gp
from config import GrammarConfig


@pytest.fixture(autouse=True)
def _reset_grammar_globals(monkeypatch):
    monkeypatch.setattr(gp, "_lt_instances", {})
    monkeypatch.setattr(gp._GECToRManager, "_instance", None)


class _FakeModel:
    def __init__(self):
        self.on_device = "cpu"

    def cpu(self):
        self.on_device = "cpu"
        return self

    def cuda(self):
        self.on_device = "cuda"
        return self


def _fake_torch(cuda_available: bool):
    class _Cuda:
        @staticmethod
        def is_available():
            return cuda_available

        @staticmethod
        def empty_cache():
            return None

    return types.SimpleNamespace(cuda=_Cuda())


def test_get_languagetool_importerror(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "language_tool_python":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert gp._get_languagetool("en-US") is None


def test_get_languagetool_exception_path(monkeypatch):
    class _BrokenLT:
        def __init__(self, _lang):
            raise RuntimeError("boom")

    monkeypatch.setitem(sys.modules, "language_tool_python", types.SimpleNamespace(LanguageTool=_BrokenLT))
    assert gp._get_languagetool("en-US") is None


def test_get_languagetool_uses_default_language_and_double_check_lock(monkeypatch):
    created = {"count": 0}

    class _LT:
        def __init__(self, _lang):
            created["count"] += 1

    monkeypatch.setitem(sys.modules, "language_tool_python", types.SimpleNamespace(LanguageTool=_LT))
    monkeypatch.setattr(gp, "_lt_instances", {"en-US": object()})
    assert gp._get_languagetool("") is not None
    assert created["count"] == 0


def test_ensure_verb_dict_existing(tmp_path, monkeypatch):
    monkeypatch.setattr(gp, "DATA_DIR", tmp_path)
    monkeypatch.setattr(gp, "BUNDLED_VERB_DICT_PATH", tmp_path / "_missing_bundled_dict.txt")
    existing = tmp_path / "verb-form-vocab.txt"
    existing.write_text("ok", encoding="utf-8")
    manager = gp._GECToRManager()
    assert manager._ensure_verb_dict() == existing


def test_ensure_verb_dict_download_success(tmp_path, monkeypatch):
    monkeypatch.setattr(gp, "DATA_DIR", tmp_path)
    monkeypatch.setattr(gp, "BUNDLED_VERB_DICT_PATH", tmp_path / "_missing_bundled_dict.txt")

    def fake_urlretrieve(_url, destination):
        destination.write_text("downloaded", encoding="utf-8")

    monkeypatch.setattr(gp.urllib.request, "urlretrieve", fake_urlretrieve)
    manager = gp._GECToRManager()
    path = manager._ensure_verb_dict()
    assert path is not None
    assert path.exists()


def test_ensure_verb_dict_download_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(gp, "DATA_DIR", tmp_path)
    monkeypatch.setattr(gp, "BUNDLED_VERB_DICT_PATH", tmp_path / "_missing_bundled_dict.txt")
    monkeypatch.setattr(gp.urllib.request, "urlretrieve", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("net")))
    manager = gp._GECToRManager()
    assert manager._ensure_verb_dict() is None


def test_initialize_returns_false_when_verb_dict_unavailable(monkeypatch):
    manager = gp._GECToRManager()
    monkeypatch.setattr(manager, "_ensure_verb_dict", lambda: None)
    assert manager.initialize("model") is False
    assert manager.get_error() == "Verb dictionary unavailable"


def test_initialize_importerror_branch(monkeypatch):
    real_import = builtins.__import__
    manager = gp._GECToRManager()
    monkeypatch.setattr(manager, "_ensure_verb_dict", lambda: gp.DATA_DIR / "verb-form-vocab.txt")

    def fake_import(name, *args, **kwargs):
        if name in {"transformers", "gector"}:
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available=False))
    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert manager.initialize("model") is False
    assert "GECToR not installed" in (manager.get_error() or "")


def test_initialize_returns_false_when_torch_unavailable(monkeypatch):
    manager = gp._GECToRManager()
    monkeypatch.setattr(manager, "_ensure_verb_dict", lambda: gp.DATA_DIR / "verb-form-vocab.txt")
    monkeypatch.setattr(gp, "_load_torch_module", lambda **_kwargs: None)

    assert manager.initialize("model") is False
    assert manager.get_error() == "PyTorch unavailable"


def test_initialize_cuda_sets_cuda_device(monkeypatch):
    manager = gp._GECToRManager()
    fake_model = _FakeModel()
    monkeypatch.setattr(manager, "_ensure_verb_dict", lambda: gp.DATA_DIR / "verb-form-vocab.txt")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available=True))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda _m: object())),
    )
    monkeypatch.setitem(sys.modules, "gector", types.SimpleNamespace(load_verb_dict=lambda _p: ({}, {})))
    monkeypatch.setattr(gp, "_load_gector_model", lambda _m: fake_model)

    assert manager.initialize("model") is True
    assert manager.device == "cuda"
    assert fake_model.on_device == "cuda"


def test_initialize_returns_immediately_when_already_initialized():
    manager = gp._GECToRManager()
    manager._initialized = True
    assert manager.initialize("model") is True


def test_unload_returns_when_not_initialized():
    manager = gp._GECToRManager()
    manager._initialized = False
    manager.unload()
    assert manager.is_available() is False


def test_unload_clears_state_and_calls_cuda_cache(monkeypatch):
    empty_calls = {"count": 0}

    class _Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def empty_cache():
            empty_calls["count"] += 1

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=_Cuda()))

    manager = gp._GECToRManager()
    manager.model = _FakeModel().cuda()
    manager.tokenizer = object()
    manager.encode = {"a": 1}
    manager.decode = {"b": 2}
    manager.device = "cuda"
    manager._initialized = True

    manager.unload()

    assert manager.model is None
    assert manager.tokenizer is None
    assert manager.encode is None
    assert manager.decode is None
    assert manager.device is None
    assert manager.is_available() is False
    assert empty_calls["count"] == 1


def test_unload_handles_cleanup_exception(monkeypatch):
    manager = gp._GECToRManager()
    manager._initialized = True
    manager.model = object()
    manager.device = "cuda"

    def broken_cpu():
        raise RuntimeError("boom")

    manager.model = types.SimpleNamespace(cpu=broken_cpu)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available=True))
    manager.unload()
    assert manager.is_available() is True


def test_correct_returns_input_when_uninitialized():
    manager = gp._GECToRManager()
    sentences = ["a", "b"]
    assert manager.correct(sentences) == sentences


def test_correct_runtime_oom_raises_and_clears_cache(monkeypatch):
    empty_calls = {"count": 0}

    class _Cuda:
        @staticmethod
        def empty_cache():
            empty_calls["count"] += 1

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=_Cuda()))
    monkeypatch.setitem(
        sys.modules,
        "gector",
        types.SimpleNamespace(predict=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("out of memory"))),
    )

    manager = gp._GECToRManager()
    manager._initialized = True
    manager.model = object()
    manager.tokenizer = object()
    manager.encode = {}
    manager.decode = {}

    with pytest.raises(RuntimeError):
        manager.correct(["hello"])
    assert empty_calls["count"] == 1


def test_correct_success_path(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "gector",
        types.SimpleNamespace(predict=lambda *_args, **_kwargs: ["fixed"]),
    )
    manager = gp._GECToRManager()
    manager._initialized = True
    manager.model = object()
    manager.tokenizer = object()
    manager.encode = {}
    manager.decode = {}
    assert manager.correct(["hello"]) == ["fixed"]


def test_correct_runtime_oom_when_empty_cache_fails(monkeypatch):
    class _Cuda:
        @staticmethod
        def empty_cache():
            raise RuntimeError("cache fail")

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=_Cuda()))
    monkeypatch.setitem(
        sys.modules,
        "gector",
        types.SimpleNamespace(predict=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("out of memory"))),
    )
    manager = gp._GECToRManager()
    manager._initialized = True
    manager.model = object()
    manager.tokenizer = object()
    manager.encode = {}
    manager.decode = {}
    with pytest.raises(RuntimeError):
        manager.correct(["hello"])


def test_correct_generic_exception_returns_original(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "gector",
        types.SimpleNamespace(predict=lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("other"))),
    )
    manager = gp._GECToRManager()
    manager._initialized = True
    manager.model = object()
    manager.tokenizer = object()
    manager.encode = {}
    manager.decode = {}
    assert manager.correct(["hello"]) == ["hello"]


def test_process_text_fallback_to_languagetool(monkeypatch):
    processor = gp.GrammarPostProcessor(GrammarConfig(backend="auto"))
    processor._active_backend = "gector"
    monkeypatch.setattr(processor, "_correct_with_gector", lambda _text: (_ for _ in ()).throw(RuntimeError("fail")))
    monkeypatch.setattr(gp, "_get_languagetool", lambda _lang: object())
    monkeypatch.setattr(processor, "_correct_with_languagetool", lambda text: (f"{text} fixed", True))
    out_text, enhanced = processor.process_text("hello")
    assert out_text == "hello fixed"
    assert enhanced is True


def test_process_text_no_backend(monkeypatch):
    processor = gp.GrammarPostProcessor(GrammarConfig(backend="gector"))
    monkeypatch.setattr(processor, "_ensure_backend", lambda: False)
    out_text, enhanced = processor.process_text("hello")
    assert out_text == "hello"
    assert enhanced is False


def test_process_text_languagetool_path(monkeypatch):
    processor = gp.GrammarPostProcessor(GrammarConfig(backend="languagetool"))
    processor._active_backend = "languagetool"
    monkeypatch.setattr(processor, "_correct_with_languagetool", lambda text: (f"{text} lt", True))
    out_text, enhanced = processor.process_text("hello")
    assert out_text == "hello lt"
    assert enhanced is True


def test_process_text_fallback_failure_returns_original(monkeypatch):
    processor = gp.GrammarPostProcessor(GrammarConfig(backend="auto"))
    processor._active_backend = "gector"
    monkeypatch.setattr(processor, "_correct_with_gector", lambda _text: (_ for _ in ()).throw(RuntimeError("fail")))
    monkeypatch.setattr(gp, "_get_languagetool", lambda _lang: object())
    monkeypatch.setattr(
        processor,
        "_correct_with_languagetool",
        lambda _text: (_ for _ in ()).throw(RuntimeError("fallback fail")),
    )
    out_text, enhanced = processor.process_text("hello")
    assert out_text == "hello"
    assert enhanced is False


def test_process_segments_fallback_to_languagetool(monkeypatch):
    processor = gp.GrammarPostProcessor(GrammarConfig(backend="auto"))
    processor._active_backend = "gector"
    sample = [{"start": 0, "end": 1, "text": "hello"}]
    monkeypatch.setattr(processor, "_process_segments_gector", lambda _data: (_ for _ in ()).throw(RuntimeError("fail")))
    monkeypatch.setattr(gp, "_get_languagetool", lambda _lang: object())
    monkeypatch.setattr(processor, "_process_segments_languagetool", lambda data: (data, True))
    out, enhanced = processor.process_segments(sample)
    assert out == sample
    assert enhanced is True


def test_process_segments_no_backend(monkeypatch):
    processor = gp.GrammarPostProcessor(GrammarConfig(backend="gector"))
    monkeypatch.setattr(processor, "_ensure_backend", lambda: False)
    sample = [{"start": 0, "end": 1, "text": "hello"}]
    out, enhanced = processor.process_segments(sample)
    assert out == sample
    assert enhanced is False


def test_process_segments_languagetool_direct_path(monkeypatch):
    processor = gp.GrammarPostProcessor(GrammarConfig(backend="languagetool"))
    processor._active_backend = "languagetool"
    sample = [{"start": 0, "end": 1, "text": "hello"}]
    monkeypatch.setattr(processor, "_process_segments_languagetool", lambda data: (data, True))
    out, enhanced = processor.process_segments(sample)
    assert out == sample
    assert enhanced is True


def test_process_segments_fallback_failure_returns_original(monkeypatch):
    processor = gp.GrammarPostProcessor(GrammarConfig(backend="auto"))
    processor._active_backend = "gector"
    sample = [{"start": 0, "end": 1, "text": "hello"}]
    monkeypatch.setattr(processor, "_process_segments_gector", lambda _data: (_ for _ in ()).throw(RuntimeError("fail")))
    monkeypatch.setattr(gp, "_get_languagetool", lambda _lang: object())
    monkeypatch.setattr(
        processor,
        "_process_segments_languagetool",
        lambda _data: (_ for _ in ()).throw(RuntimeError("fallback fail")),
    )
    out, enhanced = processor.process_segments(sample)
    assert out == sample
    assert enhanced is False


def test_correct_with_languagetool_and_segments_paths(monkeypatch):
    class _Tool:
        @staticmethod
        def check(text):
            if "bad" in text:
                return ["m"]
            return []

    fake_lt = types.SimpleNamespace(utils=types.SimpleNamespace(correct=lambda text, _m: text.replace("bad", "good")))
    monkeypatch.setitem(sys.modules, "language_tool_python", fake_lt)
    monkeypatch.setattr(gp, "_get_languagetool", lambda _lang: _Tool())

    processor = gp.GrammarPostProcessor()
    text, enhanced = processor._correct_with_languagetool("bad text")
    assert text == "good text"
    assert enhanced is True

    segments = [{"start": 0, "end": 1, "text": "bad seg"}, {"start": 2, "end": 3, "text": ""}]
    processed, seg_enhanced = processor._process_segments_languagetool(segments)
    assert processed[0]["text"] == "good seg"
    assert seg_enhanced is True


def test_correct_with_languagetool_none_and_no_matches(monkeypatch):
    processor = gp.GrammarPostProcessor()
    monkeypatch.setitem(
        sys.modules,
        "language_tool_python",
        types.SimpleNamespace(utils=types.SimpleNamespace(correct=lambda text, _m: text)),
    )
    monkeypatch.setattr(gp, "_get_languagetool", lambda _lang: None)
    out_text, enhanced = processor._correct_with_languagetool("hello")
    assert out_text == "hello"
    assert enhanced is False

    class _Tool:
        @staticmethod
        def check(_text):
            return []

    monkeypatch.setattr(gp, "_get_languagetool", lambda _lang: _Tool())
    out_text, enhanced = processor._correct_with_languagetool("hello")
    assert out_text == "hello"
    assert enhanced is False


def test_process_segments_gector_no_text_returns_unchanged():
    processor = gp.GrammarPostProcessor()
    sample = [{"start": 0, "end": 1, "text": "   "}]
    out, enhanced = processor._process_segments_gector(sample)
    assert out == [{"start": 0.0, "end": 1.0, "text": ""}]
    assert enhanced is False


def test_process_segments_gector_changes_text(monkeypatch):
    processor = gp.GrammarPostProcessor()
    processor._gector_manager = types.SimpleNamespace(correct=lambda texts, **_kwargs: [f"{t}!" for t in texts])
    sample = [
        {"start": 0, "end": 1, "text": "a"},
        {"start": 1, "end": 2, "text": ""},
        {"start": 2, "end": 3, "text": "b"},
    ]
    out, enhanced = processor._process_segments_gector(sample)
    assert out[0]["text"] == "a!"
    assert out[2]["text"] == "b!"
    assert enhanced is True


def test_process_segments_languagetool_no_tool_and_exception(monkeypatch):
    processor = gp.GrammarPostProcessor()
    monkeypatch.setitem(
        sys.modules,
        "language_tool_python",
        types.SimpleNamespace(utils=types.SimpleNamespace(correct=lambda text, _m: text)),
    )
    sample = [{"start": 0, "end": 1, "text": "x"}]

    monkeypatch.setattr(gp, "_get_languagetool", lambda _lang: None)
    out, enhanced = processor._process_segments_languagetool(sample)
    assert out == sample
    assert enhanced is False

    class _BadTool:
        @staticmethod
        def check(_text):
            raise RuntimeError("boom")

    monkeypatch.setattr(gp, "_get_languagetool", lambda _lang: _BadTool())
    out, enhanced = processor._process_segments_languagetool(sample)
    assert out == sample
    assert enhanced is False


def test_split_sentences_and_correct_with_gector(monkeypatch):
    processor = gp.GrammarPostProcessor(GrammarConfig(gector_batch_size=2, gector_iterations=3))
    processor._gector_manager = types.SimpleNamespace(correct=lambda texts, **_kwargs: ["ok 1", "ok 2"])
    assert processor._split_sentences("One. Two?") == ["One.", "Two?"]
    out_text, enhanced = processor._correct_with_gector("One. Two?")
    assert out_text == "ok 1 ok 2"
    assert enhanced is True


def test_ensure_backend_and_status_paths(monkeypatch):
    manager = types.SimpleNamespace(
        initialize=lambda _model: False,
        get_error=lambda: "boom",
        device="cpu",
    )
    processor = gp.GrammarPostProcessor(GrammarConfig(backend="gector"))
    processor._gector_manager = manager
    assert processor._ensure_backend() is False
    assert processor.is_available() is False
    assert processor.get_status() == "Unavailable: boom"

    processor2 = gp.GrammarPostProcessor(GrammarConfig(backend="auto"))
    processor2._gector_manager = manager
    monkeypatch.setattr(gp, "_get_languagetool", lambda _lang: object())
    assert processor2._ensure_backend() is True
    assert processor2._active_backend == "languagetool"
    assert processor2.get_status() == "LanguageTool"

def test_post_process_and_status_helpers(monkeypatch):
    monkeypatch.setattr(gp.GrammarPostProcessor, "process_text", lambda self, text: (f"{text} ok", True))
    monkeypatch.setattr(gp.GrammarPostProcessor, "process_segments", lambda self, segs: (segs, False))
    text, segments, enhanced = gp.post_process_grammar("hi", None, GrammarConfig())
    assert text == "hi ok"
    assert segments is None
    assert enhanced is True

    out_text, out_segments, out_enhanced = gp.post_process_grammar(
        text="ignored",
        segments_data=[{"start": 0, "end": 1, "text": "a"}],
        config=GrammarConfig(),
    )
    assert out_text == "a"
    assert out_segments is not None
    assert out_enhanced is False

    monkeypatch.setattr(gp.GrammarPostProcessor, "get_status", lambda self: "GECToR (cpu)")
    available, status = gp.check_grammar_status()
    assert available is True
    assert status == "GECToR (cpu)"


def test_unload_gector_wrapper(monkeypatch):
    called = {"count": 0}

    class _Manager:
        def unload(self):
            called["count"] += 1

    monkeypatch.setattr(gp._GECToRManager, "get_instance", classmethod(lambda cls: _Manager()))
    gp.unload_gector()
    assert called["count"] == 1


def test_load_gector_model_with_mocks(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    model_path = tmp_path / "pytorch_model.bin"
    config_path.write_text(
        json.dumps(
            {
                "model_id": "fake/model",
                "has_add_pooling_layer": True,
                "num_labels": 4,
                "d_num_labels": 3,
                "p_dropout": 0.1,
                "label_smoothing": 0.0,
            }
        ),
        encoding="utf-8",
    )
    model_path.write_text("bin", encoding="utf-8")

    class _FakeConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _FakePreTrainedModel:
        def __init__(self, _config):
            return None

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(_model_id):
            return object()

    class _FakeBert:
        def __init__(self):
            self.config = types.SimpleNamespace(vocab_size=10, hidden_size=16)

        def resize_token_embeddings(self, *_args, **_kwargs):
            return None

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(_model_id, **_kwargs):
            return _FakeBert()

    class _FakeTorchNN:
        @staticmethod
        def Linear(*_args, **_kwargs):
            return object()

        @staticmethod
        def Dropout(*_args, **_kwargs):
            return object()

    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(
            nn=_FakeTorchNN(),
            load=lambda *_args, **_kwargs: {"weights": 1},
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch.nn",
        types.SimpleNamespace(
            Linear=_FakeTorchNN.Linear,
            Dropout=_FakeTorchNN.Dropout,
            CrossEntropyLoss=lambda **_kwargs: object(),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "gector",
        types.SimpleNamespace(
            GECToR=type(
                "FakeGECToR",
                (),
                {
                    "post_init": lambda self: None,
                    "tune_bert": lambda self, *_args, **_kwargs: None,
                    "load_state_dict": lambda self, *_args, **_kwargs: None,
                },
            ),
            GECToRConfig=_FakeConfig,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoModel=_FakeAutoModel,
            AutoTokenizer=_FakeAutoTokenizer,
            PreTrainedModel=_FakePreTrainedModel,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(
            hf_hub_download=lambda repo_id, filename: str(config_path if filename == "config.json" else model_path)
        ),
    )

    model = gp._load_gector_model("fake/model")
    assert model is not None
