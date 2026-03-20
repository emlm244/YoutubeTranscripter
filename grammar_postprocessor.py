"""
Grammar Post-Processor for transcript correction.

Primary: GECToR (GPU-accelerated, state-of-the-art accuracy)
Fallback: LanguageTool (local, no API key needed)
"""

from __future__ import annotations

from collections.abc import Sequence
import logging
import sys
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, overload

from app_paths import get_resource_root, get_writable_app_data_root
from config import GrammarConfig
from torch_runtime import get_torch, get_torch_import_error
from transcript_types import (
    TranscriptSegmentLike,
    TranscriptSegments,
    coerce_transcript_segments,
    replace_segment_text,
)

logger = logging.getLogger(__name__)

# Directory roots for data files (verb dictionary, etc.)
BUNDLED_DATA_DIR = get_resource_root() / "data"
DATA_DIR = get_writable_app_data_root() / "data"
BUNDLED_VERB_DICT_PATH = BUNDLED_DATA_DIR / "verb-form-vocab.txt"

# Verb dictionary URL for GECToR
VERB_DICT_URL = "https://raw.githubusercontent.com/grammarly/gector/master/data/verb-form-vocab.txt"


# =============================================================================
# LanguageTool Backend (Fallback)
# =============================================================================

_lt_instances: Dict[str, Any] = {}
_lt_lock = threading.Lock()


@dataclass(frozen=True)
class LanguageToolRuntimeStatus:
    """Availability details for the local LanguageTool runtime."""

    available: bool
    detail: str


def _load_torch_module(*, context: str) -> Any | None:
    """Return a loaded torch module when available without crashing startup."""
    torch_module = sys.modules.get("torch")
    if torch_module is not None:
        return torch_module
    return get_torch(context=f"grammar_postprocessor:{context}")


def get_verb_dictionary_path() -> Path | None:
    """Return the currently available verb dictionary path, if any."""
    if BUNDLED_VERB_DICT_PATH.exists():
        return BUNDLED_VERB_DICT_PATH

    writable_path = DATA_DIR / "verb-form-vocab.txt"
    if writable_path.exists():
        return writable_path
    return None


def get_languagetool_runtime_status() -> LanguageToolRuntimeStatus:
    """Return whether local LanguageTool assets are available without starting the server."""
    try:
        import language_tool_python
    except ImportError:
        return LanguageToolRuntimeStatus(False, "language_tool_python is not installed")
    except Exception as exc:
        return LanguageToolRuntimeStatus(False, f"language_tool_python import failed: {exc}")

    try:
        java_path, jar_path = language_tool_python.utils.get_jar_info()
    except Exception as exc:
        return LanguageToolRuntimeStatus(False, str(exc))

    jar_root = Path(jar_path)
    java_binary = Path(java_path)
    if not jar_root.exists():
        return LanguageToolRuntimeStatus(False, f"LanguageTool assets missing at {jar_root}")
    if not java_binary.exists():
        return LanguageToolRuntimeStatus(False, f"Java runtime missing at {java_binary}")
    return LanguageToolRuntimeStatus(True, f"Ready at {jar_root}")


def _get_languagetool(language: str = "en-US") -> Optional[Any]:  # -> language_tool_python.LanguageTool
    """Get or create the LanguageTool instance (lazy loading, thread-safe)."""
    language_key = language.strip() if isinstance(language, str) else "en-US"
    if not language_key:
        language_key = "en-US"

    if language_key in _lt_instances:
        return _lt_instances[language_key]

    with _lt_lock:
        if language_key in _lt_instances:
            return _lt_instances[language_key]

        runtime_status = get_languagetool_runtime_status()
        if not runtime_status.available:
            logger.warning("LanguageTool runtime unavailable: %s", runtime_status.detail)
            return None

        try:
            import language_tool_python
            logger.info("Initializing LanguageTool...")
            tool = language_tool_python.LanguageTool(language_key)
            _lt_instances[language_key] = tool
            logger.info("LanguageTool ready")
            return tool
        except ImportError:
            logger.warning("language_tool_python not installed. Run: pip install language_tool_python")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize LanguageTool: {e}")
            return None


# =============================================================================
# GECToR Backend (Primary)
# =============================================================================


def _load_gector_model(model_id: str) -> Any:
    """
    Load GECToR model with a workaround for transformers >= 4.49.

    The standard GECToR.from_pretrained() fails because transformers now uses
    "meta device" initialization for efficient loading. However, GECToR's __init__
    calls AutoModel.from_pretrained() internally, which conflicts with the outer
    meta device context and raises an error.

    This function bypasses the issue by:
    1. Loading the config manually from HuggingFace Hub
    2. Creating a patched GECToR subclass that disables meta device loading
    3. Loading the state dict separately

    Args:
        model_id: HuggingFace model ID (e.g., 'gotutiyan/gector-roberta-large-5k')

    Returns:
        Loaded GECToR model instance
    """
    torch = _load_torch_module(context="_load_gector_model")
    if torch is None:
        raise RuntimeError("PyTorch unavailable")

    import torch.nn as nn
    from torch.nn import CrossEntropyLoss
    import json
    from gector import GECToR, GECToRConfig
    from transformers import AutoModel, AutoTokenizer, PreTrainedModel
    from huggingface_hub import hf_hub_download

    # Step 1: Download and load config
    config_path = hf_hub_download(repo_id=model_id, filename="config.json")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    config = GECToRConfig(**config_dict)

    # Step 2: Create patched GECToR class that avoids meta device issues
    class _GECToRPatched(GECToR):
        """GECToR with patched __init__ to avoid meta device conflicts."""

        def __init__(self, config):
            # Call PreTrainedModel.__init__ directly (not GECToR's)
            PreTrainedModel.__init__(self, config)
            self.config = config

            # Load tokenizer normally
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_id)

            # Load BERT with low_cpu_mem_usage=False to disable meta device
            bert_kwargs = {"low_cpu_mem_usage": False}
            if config.has_add_pooling_layer:
                bert_kwargs["add_pooling_layer"] = False
            self.bert = AutoModel.from_pretrained(config.model_id, **bert_kwargs)

            # Extend embeddings for $START token
            self.bert.resize_token_embeddings(
                self.bert.config.vocab_size + 1,
                mean_resizing=False
            )

            # Create projection layers
            self.label_proj_layer = nn.Linear(
                self.bert.config.hidden_size,
                config.num_labels - 1  # -1 for <PAD>
            )
            self.d_proj_layer = nn.Linear(
                self.bert.config.hidden_size,
                config.d_num_labels - 1
            )
            self.dropout = nn.Dropout(config.p_dropout)
            self.loss_fn = CrossEntropyLoss(label_smoothing=config.label_smoothing)

            self.post_init()
            self.tune_bert(False)

    # Step 3: Create model instance (bypasses from_pretrained's meta device)
    model = _GECToRPatched(config)

    # Step 4: Download and load weights
    model_path = hf_hub_download(repo_id=model_id, filename="pytorch_model.bin")
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)

    return model


class _GECToRManager:
    """Lazy-loaded singleton manager for GECToR model."""

    _instance: Optional["_GECToRManager"] = None
    _lock = threading.Lock()

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.encode = None
        self.decode = None
        self.device: Optional[str] = None
        self._initialized = False
        self._initialization_error: Optional[str] = None

    @classmethod
    def get_instance(cls) -> "_GECToRManager":
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _ensure_verb_dict(self) -> Optional[Path]:
        """Download verb dictionary if not present."""
        verb_dict_path = get_verb_dictionary_path()

        if verb_dict_path is not None:
            return verb_dict_path

        try:
            verb_dict_path = DATA_DIR / "verb-form-vocab.txt"
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"Downloading verb dictionary from {VERB_DICT_URL}...")
            urllib.request.urlretrieve(VERB_DICT_URL, verb_dict_path)
            logger.info("Verb dictionary downloaded")
            return verb_dict_path
        except Exception as e:
            logger.error(f"Failed to download verb dictionary: {e}")
            return None

    def initialize(self, model_id: str) -> bool:
        """Initialize GECToR model (lazy, thread-safe).

        Returns:
            True if initialization successful, False otherwise.
        """
        if self._initialized:
            return True

        with self._lock:
            if self._initialized:
                return True

            try:
                self._initialization_error = None
                logger.info(f"Loading GECToR model: {model_id}...")

                # Ensure verb dictionary exists before probing heavy runtime deps.
                verb_dict_path = self._ensure_verb_dict()
                if verb_dict_path is None:
                    self._initialization_error = "Verb dictionary unavailable"
                    return False

                torch = _load_torch_module(context="initialize")
                if torch is None:
                    self._initialization_error = "PyTorch unavailable"
                    logger.warning(self._initialization_error)
                    return False

                from transformers import AutoTokenizer
                from gector import load_verb_dict

                # Determine device
                if torch.cuda.is_available():
                    self.device = "cuda"
                else:
                    self.device = "cpu"

                # Load model using helper function (handles transformers >= 4.49 compatibility)
                self.model = _load_gector_model(model_id)

                # Move to appropriate device
                if self.device == "cuda":
                    self.model = self.model.cuda()
                    logger.info("GECToR using CUDA acceleration")
                else:
                    self.model = self.model.cpu()
                    logger.info("GECToR using CPU (no CUDA available)")

                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_id)

                # Load verb dictionary
                self.encode, self.decode = load_verb_dict(str(verb_dict_path))

                self._initialized = True
                logger.info("GECToR initialization complete")
                return True

            except ImportError as e:
                self._initialization_error = f"GECToR not installed: {e}"
                logger.warning(self._initialization_error)
                return False
            except Exception as e:
                self._initialization_error = f"GECToR initialization failed: {e}"
                logger.error(self._initialization_error)
                return False

    def is_available(self) -> bool:
        """Check if GECToR is initialized and ready."""
        return self._initialized

    def get_error(self) -> Optional[str]:
        """Get initialization error message if any."""
        return self._initialization_error

    def unload(self) -> None:
        """Unload GECToR model from GPU to free VRAM.

        The model can be reloaded later by calling initialize() again.
        """
        with self._lock:
            if not self._initialized:
                return

            try:
                torch = _load_torch_module(context="unload")

                if self.model is not None:
                    # Move model to CPU first to free GPU memory
                    if self.device == "cuda":
                        self.model = self.model.cpu()
                    del self.model
                    self.model = None

                if self.tokenizer is not None:
                    del self.tokenizer
                    self.tokenizer = None

                self.encode = None
                self.decode = None
                self.device = None
                self._initialized = False
                self._initialization_error = None

                # Clear CUDA cache
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                logger.info("GECToR model unloaded from GPU")

            except Exception as e:
                logger.warning(f"Error unloading GECToR: {e}")

    def correct(
        self,
        sentences: List[str],
        batch_size: int = 8,
        n_iterations: int = 5,
    ) -> List[str]:
        """Correct a list of sentences using GECToR.

        Args:
            sentences: List of sentences to correct.
            batch_size: Batch size for inference.
            n_iterations: Number of correction iterations.

        Returns:
            List of corrected sentences.
        """
        if not self._initialized or not sentences:
            return sentences

        try:
            from gector import predict

            corrected = predict(
                self.model,  # type: ignore[arg-type]
                self.tokenizer,  # type: ignore[arg-type]
                sentences,
                self.encode,  # type: ignore[arg-type]
                self.decode,  # type: ignore[arg-type]
                keep_confidence=0.0,
                min_error_prob=0.0,
                n_iteration=n_iterations,
                batch_size=batch_size,
            )
            return corrected
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("GECToR CUDA out of memory, clearing cache")
                try:
                    torch = _load_torch_module(context="correct")
                    if torch is not None:
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            raise
        except Exception as e:
            logger.error(f"GECToR correction failed: {e}")
            return sentences


# =============================================================================
# Main Grammar Processor
# =============================================================================

class GrammarPostProcessor:
    """Handles grammar correction with GECToR (primary) and LanguageTool (fallback)."""

    def __init__(self, config: Optional[GrammarConfig] = None):
        """Initialize the post-processor.

        Args:
            config: Grammar configuration. Uses defaults if not provided.
        """
        self.config = config or GrammarConfig()
        self._gector_manager: Any = _GECToRManager.get_instance()
        self._active_backend: Optional[str] = None

    def _ensure_backend(self) -> bool:
        """Initialize the appropriate backend based on config.

        Returns:
            True if a backend is available, False otherwise.
        """
        if self._active_backend:
            return True

        backend = self.config.backend.lower()

        # Try GECToR first (unless explicitly set to languagetool)
        if backend in ("auto", "gector"):
            if self._gector_manager.initialize(self.config.gector_model):
                self._active_backend = "gector"
                return True
            elif backend == "gector":
                # GECToR explicitly required but failed
                return False

        # Try LanguageTool as fallback
        if backend in ("auto", "languagetool"):
            tool = _get_languagetool(self.config.language)
            if tool is not None:
                self._active_backend = "languagetool"
                return True

        return False

    def is_available(self) -> bool:
        """Check if any grammar backend is available.

        Returns:
            True if a backend can be used.
        """
        return self._ensure_backend()

    def get_status(self) -> str:
        """Get human-readable status.

        Returns:
            Status string describing the current state.
        """
        if not self.config.enabled:
            return "Disabled"

        if self._ensure_backend():
            if self._active_backend == "gector":
                device = self._gector_manager.device or "unknown"
                return f"GECToR ({device})"
            return "LanguageTool"

        # Show why it failed
        gector_err = self._gector_manager.get_error()
        if gector_err:
            return f"Unavailable: {gector_err}"
        return "No backend available"

    def peek_status(self) -> str:
        """Return a startup-safe status string without initializing heavy backends."""
        if not self.config.enabled:
            return "Disabled"

        if self._active_backend == "gector":
            device = self._gector_manager.device or "unknown"
            return f"GECToR ({device})"
        if self._active_backend == "languagetool":
            return "LanguageTool"

        backend = self.config.backend.lower()
        gector_error = self._gector_manager.get_error()
        torch_error = get_torch_import_error()
        verb_dict_path = get_verb_dictionary_path()
        lt_status = get_languagetool_runtime_status()

        if backend in ("auto", "gector"):
            if gector_error:
                if backend == "gector":
                    return f"Unavailable: {gector_error}"
            elif verb_dict_path is not None and torch_error is None:
                return "GECToR (ready on demand)"
            elif backend == "gector" and torch_error is not None:
                return "Unavailable: PyTorch unavailable"

        if backend in ("auto", "languagetool"):
            if lt_status.available:
                return "LanguageTool (ready on demand)"
            if backend == "languagetool":
                return f"Unavailable: {lt_status.detail}"

        if gector_error:
            return f"Unavailable: {gector_error}"
        if backend in ("auto", "languagetool") and not lt_status.available:
            return f"Unavailable: {lt_status.detail}"
        return "No backend available"

    def process_text(self, text: str) -> Tuple[str, bool]:
        """Process text for grammar/spelling correction.

        Args:
            text: Full transcript text.

        Returns:
            Tuple of (processed_text, was_enhanced).
        """
        if not self.config.enabled:
            return text, False

        if not text or not text.strip():
            return text, False

        if not self._ensure_backend():
            logger.warning("No grammar backend available, skipping correction")
            return text, False

        logger.info(f"Starting grammar correction with {self._active_backend}...")

        try:
            if self._active_backend == "gector":
                return self._correct_with_gector(text)
            else:
                return self._correct_with_languagetool(text)
        except Exception as e:
            logger.error(f"Grammar correction failed: {e}")
            # Try fallback if using GECToR
            if self._active_backend == "gector" and self.config.backend == "auto":
                logger.info("Attempting LanguageTool fallback...")
                self._active_backend = None
                tool = _get_languagetool(self.config.language)
                if tool is not None:
                    self._active_backend = "languagetool"
                    try:
                        return self._correct_with_languagetool(text)
                    except Exception as e2:
                        logger.error(f"LanguageTool fallback also failed: {e2}")
            return text, False

    def _correct_with_gector(self, text: str) -> Tuple[str, bool]:
        """Correct text using GECToR."""
        # Split into sentences for better handling
        sentences = self._split_sentences(text)
        if not sentences:
            return text, False

        corrected = self._gector_manager.correct(
            sentences,
            batch_size=self.config.gector_batch_size,
            n_iterations=self.config.gector_iterations,
        )

        result = " ".join(corrected)
        was_changed = result.strip() != text.strip()

        if was_changed:
            logger.info("GECToR corrections applied")
        else:
            logger.info("GECToR: no corrections needed")

        return result, was_changed

    def _correct_with_languagetool(self, text: str) -> Tuple[str, bool]:
        """Correct text using LanguageTool (fallback)."""
        import language_tool_python

        tool = _get_languagetool(self.config.language)
        if tool is None:
            return text, False

        matches = tool.check(text)
        if not matches:
            logger.info("LanguageTool: no corrections needed")
            return text, False

        corrected = language_tool_python.utils.correct(text, matches)
        logger.info(f"LanguageTool: {len(matches)} corrections applied")
        return corrected, True

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences for GECToR processing."""
        import re
        # Split on sentence-ending punctuation followed by whitespace
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def process_segments(
        self,
        segments_data: Sequence[TranscriptSegmentLike],
    ) -> Tuple[TranscriptSegments, bool]:
        """Process transcript segments for grammar correction.

        Preserves timestamp information while correcting text.

        Args:
            segments_data: List of dicts with 'start', 'end', 'text' keys.

        Returns:
            Tuple of (processed_segments, was_enhanced).
        """
        normalized_segments = coerce_transcript_segments(segments_data)
        if not self.config.enabled or not normalized_segments:
            return normalized_segments, False

        if not self._ensure_backend():
            return normalized_segments, False

        logger.info(f"Starting segment grammar correction with {self._active_backend}...")

        try:
            if self._active_backend == "gector":
                return self._process_segments_gector(normalized_segments)
            else:
                return self._process_segments_languagetool(normalized_segments)
        except Exception as e:
            logger.error(f"Segment correction failed: {e}")
            # Try fallback
            if self._active_backend == "gector" and self.config.backend == "auto":
                logger.info("Attempting LanguageTool fallback for segments...")
                self._active_backend = None
                tool = _get_languagetool(self.config.language)
                if tool is not None:
                    self._active_backend = "languagetool"
                    try:
                        return self._process_segments_languagetool(normalized_segments)
                    except Exception as e2:
                        logger.error(f"LanguageTool fallback also failed: {e2}")
            return normalized_segments, False

    def _process_segments_gector(
        self,
        segments_data: Sequence[TranscriptSegmentLike],
    ) -> Tuple[TranscriptSegments, bool]:
        """Process segments using GECToR with batch optimization."""
        normalized_segments = coerce_transcript_segments(segments_data)
        # Collect all segment texts
        texts = [seg.get("text", "").strip() for seg in normalized_segments]
        non_empty_indices = [i for i, t in enumerate(texts) if t]
        non_empty_index_set = set(non_empty_indices)
        non_empty_texts = [texts[i] for i in non_empty_indices]

        if not non_empty_texts:
            return normalized_segments, False

        # Batch correct all texts at once
        corrected = self._gector_manager.correct(
            non_empty_texts,
            batch_size=self.config.gector_batch_size,
            n_iterations=self.config.gector_iterations,
        )

        # Map corrected texts back to segments
        processed_segments: TranscriptSegments = []
        corrected_iter = iter(corrected)
        any_enhanced = False

        for i, seg in enumerate(normalized_segments):
            if i in non_empty_index_set:
                new_text = next(corrected_iter)
                if new_text != seg.get("text", ""):
                    any_enhanced = True
                processed_segments.append(replace_segment_text(seg, new_text))
            else:
                processed_segments.append(seg)

        if any_enhanced:
            logger.info("GECToR segment corrections applied")
        else:
            logger.info("GECToR: no segment corrections needed")

        return processed_segments, any_enhanced

    def _process_segments_languagetool(
        self,
        segments_data: Sequence[TranscriptSegmentLike],
    ) -> Tuple[TranscriptSegments, bool]:
        """Process segments using LanguageTool."""
        import language_tool_python

        tool = _get_languagetool(self.config.language)
        if tool is None:
            return coerce_transcript_segments(segments_data), False

        normalized_segments = coerce_transcript_segments(segments_data)
        processed_segments: TranscriptSegments = []
        any_enhanced = False

        for seg in normalized_segments:
            original_text = seg.get("text", "")

            if not original_text.strip():
                processed_segments.append(seg)
                continue

            try:
                matches = tool.check(original_text)
                if matches:
                    corrected = language_tool_python.utils.correct(original_text, matches)
                    processed_segments.append(replace_segment_text(seg, corrected))
                    any_enhanced = True
                else:
                    processed_segments.append(seg)
            except Exception as e:
                logger.debug(f"Segment correction failed: {e}")
                processed_segments.append(seg)

        if any_enhanced:
            logger.info("LanguageTool segment corrections applied")
        else:
            logger.info("LanguageTool: no segment corrections needed")

        return processed_segments, any_enhanced


# =============================================================================
# Convenience Functions
# =============================================================================

@overload
def post_process_grammar(
    text: str = "",
    segments_data: None = None,
    config: Optional[GrammarConfig] = None,
) -> Tuple[str, None, bool]: ...


@overload
def post_process_grammar(
    text: str,
    segments_data: Sequence[TranscriptSegmentLike],
    config: Optional[GrammarConfig] = None,
) -> Tuple[str, TranscriptSegments, bool]: ...


def post_process_grammar(
    text: str = "",
    segments_data: Optional[Sequence[TranscriptSegmentLike]] = None,
    config: Optional[GrammarConfig] = None,
) -> Tuple[str, Optional[TranscriptSegments], bool]:
    """Post-process transcript with grammar correction.

    Convenience function for easy integration into existing code.

    Args:
        text: Full transcript text.
        segments_data: Optional list of segment dicts with timestamps.
        config: Grammar configuration.

    Returns:
        Tuple of (processed_text, processed_segments, was_enhanced).
    """
    processor = GrammarPostProcessor(config)

    if segments_data is not None:
        processed_segments, was_enhanced = processor.process_segments(segments_data)
        processed_text = " ".join(seg.get("text", "") for seg in processed_segments)
        return processed_text, processed_segments, was_enhanced
    else:
        processed_text, was_enhanced = processor.process_text(text)
        return processed_text, None, was_enhanced


def check_grammar_status(*, lazy: bool = False) -> Tuple[bool, str]:
    """Check if grammar post-processing is available.

    Returns:
        Tuple of (is_available, status_message).
    """
    processor = GrammarPostProcessor()
    status = processor.peek_status() if lazy else processor.get_status()
    is_available = "Unavailable" not in status and status != "Disabled" and status != "No backend available"
    return is_available, status


def unload_gector() -> None:
    """Unload GECToR model from GPU to free VRAM.

    Call this before loading other large models (like Whisper) to prevent
    CUDA out of memory errors. The model will be automatically reloaded
    when grammar correction is next used.
    """
    manager = _GECToRManager.get_instance()
    manager.unload()

