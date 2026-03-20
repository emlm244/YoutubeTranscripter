# YouTubeTranscripter Codereviewer Playbook

## Mission

This is the handoff for a Codex-to-Codex cleanup pass.

The reviewer should act like a skeptical maintainer, not a feature sprinter:

- remove overlapping or hanging implementation only when usage is disproven
- implement missing intent only when the behavior is already clear in nearby code
- keep the repo verification-clean before widening the cleanup surface
- update docs when runtime reality and documentation diverge

This repo is a PyQt desktop app with three flows that must survive every cleanup:

- YouTube URL transcription
- local file transcription
- microphone recording

## Required Thinking Style

- Use Auggie first for any unfamiliar module or cross-file behavior.
- After Auggie, use exact search before deleting anything.
- Prefer proving a claim over inferring it.
- Treat tests and packaging scripts as production code, not support clutter.
- Fix the verification loop before trusting broad refactors.
- Keep source and packaged behavior aligned.
- Do not remove anything that is only "unused-looking" if it protects Windows startup, cache setup, grammar laziness, or typed transcript normalization.

## Tools And Skills

Mandatory tools:

- Auggie (`mcp__auggie__codebase_retrieval`) for repo context and symbol mapping
- exact search for callsite confirmation before deletions
- explorer subagents for parallel read-only audits of core, GUI, grammar, and packaging/test surfaces
- the normal verification loop: `pytest`, `ruff check .`, `pyright`

Skill guidance:

- No Codex skill is required to review this repo well.
- Skip `figma` and `figma-implement-design`; this is not a design-to-code task.
- Skip `playwright` and `playwright-interactive` for normal audit work; this repo is a desktop PyQt app, not a browser UI.
- If a future pass adds browser-based docs verification or web tooling, reassess those skills then.

## Source Boundary

Primary authored surface:

- core/runtime: `youtube_transcriber.py`, `audio_preprocessor.py`, `transcript_types.py`, `app_paths.py`, `torch_runtime.py`, `runtime_bootstrap.py`, `launcher_preflight.py`, `exceptions.py`
- GUI/config/theme: `gui_transcriber.py`, `config.py`, `themes.py`, `widgets/`
- grammar: `grammar_postprocessor.py`, `data/verb-form-vocab.txt`
- packaging/docs: `run_gui.bat`, `build_standalone.ps1`, `build_cache_manifest.py`, `youtube_transcriber.spec`, `README.md`, `AGENTS.md`, `pyproject.toml`, `requirements*.txt`
- tests: `tests/`

Generated or disposable artifacts stay out of scope unless verifying packaging behavior:

- `build/`
- `dist/`
- `venv/`
- `hf-cache/`
- `tmp/`
- logs, caches, and `__pycache__`

Note on `data/verb-form-vocab.txt`:

- treat it as a shipped asset, not as general-purpose source to "simplify"
- review its integration points and packaging behavior, not the vocabulary content itself

## Current Verified Repo Facts

These were verified during this audit and should shape the next cleanup pass.

1. Git metadata is present.
   - `.git` exists in the repo root, so commit/branch work is valid here.

2. The worktree is already dirty before the cleanup pass starts.
   - current modified files: `gui_transcriber.py`, `tests/test_gui_logging.py`, `tests/test_transcriber.py`, `youtube_transcriber.py`
   - do not fold those unrelated edits into cleanup commits unless the reviewer intentionally builds on top of them

3. `ruff` and `pyright` are currently clean.
   - `ruff check .` passes
   - `pyright` passes with `0 errors, 0 warnings`

4. Full `pytest` is not suite-clean even though the individual test files pass.
   - `pytest` timed out twice at more than 5 minutes.
   - every test file passes when run on its own
   - `pytest tests/test_app_paths.py tests/test_grammar.py -q` times out
   - `pytest tests/test_grammar.py tests/test_app_paths.py -q` also times out
   - inference: this is a shared-state or non-hermetic test interaction, not just a simple ordering bug

5. The strongest current suite-clean suspects are the runtime-env tests and the non-lazy grammar status test.
   - `app_paths.py` writes `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, and `TRANSFORMERS_CACHE` directly in `configure_runtime_environment()`
   - `tests/test_app_paths.py` calls that function and asserts the mutated environment
   - `tests/test_grammar.py` still contains a real `check_grammar_status()` probe instead of a fully hermetic fake
   - review `tests/test_app_paths.py`, `tests/test_grammar.py`, and `grammar_postprocessor.py` first before trusting any broader cleanup work

6. The README says this is "Desktop and CLI transcription tooling", but the README only documents the GUI launcher.
   - the CLI entrypoint is real in `youtube_transcriber.py`
   - CLI usage and flags are undocumented in `README.md`

7. The launcher's log-location messaging is incomplete.
   - `run_gui.bat` says logs will be saved to `gui_transcriber.log` and `youtube_transcriber.log`
   - actual logging uses `get_log_path(...)`, which can fall back to a writable app-data directory instead of the working folder

8. There is confirmed dormant config surface in `UIConfig`.
   - `theme`, `accent_color`, `window_width`, `window_height`, `splitter_ratios`, and `remember_window_position` are persisted
   - the GUI currently sizes itself dynamically, persists splitter state through `QSettings`, and does not consume the theme/accent/window-position fields
   - this is a real cleanup candidate, but tests will need updating if the fields are removed

9. There is confirmed dormant widget/theme surface.
   - exact search found no callsites for `themes.set_theme()`
   - exact search found no callsites for `ThemeManager.get_glass_card_style()` or `ThemeManager.get_glow_color()`
   - exact search found no repo usage of `SpinnerRing`, `ProcessingOverlay`, or `RecordingPulse`
   - exact search also found no external usage of `FadeEffect`, `ScaleAnimation`, `RevealAnimation`, or exported `Breakpoint`
   - `AnimationManager`, `GlassCard`, `MaterialCard`, `MaterialButton`, and `ResponsiveSplitter` are actively used

10. The GUI currently has overlapping animation layers.
   - `GlassCard` already performs its own reveal animation on first show
   - `TranscriberGUI.run()` also triggers `AnimationManager.staggered_reveal(...)` across the same cards
   - review `gui_transcriber.py`, `widgets/material_card.py`, and `widgets/animations.py` together before keeping or removing any animation surface

11. There is at least one hanging UI implementation detail in the live widget path.
   - `MaterialButton.setVariant()` resets `self._glow_effect`, but the class never defines or uses a glow effect anywhere else
   - treat that as a high-confidence cleanup or simplification target once tests exist for button behavior

12. `ResponsiveSplitter` exposes more API than the app currently uses.
   - `ResponsiveSplitter` itself is live
   - exact search found no external usage of `Breakpoint`, `breakpointChanged`, `resetUserAdjustment()`, `currentBreakpoint()`, or `_last_user_sizes`
   - preserve the splitter unless or until the reviewer proves the breakpoint-only surface is dead

13. GUI shutdown logic is narrower than the app's actual worker model.
   - `closeEvent()` only signals `_youtube_cancel_event` plus `_recording_stop_event`
   - local-file transcription and post-recording transcription do not use a shared cancellation token
   - that does not prove a user-visible bug in every case, but it is a real thread-ownership hotspot to review before simplifying worker code

14. The GUI collapses two functional preprocessing settings behind one checkbox.
   - `_save_settings()` and `_build_transcription_config()` set both `noise_reduction_enabled` and `normalize_audio` from the same checkbox
   - decide whether normalization is supposed to be independently configurable; if not, collapse the config surface instead of preserving two settings that always move together

15. Packaging and launcher scripts have weak automated coverage.
   - tests cover `launcher_preflight.py`
   - there is no automated test coverage for `run_gui.bat`, `build_standalone.ps1`, `build_cache_manifest.py`, or `youtube_transcriber.spec`
   - packaging parity is enforced mostly by convention and manual smoke-testing

16. GUI/widget test coverage is shallow compared with the amount of custom Qt surface.
   - `tests/test_gui_logging.py` exercises worker-thread helpers with a fake GUI object
   - there are no direct tests that instantiate `MaterialButton`, `GlassCard`, `ResponsiveSplitter`, or `ThemeManager`
   - reviewer should add focused behavior tests before deleting live GUI abstractions

## Do-Not-Break Rules

- `gui_runtime_bootstrap.py` must remain the Windows-safe import path that primes `torch` before `PyQt6`.
- `runtime_bootstrap.py` remains the single owner of Hugging Face cache env setup.
- `AppConfig` owns functional settings; `QSettings` is only for Qt UI state.
- grammar status checks must stay lazy and side-effect free at startup
- typed transcript normalization from `transcript_types.py` must stay consistent across YouTube, local-file, and recording flows
- packaged and source runs must keep equivalent startup behavior

## Review Order

Work in this order. Do not skip the first pass.

1. Restore a trustworthy verification loop.
   - fix the combined `pytest` hang first
   - likely starting point: make `tests/test_app_paths.py` fully restore env mutations and make `tests/test_grammar.py` hermetic around `check_grammar_status()`
   - after the fix, rerun full `pytest`

2. Remove or fully wire dormant config surface.
   - decide whether `UIConfig` really owns theme/accent/window geometry/splitter ratios
   - if yes, wire them into the GUI and drop the overlapping `QSettings` usage only where appropriate
   - if no, prune the dead fields and update serialization/tests/docs together

3. Shrink dormant GUI/theme/widget surface.
   - validate unused widget helpers with Auggie plus exact search
   - remove unused loading indicators/theme hooks only when no product/design plan depends on them
   - keep the live Material-card/button/splitter path intact
   - simplify overlapping animation systems before deleting the live card widgets themselves
   - check whether `normalize_audio` should be merged into the live checkbox model or surfaced separately

4. Align docs with actual runtime behavior.
   - document the CLI if the repo is going to keep calling itself CLI-capable
   - document real log/config/cache locations
   - document that `-IncludeCachedModels` depends on the configured/cached model set on the build machine

5. Harden packaging parity.
   - make sure `run_gui.bat`, `build_standalone.ps1`, `build_cache_manifest.py`, and `youtube_transcriber.spec` still describe the same startup/runtime model
   - add focused tests where practical, especially for script-generated assumptions

6. Only after the above, widen into core simplification.
   - use Auggie to look for duplication in Whisper setup, config snapshotting, grammar fallback, and queue message handling
   - remove overlapping implementation only with regression coverage in place

## High-Confidence Cleanup Candidates

These are the best janitor targets because they are confirmed by exact search, not hunches.

- dormant `UIConfig` fields that are persisted but not consumed by the live GUI
- `themes.set_theme()` with no callsites
- `ThemeManager.get_glass_card_style()` and `ThemeManager.get_glow_color()` with no callsites
- unused loading-indicator widgets: `SpinnerRing`, `ProcessingOverlay`, `RecordingPulse`
- dangling `MaterialButton.setVariant()` `_glow_effect` reset with no implemented glow system
- stale "no skill required" guidance from the previous version of this plan
- README CLI/documentation drift
- launcher messaging that implies logs always live beside the batch file

## Medium-Confidence Candidates

These need intent checks before deletion.

- exported animation helper classes that are unused outside the widget package but still support `AnimationManager`
- overlapping startup animation paths between `GlassCard` and `AnimationManager`
- dormant theme-selection config fields that may represent an abandoned roadmap rather than pure clutter
- `ResponsiveSplitter` breakpoint-related API surface that currently has no external consumers
- separate `normalize_audio` config that is always driven by the noise-reduction checkbox
- any packaging helper that looks redundant but is preserving bundled/source parity on Windows

## Low-Confidence Or Preserve-By-Default Areas

- `data/verb-form-vocab.txt`
- `gui_runtime_bootstrap.py`
- `runtime_bootstrap.py`
- `torch_runtime.py`
- lazy grammar probing code
- transcript normalization helpers in `transcript_types.py`

## Exact File Hotspots

Verification blocker:

- `tests/test_app_paths.py`
- `tests/test_grammar.py`
- `app_paths.py`
- `grammar_postprocessor.py`

Dormant config/theme surface:

- `config.py`
- `gui_transcriber.py`
- `themes.py`

Dormant widget surface:

- `widgets/loading_indicator.py`
- `widgets/animations.py`
- `widgets/material_button.py`
- `widgets/responsive_layout.py`
- `widgets/__init__.py`
- `gui_transcriber.py`

Docs and launcher drift:

- `README.md`
- `run_gui.bat`
- `build_standalone.ps1`
- `build_cache_manifest.py`
- `youtube_transcriber.spec`

## Suggested Verification Loop

Baseline after every meaningful cleanup step:

```powershell
pytest
ruff check .
pyright
```

When touching packaging or startup:

```powershell
python launcher_preflight.py
python build_cache_manifest.py
```

Manual smoke-tests when launcher/packaging behavior changes:

- `run_gui.bat`
- `build_standalone.ps1`

If `pytest` hangs again, keep this repro pair handy:

```powershell
pytest tests/test_app_paths.py tests/test_grammar.py -q
pytest tests/test_grammar.py tests/test_app_paths.py -q
```

## Commit And PR Discipline

- verify `.git` exists before promising branch or PR work
- do not disturb unrelated local edits already present in the worktree
- prefer one cleanup family per commit when feasible:
  - verification fix
  - dormant config cleanup
  - widget/theme cleanup
  - docs/launcher alignment
- in the PR summary, separate:
  - verified removals
  - behavior-preserving simplifications
  - docs corrections
  - anything still unproven by automation

## Exit Criteria For The Reviewer

The cleanup pass is only done when all of this is true:

- `pytest` passes as a full suite, not just file-by-file
- `ruff check .` passes
- `pyright` passes
- the Windows-safe `torch` before `PyQt6` startup path is preserved
- docs describe the real GUI, CLI, log-path, cache-path, and packaging behavior
- every removed surface was validated with Auggie plus exact search
