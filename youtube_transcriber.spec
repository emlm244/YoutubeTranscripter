# -*- mode: python ; coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

project_root = Path(SPECPATH).resolve()


def _dedupe_entries(entries: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[tuple[str, str]] = []
    for entry in entries:
        if entry in seen:
            continue
        seen.add(entry)
        deduped.append(entry)
    return deduped

packages_to_collect = (
    "av",
    "ctranslate2",
    "faster_whisper",
    "gector",
    "huggingface_hub",
    "language_tool_python",
    "noisereduce",
    "sounddevice",
    "tokenizers",
    "torch",
    "transformers",
    "youtube_transcript_api",
    "yt_dlp",
)

optional_packages_to_collect = ("nvidia",)

datas = [(str(project_root / "data"), "data")]
binaries = []
hiddenimports = ["PyQt6.sip", "sip"]

for package in packages_to_collect:
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

for package in optional_packages_to_collect:
    try:
        package_datas, package_binaries, package_hiddenimports = collect_all(package)
    except Exception:
        continue
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

hiddenimports += collect_submodules("widgets")
hiddenimports = sorted(set(hiddenimports))
datas = _dedupe_entries(datas)
binaries = _dedupe_entries(binaries)

a = Analysis(
    [str(project_root / "gui_transcriber.py")],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="YouTubeTranscriber",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="YouTubeTranscriber",
)
