"""Auto-caption a video with Whisper.

Usage:
    python caption.py lesson.mp4                       # a local file
    python caption.py https://host.com/lesson.mp4      # a link
    python caption.py links.txt                        # a list: one path/URL per line
    python caption.py a.mp4 https://host.com/b.mp4 more-links.txt   # any mix

Videos already in the manifest are skipped (--force to redo); failures go to failed.txt.

Detects the spoken language, writes captions in that language plus an English
translation (skipped when the video is already English), and registers the video
in captions/manifest.js so index.html picks it up.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

ROOT = Path(__file__).parent
OUT = ROOT / "captions"
MANIFEST = OUT / "manifest.js"

LANG_NAMES = {
    "tr": "Türkçe", "en": "English", "de": "Deutsch", "fr": "Français", "es": "Español",
    "it": "Italiano", "pt": "Português", "ar": "العربية", "ru": "Русский", "nl": "Nederlands",
    "ja": "日本語", "ko": "한국어", "zh": "中文", "pl": "Polski", "uk": "Українська",
}


def load_audio(src):
    # Download URLs first: ffmpeg streaming straight from a URL fails on hosts that
    # don't support range requests (it needs to seek to the MP4's index)
    if re.match(r"https?://", src):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            with urllib.request.urlopen(src, timeout=60) as r:
                shutil.copyfileobj(r, tmp)
            tmp.flush()
            return load_audio(tmp.name)
    pcm = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", src, "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
        capture_output=True, check=True,
    ).stdout
    if len(pcm) < 16000 * 4:  # under one second of audio
        raise RuntimeError("no audio could be decoded (missing audio track or broken file)")
    return np.frombuffer(pcm, np.float32)


def ts(t):
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    return f"{int(h):02}:{int(m):02}:{s:06.3f}"


def write_vtt(segments, path):
    cues = []
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for s in segments:
            text = s.text.strip()
            if not text:
                continue
            f.write(f"{ts(s.start)} --> {ts(s.end)}\n{text}\n\n")
            cues.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": text})
    return cues


def load_manifest():
    if not MANIFEST.exists():
        return []
    body = MANIFEST.read_text(encoding="utf-8")
    return json.loads(body[body.index("=") + 1:].strip().rstrip(";"))


def save_manifest(entries):
    MANIFEST.write_text(
        "window.CAPTIONS = " + json.dumps(entries, ensure_ascii=False, indent=2) + ";\n",
        encoding="utf-8",
    )


def name_for(source, taken):
    name = re.sub(r"[^\w-]+", "-", Path(source.split("?")[0]).stem).strip("-") or "video"
    # Different URLs often share a filename (e.g. .../1/video.mp4, .../2/video.mp4)
    if taken.get(name, source) != source:
        name += "-" + hashlib.sha1(source.encode()).hexdigest()[:6]
    return name


def caption_one(model, model_name, source, name):
    is_url = re.match(r"https?://", source)
    t0 = time.time()
    audio = load_audio(source)
    opts = dict(beam_size=5, vad_filter=True)

    segments, info = model.transcribe(audio, **opts)
    lang = info.language
    log(f"  detected language: {lang} ({info.language_probability:.0%})")
    tracks = [{
        "lang": lang, "label": LANG_NAMES.get(lang, lang), "kind": "captions",
        "src": f"captions/{name}.{lang}.vtt",
        "cues": write_vtt(segments, OUT / f"{name}.{lang}.vtt"),
    }]

    if lang != "en":
        segments, _ = model.transcribe(audio, task="translate", language=lang, **opts)
        tracks.append({
            "lang": "en", "label": "English (translated)", "kind": "subtitles",
            "src": f"captions/{name}.en.vtt",
            "cues": write_vtt(segments, OUT / f"{name}.en.vtt"),
        })

    # Page lives next to this script, so local paths are made relative to it
    video_src = source if is_url else os.path.relpath(Path(source).resolve(), ROOT.resolve())
    return {
        "id": name, "source": source, "video": video_src, "language": lang,
        "confidence": round(info.language_probability, 3),
        "duration": round(info.duration, 1), "model": model_name,
        "processing_seconds": round(time.time() - t0, 1), "tracks": tracks,
    }


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="*",
                    help="video files, http(s) links, or .txt lists with one path/link per line")
    ap.add_argument("--list", action="append", default=[], help="list file (same as passing a .txt)")
    ap.add_argument("--name", help="id for output files (single source only)")
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--force", action="store_true", help="re-caption videos that are already done")
    args = ap.parse_args()

    sources = []
    for arg in args.sources + args.list:
        if arg in args.list or (arg.lower().endswith(".txt") and not re.match(r"https?://", arg)):
            lines = Path(arg).read_text(encoding="utf-8").splitlines()
            sources += [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]
        else:
            sources.append(arg)
    sources = list(dict.fromkeys(sources))  # drop duplicates, keep order
    if not sources:
        ap.error("give at least one video file, link, or .txt list")
    if args.name and len(sources) > 1:
        ap.error("--name only works with a single video")

    OUT.mkdir(exist_ok=True)
    entries = load_manifest()
    done = {e.get("source", e["video"]) for e in entries}
    todo = [s for s in sources if args.force or s not in done]
    if len(todo) < len(sources):
        log(f"Skipping {len(sources) - len(todo)} already captioned (use --force to redo)")
    if not todo:
        return

    model = WhisperModel(args.model, device="cpu", compute_type="int8")  # loaded once for the whole batch
    failed = []
    for i, source in enumerate(todo, 1):
        taken = {e["id"]: e.get("source", e["video"]) for e in entries}
        name = args.name or name_for(source, taken)
        log(f"[{i}/{len(todo)}] {source}")
        try:
            entry = caption_one(model, args.model, source, name)
        except Exception as e:  # one bad link shouldn't stop the batch
            msg = e.stderr.decode(errors="replace").strip() if isinstance(e, subprocess.CalledProcessError) else str(e)
            log(f"  FAILED: {msg.splitlines()[-1] if msg else e}")
            failed.append(source)
            continue
        entries = [e for e in entries if e["id"] != name] + [entry]
        save_manifest(entries)  # saved after each video, so progress survives a crash
        log(f"  done in {entry['processing_seconds']}s -> {', '.join(t['src'] for t in entry['tracks'])}")

    log(f"\nFinished: {len(todo) - len(failed)} captioned, {len(failed)} failed")
    if failed:
        Path("failed.txt").write_text("\n".join(failed) + "\n", encoding="utf-8")
        log("Failed links written to failed.txt (retry with: python caption.py failed.txt)")
        sys.exit(1)


if __name__ == "__main__":
    main()
