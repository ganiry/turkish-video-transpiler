# Turkish Video Transpiler

Automatic captions and English translations for videos, using [Whisper](https://github.com/openai/whisper) running locally through [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Built for Turkish lessons, but the spoken language is detected automatically, so it works for any language Whisper supports.

For every video it produces:

- **Captions in the spoken language** (e.g. `lesson.tr.vtt`)
- **An English translation** (`lesson.en.vtt`). Skipped when the video is already in English.

Both are standard [WebVTT](https://developer.mozilla.org/en-US/docs/Web/API/WebVTT_API) files, so they work with any HTML5 player, Mux, Vimeo, and most video platforms. A viewer page (`index.html`) shows each video with both caption tracks and a clickable transcript.

Everything runs on your machine: no API keys, no per-minute costs, and videos are never uploaded anywhere.

## Requirements

- Python 3.9+
- [ffmpeg](https://ffmpeg.org/) on your `PATH`
- About 3 GB of disk space for the `large-v3` model, which downloads automatically on first run to `~/.cache/huggingface/` (set `HF_HOME` to change this)
- About 4 GB of free RAM for `large-v3`

## Install

```bash
git clone https://github.com/ganiry/turkish-video-transpiler.git
cd turkish-video-transpiler
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Pass video files, links, or `.txt` lists in any combination:

```bash
python caption.py lesson.mp4                          # a local file
python caption.py https://example.com/lesson.mp4      # a link
python caption.py links.txt                           # a list
python caption.py a.mp4 https://example.com/b.mp4 more.txt   # any mix
```

A list file has one video path or link per line. Blank lines and lines starting with `#` are ignored:

```
# Course 1
https://example.com/course-1/lesson-01.mp4
https://example.com/course-1/lesson-02.mp4
videos/intro.mp4
```

Then open `index.html` in your browser to watch the results. It works straight from disk, with no server needed.

### Output

```
captions/
  lesson-01.tr.vtt     # captions in the detected language
  lesson-01.en.vtt     # English translation
  manifest.js          # index of captioned videos, read by index.html
```

To use the captions in your own player:

```html
<video src="https://example.com/lesson-01.mp4" controls>
  <track kind="captions" src="captions/lesson-01.tr.vtt" srclang="tr" label="Türkçe" default>
  <track kind="subtitles" src="captions/lesson-01.en.vtt" srclang="en" label="English">
</video>
```

### Options

| Option | Description |
|---|---|
| `--model NAME` | Whisper model: `large-v3` (default, most accurate), `medium`, `small` (faster, less accurate) |
| `--force` | Re-caption videos that are already done |
| `--name NAME` | Output file name, when captioning a single video |
| `--list FILE` | Read a list file (same as passing a `.txt` directly) |

### Batch behaviour

- The model loads **once** per run and is reused for every video.
- **Already captioned videos are skipped**, so an interrupted batch can be re-run with the same command to continue where it stopped.
- Progress is saved after each video.
- **A broken link doesn't stop the batch.** Failed entries are written to `failed.txt`; retry with `python caption.py failed.txt`.
- Links that share a file name (e.g. `/course-1/video.mp4` and `/course-2/video.mp4`) get unique output names.
- Links are downloaded to a temporary file before processing, which works with any host. Only the audio is used, and the file is deleted afterwards.

## Performance

On a typical laptop CPU with `large-v3`, processing takes roughly **2–3× the video's length** (about 4 minutes for a 1m45s video), covering both the captions and the translation. Use `--model small` for roughly 10× faster runs at lower accuracy. An NVIDIA GPU is much faster; change `device="cpu"` to `device="cuda"` in `caption.py`.

## Limitations

- **Translation is English only.** That's a limit of Whisper itself. Other target languages need a separate translation step.
- **The language is detected from the first 30 seconds.** A video that switches language partway through is captioned as its opening language.
- **Accuracy drops** with background music, heavy noise, or overlapping speakers. Technical terms may occasionally be misheard.
