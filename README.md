# thought_dump

Continuous voice chat with a local Gemma model. Press Left Option to start
listening — speak naturally, and each time you pause it transcribes locally
(faster-whisper) and sends to a local Gemma model (via Ollama), streaming a
Markdown-rendered response (proper code blocks, headers, etc.) straight into
the terminal. Press Right Option to stop listening.

## Setup

```bash
brew install ollama
ollama pull gemma3:4b     # ~3.3GB, good balance of speed/quality on 16GB Macs
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Ollama needs to be running (`ollama serve`, or just launch the Ollama app —
it serves in the background).

## Usage

```bash
python3 voice_chat.py
```

Press **Left Option** to start listening, then just talk — every time you
pause for ~0.7s, that utterance is sent off and the response streams in.
Press **Left Option** again to stop listening. Ctrl+C to quit.

Left Option is a modifier key, so it never types a character into the
terminal — no echo/input-buffer issues to work around.

The first run will prompt macOS for **Microphone** and **Input
Monitoring/Accessibility** permissions for your terminal app — grant both,
you may need to restart the terminal afterward.

## Tuning

All near the top of `voice_chat.py`:

- `OLLAMA_MODEL` — swap to `gemma2:2b` for faster/lower-quality responses.
- `WHISPER_MODEL_SIZE` — `tiny.en` is faster but less accurate than the
  default `base.en`.
- `ENERGY_THRESHOLD` — raise if background noise triggers false starts,
  lower if quiet speech isn't being picked up.
- `SILENCE_HANGOVER_BLOCKS` — how long a pause (in ~30ms blocks) before an
  utterance is considered finished and sent.
