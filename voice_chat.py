"""
Continuous voice chat with a local Gemma model via Ollama.

Press LEFT OPTION to start listening. Speak naturally -- each time you pause
for ~0.7s, that utterance is transcribed (faster-whisper) and sent to
Ollama, with the response streamed into the terminal as Markdown/code. Keep
talking and it keeps answering. Press LEFT OPTION again to stop listening.
Ctrl+C to quit.

Setup:
    brew install ollama
    ollama pull gemma3:4b
    ollama serve   # (or just run `ollama` app, it serves in the background)
    pip install -r requirements.txt
    python voice_chat.py

macOS will prompt for Microphone + Input Monitoring/Accessibility
permissions for your terminal app the first time you run this.
"""

import argparse
import queue
import re
import sys
import threading
import time

import numpy as np
import ollama
import sounddevice as sd
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro
from pynput import keyboard
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

# --- Config ---------------------------------------------------------------

OLLAMA_MODEL = "gemma3:4b"
WHISPER_MODEL_SIZE = "base.en"
WHISPER_COMPUTE_TYPE = "int8"
SAMPLE_RATE = 16_000
TOGGLE_KEY = keyboard.Key.alt_l
KEEP_ALIVE = -1  # keep the model resident in memory between turns
NUM_CTX = 8192
TEMPERATURE = 0.2  # lower = terser, more literal, less chatty filler

TTS_ENABLED = True
KOKORO_MODEL_PATH = "models/kokoro-v1.0.onnx"
KOKORO_VOICES_PATH = "models/voices-v1.0.bin"
KOKORO_VOICE = "af_heart"
KOKORO_LANG = "en-us"
KOKORO_SPEED = 1.0
AI_SPEAKING_HANGOVER_S = 0.4  # keep mic muted this long after playback ends, for speaker/room echo decay

BLOCK_MS = 30
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_MS / 1000)
ENERGY_THRESHOLD = 0.012          # RMS above this counts as speech
SILENCE_HANGOVER_BLOCKS = 23      # ~0.7s of silence ends an utterance
MIN_UTTERANCE_SECONDS = 0.3

SYSTEM_PROMPT = (
    "You are an AI agent that helps guide and teach C++. You are a patient, "
    "knowledgeable tutor having a spoken conversation with a learner -- they "
    "will ask questions out loud, sometimes casually or imprecisely phrased, "
    "and you help them understand C++ concepts and write correct code.\n\n"
    "Every answer must include a relevant, runnable code example in a fenced "
    "code block with a language tag -- even for conceptual questions, ground "
    "the explanation in a short snippet that demonstrates the idea. Structure "
    "each answer as: one or two sentences explaining the concept or approach, "
    "then the code example, then at most one short sentence calling out the "
    "key takeaway if it's not obvious from the code.\n\n"
    "Keep answers tight and spoken-friendly -- no walls of text, no giant "
    "bullet lists, no headers. Since this is voice, never respond with just "
    "a bare word or a code block with no explanation. If the question is "
    "ambiguous, briefly note the ambiguity and answer the most likely "
    "interpretation rather than asking a clarifying question back."
)

FEWSHOT_EXAMPLES = [
    {"role": "user", "content": "What even is a smart pointer in C++?"},
    {
        "role": "assistant",
        "content": (
            "A smart pointer is a class that wraps a raw pointer and automatically frees the "
            "memory it owns when it goes out of scope, so you stop having to call delete "
            "yourself.\n"
            "```cpp\n"
            "#include <memory>\n\n"
            "void example() {\n"
            "    std::unique_ptr<int> p = std::make_unique<int>(42);\n"
            "    std::cout << *p << std::endl;\n"
            "}  // p's memory is freed automatically here\n"
            "```\n"
            "`unique_ptr` is the default choice; reach for `shared_ptr` only when multiple "
            "owners genuinely need to share the same object."
        ),
    },
    {"role": "user", "content": "How do I even loop over a map and print the keys and values?"},
    {
        "role": "assistant",
        "content": (
            "Range-based for loops work great on `std::map` since each element is a "
            "`std::pair<const Key, Value>`.\n"
            "```cpp\n"
            "#include <map>\n"
            "#include <iostream>\n\n"
            "void printMap(const std::map<std::string, int>& m) {\n"
            "    for (const auto& [key, value] : m) {\n"
            "        std::cout << key << \": \" << value << std::endl;\n"
            "    }\n"
            "}\n"
            "```"
        ),
    },
    {"role": "user", "content": "Why would I ever use a reference instead of just a pointer?"},
    {
        "role": "assistant",
        "content": (
            "References can't be null and can't be reseated after initialization, so they "
            "communicate 'this always refers to a valid object' more strongly than a pointer "
            "does, with cleaner call-site syntax.\n"
            "```cpp\n"
            "void increment(int& x) { x++; }        // reference: no null check needed\n"
            "void incrementPtr(int* x) {              // pointer: caller could pass nullptr\n"
            "    if (x) (*x)++;\n"
            "}\n"
            "```\n"
            "Prefer references for function parameters unless you need optional-ness or "
            "reseating, in which case a pointer (or `std::optional`) is the right tool."
        ),
    },
]

CODE_FENCE_DELIM = "```"
INLINE_CODE_RE = re.compile(r"`([^`]*)`")
MD_SYMBOLS_RE = re.compile(r"[#*_>-]")
SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")
MIN_TTS_CHUNK_CHARS = 12  # merge only pathological fragments (bare list markers like "1.") -- real sentences clear this easily


def extract_ready_speech(pending: str, min_chars: int = MIN_TTS_CHUNK_CHARS):
    """Pull off a chunk of `pending` ending on a sentence boundary, but only
    once it's at least `min_chars` long -- short fragments (like a lone '1.'
    from a list) don't give the TTS model enough context and come out
    garbled. Returns (chunk_or_None, remainder)."""
    matches = list(SENTENCE_END_RE.finditer(pending))
    if not matches:
        return None, pending
    last_end = matches[-1].end()
    if last_end < min_chars:
        return None, pending
    return pending[:last_end], pending[last_end:]


def prose_so_far(buffer: str) -> str:
    """Return the buffer text that lies outside fenced code blocks. Text
    inside a fence that hasn't closed yet is withheld -- we don't yet know
    if/when it'll close, so it's never spoken until it does."""
    parts = buffer.split(CODE_FENCE_DELIM)
    if len(parts) % 2 == 1:
        prose_parts = parts[0::2]        # buffer currently outside any fence
    else:
        prose_parts = parts[0:-1:2]      # last part is inside an open fence -- drop it
    return "".join(prose_parts)


def clean_for_speech(text: str) -> str:
    """Strip markdown decoration but keep inline-code identifiers (unwrapped)
    so sentences like 'use `unique_ptr` here' still read naturally."""
    text = INLINE_CODE_RE.sub(r"\1", text)
    text = MD_SYMBOLS_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def queue_speech(text: str):
    if not TTS_ENABLED or kokoro is None:
        return
    sentence = clean_for_speech(text)
    if sentence:
        _tts_queue.put(sentence)


def barge_in():
    """One Left Option press while the AI is talking: cut it off immediately
    and keep listening, instead of waiting for it to finish."""
    global ai_speaking
    _cancel_speech.set()
    while True:
        try:
            _tts_queue.get_nowait()
        except queue.Empty:
            break
    if _tts_stream is not None and _tts_stream.active:
        _tts_stream.abort()
    ai_speaking = False
    console.print("\n[dim]interrupted -- listening[/dim]\n")


_tts_stream = None


def _get_tts_stream(samplerate: int) -> sd.OutputStream:
    """A fresh sd.play() per sentence opens a new output stream each time,
    and that stream's startup/buffer-fill transient is what was garbling the
    first moment of every utterance. Keep one stream open for the process
    lifetime and write to it instead."""
    global _tts_stream
    if _tts_stream is None:
        _tts_stream = sd.OutputStream(samplerate=samplerate, channels=1, dtype="float32")
        _tts_stream.start()
    return _tts_stream


def _tts_worker():
    """Runs on its own thread so synthesis+playback of one sentence overlaps
    with the model still generating/streaming the next one.

    Sets ai_speaking around playback so the mic capture callback can gate
    itself out -- otherwise the mic picks up Kokoro's own output from the
    speakers, transcribes it, and feeds it back into the LLM in a loop."""
    global ai_speaking, _in_speech, _silence_run
    while True:
        sentence = _tts_queue.get()
        if _cancel_speech.is_set():
            continue  # stray sentence from a turn that got barged in on
        ai_speaking = True
        with _lock:
            _utterance_chunks.clear()
            _in_speech = False
            _silence_run = 0
        try:
            samples, sr = kokoro.create(sentence, voice=KOKORO_VOICE, speed=KOKORO_SPEED, lang=KOKORO_LANG)
            if _cancel_speech.is_set():
                continue
            stream = _get_tts_stream(sr)
            stream.write(samples.astype(np.float32).reshape(-1, 1))
        except Exception as e:
            console.print(f"[dim]TTS error: {e}[/dim]")
        finally:
            if _tts_queue.empty():
                time.sleep(AI_SPEAKING_HANGOVER_S)
                if _tts_queue.empty():
                    ai_speaking = False


# --- State ------------------------------------------------------------------

console = Console()
history = [{"role": "system", "content": SYSTEM_PROMPT}] + FEWSHOT_EXAMPLES
kokoro = None
ai_speaking = False
_cancel_speech = threading.Event()

_listening = False
_stream = None
_lock = threading.Lock()
_audio_queue = queue.Queue()
_tts_queue = queue.Queue()

_utterance_chunks = []
_in_speech = False
_silence_run = 0


def _finalize_utterance():
    """Must hold _lock when calling. Queues the buffered utterance (if any)
    for the worker thread and resets VAD state."""
    global _utterance_chunks, _in_speech, _silence_run
    if _utterance_chunks:
        audio = np.concatenate(_utterance_chunks)
        if len(audio) / SAMPLE_RATE >= MIN_UTTERANCE_SECONDS:
            _audio_queue.put(audio)
    _utterance_chunks = []
    _in_speech = False
    _silence_run = 0


def _audio_callback(indata, frames, time_info, status):
    """Runs on PortAudio's own realtime thread -- separate from the OS
    keyboard tap, so it's safe to do lightweight VAD work here."""
    global _in_speech, _silence_run
    if status:
        console.print(f"[dim]{status}[/dim]")

    if ai_speaking:
        # AI is talking (or just finished) -- ignore mic input entirely so its
        # own voice from the speakers can't get transcribed and fed back in.
        with _lock:
            _utterance_chunks.clear()
            _in_speech = False
            _silence_run = 0
        return

    chunk = indata[:, 0].copy()
    rms = float(np.sqrt(np.mean(chunk**2)))

    with _lock:
        if rms > ENERGY_THRESHOLD:
            _in_speech = True
            _silence_run = 0
            _utterance_chunks.append(chunk)
        elif _in_speech:
            _utterance_chunks.append(chunk)
            _silence_run += 1
            if _silence_run >= SILENCE_HANGOVER_BLOCKS:
                _finalize_utterance()
        # else: silence before any speech started -- discard


def start_listening():
    global _listening, _stream
    with _lock:
        if _listening:
            return
        _listening = True
    _stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=BLOCK_SIZE,
        callback=_audio_callback,
    )
    _stream.start()
    console.print("\n[bold red]● listening...[/bold red] (Left Option to stop)\n")


def stop_listening():
    global _listening, _stream
    with _lock:
        if not _listening:
            return
        _listening = False
    _stream.stop()
    _stream.close()
    _stream = None
    with _lock:
        _finalize_utterance()
    console.print("\n[dim]stopped listening (Left Option to resume)[/dim]\n")



def worker_loop():
    while True:
        audio = _audio_queue.get()
        process_audio(audio)


def process_audio(audio: np.ndarray):
    transcript = transcribe(audio)
    if not transcript.strip():
        return

    console.print(f"[bold cyan]You:[/bold cyan] {transcript}")
    ask_model(transcript)


def transcribe(audio: np.ndarray) -> str:
    segments, _ = whisper.transcribe(audio, language="en", beam_size=1, vad_filter=True)
    return " ".join(seg.text.strip() for seg in segments)


def ask_model(user_text: str):
    _cancel_speech.clear()
    history.append({"role": "user", "content": user_text})

    console.print("[bold green]Gemma:[/bold green]")
    buffer = ""
    spoken_prose_len = 0
    pending = ""
    with Live(Markdown(""), console=console, auto_refresh=False, vertical_overflow="visible") as live:
        stream = ollama.chat(
            model=OLLAMA_MODEL,
            messages=history,
            stream=True,
            keep_alive=KEEP_ALIVE,
            options={"num_ctx": NUM_CTX, "temperature": TEMPERATURE},
        )
        for chunk in stream:
            if _cancel_speech.is_set():
                break  # barged in -- stop generating, we're not speaking the rest anyway
            piece = chunk.get("message", {}).get("content", "")
            if not piece:
                continue
            buffer += piece
            live.update(Markdown(buffer))
            live.refresh()

            full_prose = prose_so_far(buffer)
            new_prose = full_prose[spoken_prose_len:]
            if new_prose:
                spoken_prose_len = len(full_prose)
                pending += new_prose
                ready, pending = extract_ready_speech(pending)
                if ready:
                    queue_speech(ready)

    if not _cancel_speech.is_set():
        full_prose = prose_so_far(buffer)
        pending += full_prose[spoken_prose_len:]
        if pending.strip():
            queue_speech(pending)

    history.append({"role": "assistant", "content": buffer})


def on_press(key):
    if key != TOGGLE_KEY:
        return
    if ai_speaking:
        barge_in()
    elif _listening:
        stop_listening()
    else:
        start_listening()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--no-tts", action="store_true",
        help="Disable spoken responses -- text-only, like the old behavior.",
    )
    parser.add_argument(
        "--system-prompt", type=str, default=None,
        help="Override the system prompt (default: built-in C++ tutor persona). "
             "Replaces the few-shot examples too, since those are C++-specific.",
    )
    return parser.parse_args()


def main():
    global whisper, kokoro, history, TTS_ENABLED
    args = parse_args()
    TTS_ENABLED = not args.no_tts
    if args.system_prompt:
        history = [{"role": "system", "content": args.system_prompt}]

    console.print(f"[dim]Loading Whisper ({WHISPER_MODEL_SIZE})...[/dim]")
    whisper = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type=WHISPER_COMPUTE_TYPE)

    if TTS_ENABLED:
        console.print("[dim]Loading Kokoro TTS...[/dim]")
        kokoro = Kokoro(KOKORO_MODEL_PATH, KOKORO_VOICES_PATH)

    console.print(f"[dim]Warming up {OLLAMA_MODEL} in Ollama...[/dim]")
    try:
        ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            keep_alive=KEEP_ALIVE,
            options={"num_ctx": NUM_CTX, "temperature": TEMPERATURE},
        )
    except Exception as e:
        console.print(f"[bold red]Could not reach Ollama:[/bold red] {e}")
        console.print("Is `ollama serve` running and have you run `ollama pull {}`?".format(OLLAMA_MODEL))
        sys.exit(1)

    console.print(
        "[bold]Ready.[/bold] Press [bold]Left Option[/bold] to start listening, press it again to stop. "
        "While the AI is talking, press it once to cut it off and keep listening. Ctrl+C to quit.\n"
    )

    threading.Thread(target=worker_loop, daemon=True).start()
    if TTS_ENABLED:
        threading.Thread(target=_tts_worker, daemon=True).start()

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    try:
        while listener.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        console.print("\n[dim]Bye.[/dim]")
        listener.stop()


if __name__ == "__main__":
    main()
