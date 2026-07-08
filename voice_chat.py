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

import queue
import sys
import threading
import time

import numpy as np
import ollama
import sounddevice as sd
from faster_whisper import WhisperModel
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

# --- State ------------------------------------------------------------------

console = Console()
history = [{"role": "system", "content": SYSTEM_PROMPT}] + FEWSHOT_EXAMPLES

_listening = False
_stream = None
_lock = threading.Lock()
_audio_queue = queue.Queue()

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
    console.print("\n[bold red]● listening...[/bold red] (Right Option to stop)\n")


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
    history.append({"role": "user", "content": user_text})

    console.print("[bold green]Gemma:[/bold green]")
    buffer = ""
    with Live(Markdown(""), console=console, auto_refresh=False, vertical_overflow="visible") as live:
        stream = ollama.chat(
            model=OLLAMA_MODEL,
            messages=history,
            stream=True,
            keep_alive=KEEP_ALIVE,
            options={"num_ctx": NUM_CTX, "temperature": TEMPERATURE},
        )
        for chunk in stream:
            piece = chunk.get("message", {}).get("content", "")
            if not piece:
                continue
            buffer += piece
            live.update(Markdown(buffer))
            live.refresh()

    history.append({"role": "assistant", "content": buffer})


def on_press(key):
    if key == TOGGLE_KEY:
        if _listening:
            stop_listening()
        else:
            start_listening()


def main():
    global whisper
    console.print(f"[dim]Loading Whisper ({WHISPER_MODEL_SIZE})...[/dim]")
    whisper = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type=WHISPER_COMPUTE_TYPE)

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
        "[bold]Ready.[/bold] Press [bold]Left Option[/bold] to start listening, press it again to stop. Ctrl+C to quit.\n"
    )

    threading.Thread(target=worker_loop, daemon=True).start()

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
