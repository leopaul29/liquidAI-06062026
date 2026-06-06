from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import time
import wave
from pathlib import Path

import gradio as gr
import numpy as np


APP_TITLE = "NyanGuard AI"
NYAN_TOKEN = "Nyan Nyan"
DEFAULT_SAMPLE_RATE = 24_000

TOXIC_PATTERNS = [
    r"\b(?:idiot|stupid|dumb|moron|loser|trash|garbage|baka)\b",
    r"\b(?:kill yourself|kys|shut up|go die)\b",
    r"\b(?:hate you|you suck|worthless)\b",
    (
        r"\b(?:\u3070\u304b|\u30d0\u30ab|\u99ac\u9e7f|\u6b7b\u306d|"
        r"\u304d\u3082\u3044|\u6d88\u3048\u308d|\u3046\u3056\u3044|"
        r"\u30af\u30ba|\u304f\u305d|\u30d0\u30fc\u30ab)\b"
    ),
]


def _safe_path(file_obj) -> str | None:
    if file_obj is None:
        return None
    if isinstance(file_obj, str):
        return file_obj
    return getattr(file_obj, "name", None)


def _extract_audio_if_needed(input_path: str, workdir: Path) -> str:
    suffix = Path(input_path).suffix.lower()
    if suffix in {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus"}:
        return input_path

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "Video upload requires ffmpeg. Install ffmpeg or upload an audio file."
        )

    audio_path = workdir / "extracted_audio.wav"
    command = [
        ffmpeg,
        "-y",
        "-i",
        input_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(DEFAULT_SAMPLE_RATE),
        str(audio_path),
    ]
    subprocess.run(command, check=True, capture_output=True)
    return str(audio_path)


def _transcribe_audio(audio_path: str, model_size: str, language: str) -> tuple[str, str]:
    try:
        from faster_whisper import WhisperModel
    except Exception as exc:
        return "", f"ASR unavailable: faster-whisper is not installed ({exc})."

    started = time.perf_counter()
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        audio_path,
        language=None if language == "auto" else language,
        vad_filter=True,
        beam_size=1,
    )
    transcript = " ".join(segment.text.strip() for segment in segments).strip()
    elapsed_ms = (time.perf_counter() - started) * 1000
    detected = getattr(info, "language", "unknown")
    return transcript, f"Local ASR: faster-whisper/{model_size}, language={detected}, {elapsed_ms:.0f} ms."


def _moderate_text(text: str, replacement: str) -> tuple[str, int]:
    sanitized = text
    total_hits = 0
    for pattern in TOXIC_PATTERNS:
        sanitized, hits = re.subn(
            pattern,
            replacement,
            sanitized,
            flags=re.IGNORECASE,
        )
        total_hits += hits
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    return sanitized, total_hits


def _write_wave(path: Path, audio: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE) -> str:
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return str(path)


def _tone(frequency: float, duration: float, sample_rate: int) -> np.ndarray:
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    envelope = np.minimum(1.0, np.linspace(0, 8, t.size))
    envelope *= np.minimum(1.0, np.linspace(8, 0, t.size))
    return 0.23 * np.sin(2 * np.pi * frequency * t) * envelope


def _synthesize_nyan_audio(text: str, path: Path) -> str:
    words = re.findall(r"[\w\u3040-\u30ff\u4e00-\u9fff']+|[.,!?]", text)
    chunks: list[np.ndarray] = []
    for token in words or [NYAN_TOKEN]:
        normalized = token.lower()
        if normalized in {"nyan", "\u30cb\u30e3\u30f3", "\u306b\u3083\u3093"}:
            chunks.append(_tone(880, 0.16, DEFAULT_SAMPLE_RATE))
            chunks.append(_tone(1174.66, 0.18, DEFAULT_SAMPLE_RATE))
        elif token in {".", ",", "!", "?"}:
            chunks.append(np.zeros(int(DEFAULT_SAMPLE_RATE * 0.12)))
        else:
            frequency = 360 + (sum(ord(char) for char in token) % 220)
            chunks.append(_tone(frequency, 0.08, DEFAULT_SAMPLE_RATE))
            chunks.append(np.zeros(int(DEFAULT_SAMPLE_RATE * 0.025)))
    audio = np.concatenate(chunks) if chunks else np.zeros(DEFAULT_SAMPLE_RATE // 2)
    return _write_wave(path, audio)


def _synthesize_tts(text: str, output_path: Path, voice_rate: int) -> tuple[str, str]:
    try:
        import pyttsx3
    except Exception as exc:
        return _synthesize_nyan_audio(text, output_path), f"Fallback synth: pyttsx3 unavailable ({exc})."

    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", voice_rate)
        engine.save_to_file(text or NYAN_TOKEN, str(output_path))
        engine.runAndWait()
        if output_path.exists() and output_path.stat().st_size > 0:
            return str(output_path), "Local TTS: pyttsx3."
    except Exception as exc:
        return _synthesize_nyan_audio(text, output_path), f"Fallback synth: pyttsx3 failed ({exc})."

    return _synthesize_nyan_audio(text, output_path), "Fallback synth: generated NyanGuard tone track."


def process_media(
    media_file,
    optional_transcript: str,
    replacement: str,
    model_size: str,
    language: str,
    voice_rate: int,
) -> tuple[str, str, str, str]:
    input_path = _safe_path(media_file)
    if not input_path and not optional_transcript.strip():
        raise gr.Error("Upload an audio/video file or paste a transcript.")

    replacement = replacement.strip() or NYAN_TOKEN
    workdir = Path(tempfile.mkdtemp(prefix="nyanguard_"))
    status: list[str] = []

    started = time.perf_counter()
    transcript = optional_transcript.strip()

    if input_path and not transcript:
        audio_path = _extract_audio_if_needed(input_path, workdir)
        transcript, asr_status = _transcribe_audio(audio_path, model_size, language)
        status.append(asr_status)
        if not transcript:
            transcript = (
                "Demo transcript: this chat message is stupid and should go away. "
                "NyanGuard keeps the stream friendly."
            )
            status.append("Demo transcript used because local ASR did not return text.")
    elif optional_transcript.strip():
        status.append("Transcript source: manual input.")

    sanitized_text, hits = _moderate_text(transcript, replacement)
    output_audio = workdir / "nyanguard_moderated.wav"
    audio_path, tts_status = _synthesize_tts(sanitized_text, output_audio, voice_rate)
    status.append(tts_status)

    elapsed_ms = (time.perf_counter() - started) * 1000
    report = "\n".join(
        [
            f"Moderation hits: {hits}",
            f"Replacement token: {replacement}",
            f"End-to-end demo latency: {elapsed_ms:.0f} ms",
            *status,
            "Privacy mode: all processing is local to this machine.",
        ]
    )
    return transcript, sanitized_text, audio_path, report


def build_demo() -> gr.Blocks:
    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(
            """
            <div class="nyan-hero">
              <h1>NyanGuard AI</h1>
              <p>Local real-time moderation demo for AITuber and VTuber streams.</p>
            </div>
            """
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                media_file = gr.File(
                    label="Upload audio or video",
                    file_types=[
                        ".wav",
                        ".mp3",
                        ".m4a",
                        ".aac",
                        ".flac",
                        ".ogg",
                        ".opus",
                        ".mp4",
                        ".mov",
                        ".mkv",
                        ".webm",
                    ],
                )
                optional_transcript = gr.Textbox(
                    label="Optional transcript",
                    placeholder="Paste a transcript to skip ASR during the demo...",
                    lines=5,
                )
                replacement = gr.Textbox(
                    label="Replacement phrase",
                    value=NYAN_TOKEN,
                )
                with gr.Accordion("Local model settings", open=False):
                    model_size = gr.Dropdown(
                        label="ASR model",
                        choices=["tiny", "base", "small", "medium"],
                        value="tiny",
                    )
                    language = gr.Dropdown(
                        label="Language",
                        choices=["auto", "en", "ja", "fr"],
                        value="auto",
                    )
                    voice_rate = gr.Slider(
                        label="Local TTS speed",
                        minimum=120,
                        maximum=240,
                        value=185,
                        step=5,
                    )
                run_button = gr.Button("Moderate stream audio", variant="primary")

            with gr.Column(scale=1):
                output_audio = gr.Audio(label="Moderated playable audio", type="filepath")
                report = gr.Textbox(label="Runtime report", lines=7, elem_classes=["metric-box"])

        with gr.Row():
            transcript = gr.Textbox(label="Detected transcript", lines=6)
            sanitized_text = gr.Textbox(label="Sanitized transcript", lines=6)

        gr.Examples(
            examples=[
                [
                    None,
                    "You are stupid and this stream is trash. baka. But the music is great!",
                    NYAN_TOKEN,
                    "tiny",
                    "auto",
                    185,
                ],
                [
                    None,
                    "Welcome everyone. Please stop saying kill yourself in chat. Let's keep it friendly.",
                    NYAN_TOKEN,
                    "tiny",
                    "en",
                    185,
                ],
            ],
            inputs=[
                media_file,
                optional_transcript,
                replacement,
                model_size,
                language,
                voice_rate,
            ],
        )

        run_button.click(
            fn=process_media,
            inputs=[
                media_file,
                optional_transcript,
                replacement,
                model_size,
                language,
                voice_rate,
            ],
            outputs=[transcript, sanitized_text, output_audio, report],
        )

    return demo


def launch_demo() -> None:
    css = """
    .nyan-hero {
        border-bottom: 1px solid #e7e7e7;
        padding-bottom: 14px;
        margin-bottom: 12px;
    }
    .metric-box textarea {
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    """
    build_demo().launch(theme=gr.themes.Soft(), css=css)


if __name__ == "__main__":
    launch_demo()
