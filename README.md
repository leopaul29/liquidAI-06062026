# NyanGuard AI Gradio Demo

NyanGuard AI is a local-first moderation demo for AITuber / VTuber live chat and audio streams.
It accepts an audio or video file, transcribes speech locally when a local ASR backend is available,
replaces toxic language with `Nyan Nyan`, and returns a playable moderated audio file.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
python app.py
```

Then open the local Gradio URL printed in the terminal.

## Local AI Notes

- `faster-whisper` is used for local speech recognition when installed. The first ASR run may download the selected Whisper model.
- `pyttsx3` is used for local text-to-speech when available.
- If ASR is not available, paste a transcript in the "Optional transcript" field to keep the demo fully local and deterministic.
- The app is designed so the model layer can be replaced by an AMD Ryzen AI / ONNX Runtime / LFM backend without changing the UI.
