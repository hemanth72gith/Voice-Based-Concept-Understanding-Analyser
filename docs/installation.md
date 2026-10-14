# Windows installation

## Prerequisites

- 64-bit Windows 10 or 11.
- Python 3.11 recommended. The upstream Whisper project documents Python 3.8–3.11 as its expected range; the workspace used to develop this version had Python 3.14.4, where the app and unit tests run but the complete model workflow was not verified.
- Internet access for installing packages and for the first download of the Whisper and Sentence Transformers model weights.

## Set up the project

Open PowerShell in the project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If script activation is blocked by your PowerShell policy, activate for the current terminal with:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

`imageio-ffmpeg` provides an FFmpeg executable for Whisper and compressed audio decoding. The app can also use a system FFmpeg on PATH. To specify another path, copy `.env.example` to `.env` and set `VBCUA_FFMPEG_PATH` to the full path of `ffmpeg.exe`.

## Model downloads

The first transcription downloads Whisper's selected model (`tiny` by default). The first semantic comparison downloads `sentence-transformers/all-MiniLM-L6-v2`. These are local model files and can take time and disk space; later runs use the local caches. No API key is required.
