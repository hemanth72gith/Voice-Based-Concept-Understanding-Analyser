"""Local transcription with OpenAI Whisper; no API key is used."""

from __future__ import annotations

from pathlib import Path

from utils.audio_utils import FfmpegUnavailableError, ensure_ffmpeg_available


class TranscriptionError(RuntimeError):
    """Raised when local audio transcription cannot run."""


def load_whisper_model(model_name: str = "tiny"):
    """Load a Whisper model. The first call downloads model weights."""
    try:
        ensure_ffmpeg_available()
    except FfmpegUnavailableError as exc:
        raise TranscriptionError(str(exc)) from exc
    try:
        import whisper

        return whisper.load_model(model_name)
    except Exception as exc:
        raise TranscriptionError(
            f"Could not load Whisper model '{model_name}'. Its weights may need to be "
            f"downloaded on first use, and the computer needs enough memory. Details: {exc}"
        ) from exc


def transcribe_with_model(model, audio_path: str | Path, language: str | None = None) -> str:
    """Transcribe an audio path using a previously loaded Whisper model."""
    try:
        ensure_ffmpeg_available()
    except FfmpegUnavailableError as exc:
        raise TranscriptionError(str(exc)) from exc
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise TranscriptionError("The temporary audio file is no longer available. Upload it again.")
    try:
        options = {"fp16": "cuda" in str(getattr(model, "device", "cpu")).lower()}
        if language:
            options["language"] = language
        result = model.transcribe(str(audio_path), **options)
        text = str(result.get("text", "")).strip()
        if not text:
            raise TranscriptionError(
                "Whisper returned an empty transcript. Try a clearer or longer recording."
            )
        return text
    except TranscriptionError:
        raise
    except Exception as exc:
        raise TranscriptionError(
            f"Whisper could not transcribe this recording: {exc}. Check that FFmpeg is available "
            "and the audio is readable."
        ) from exc
