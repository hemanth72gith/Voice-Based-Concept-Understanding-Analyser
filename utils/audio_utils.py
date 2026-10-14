"""Audio decoding, FFmpeg discovery, and safe temporary upload handling."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

_TEMP_ROOT = Path(__file__).resolve().parent.parent / "outputs"


class FfmpegUnavailableError(RuntimeError):
    """Raised when FFmpeg is needed but cannot be found."""


class AudioDecodeError(ValueError):
    """Raised when an audio file cannot be decoded."""


def ensure_ffmpeg_available() -> str:
    """Find FFmpeg, including the executable distributed by imageio-ffmpeg."""
    configured = os.getenv("VBCUA_FFMPEG_PATH")
    if configured:
        configured_path = Path(configured).expanduser()
        if configured_path.is_file():
            os.environ["PATH"] = str(configured_path.parent) + os.pathsep + os.environ.get("PATH", "")
            return str(configured_path)
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg

        bundled_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        os.environ["PATH"] = str(Path(bundled_ffmpeg).parent) + os.pathsep + os.environ.get("PATH", "")
        return bundled_ffmpeg
    except Exception as exc:
        raise FfmpegUnavailableError(
            "FFmpeg is needed to decode this format and for Whisper transcription. Install the "
            "project requirements (which include imageio-ffmpeg), install FFmpeg and add it to "
            "PATH, or set VBCUA_FFMPEG_PATH to ffmpeg.exe."
        ) from exc


@contextmanager
def save_uploaded_audio(audio_bytes: bytes, suffix: str = ".wav"):
    """Write upload bytes to a temporary path and always remove the file afterwards."""
    suffix = suffix.lower()
    if not suffix.startswith(".") or len(suffix) > 12 or not suffix[1:].isalnum():
        suffix = ".wav"
    temp_path = None
    try:
        _TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix="vbcua_", suffix=suffix, delete=False, dir=_TEMP_ROOT
        ) as handle:
            handle.write(audio_bytes)
            temp_path = handle.name
        yield temp_path
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def load_audio_mono(audio_path: str | Path, sample_rate_hz: int | None = None):
    """Decode via SoundFile, then Librosa or FFmpeg when the format needs them."""
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise AudioDecodeError("The audio file could not be found.")
    try:
        # SoundFile is fast and covers WAV/FLAC/OGG and MP3 where libsndfile has
        # MP3 support. Librosa and FFmpeg below handle other readable formats.
        samples, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
        samples = samples.mean(axis=1)
        if sample_rate_hz and sample_rate != sample_rate_hz:
            samples = _resample(samples, int(sample_rate), int(sample_rate_hz))
            sample_rate = sample_rate_hz
    except Exception:
        try:
            samples, sample_rate = librosa.load(
                str(audio_path), sr=sample_rate_hz, mono=True
            )
        except Exception as librosa_error:
            initial_error = librosa_error
            try:
                ffmpeg = ensure_ffmpeg_available()
                target_rate = sample_rate_hz or 22050
                _TEMP_ROOT.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    prefix="vbcua_decode_", suffix=".wav", delete=False, dir=_TEMP_ROOT
                ) as handle:
                    wav_path = Path(handle.name)
                try:
                    completed = subprocess.run(
                        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(audio_path),
                         "-vn", "-ac", "1", "-ar", str(target_rate), str(wav_path)],
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=120,
                    )
                    if completed.returncode != 0 or not wav_path.exists():
                        details = completed.stderr.strip() or str(initial_error)
                        raise AudioDecodeError(f"FFmpeg could not decode this audio: {details}")
                    samples, sample_rate = sf.read(
                        str(wav_path), dtype="float32", always_2d=True
                    )
                    samples = samples.mean(axis=1)
                finally:
                    wav_path.unlink(missing_ok=True)
            except FfmpegUnavailableError as exc:
                raise AudioDecodeError(
                    f"This audio format needs FFmpeg. Install the project requirements or provide "
                    f"a WAV file. Details: {exc}"
                ) from exc
            except AudioDecodeError:
                raise
            except Exception as exc:
                raise AudioDecodeError(f"Could not decode this recording: {exc}") from exc
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size == 0:
        raise AudioDecodeError("The uploaded audio contains no decodable samples.")
    return samples, int(sample_rate)


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample with Librosa when available, falling back to SciPy polyphase."""
    try:
        return librosa.resample(samples, orig_sr=source_rate, target_sr=target_rate)
    except Exception:
        from math import gcd
        from scipy.signal import resample_poly

        divisor = gcd(source_rate, target_rate)
        return resample_poly(samples, target_rate // divisor, source_rate // divisor).astype(
            np.float32
        )
