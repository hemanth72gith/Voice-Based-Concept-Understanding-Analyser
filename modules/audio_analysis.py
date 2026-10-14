"""Basic duration, RMS energy, and frame-energy pause estimates."""

from __future__ import annotations

import numpy as np

from utils.audio_utils import AudioDecodeError, load_audio_mono


class AudioAnalysisError(ValueError):
    """Raised when an audio file cannot be decoded or measured."""


def analyze_audio_samples(
    samples: np.ndarray,
    sample_rate_hz: int,
    *,
    silence_threshold_db: float = -40.0,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> dict:
    """Measure audio and estimate quiet-frame ratio relative to peak frame RMS."""
    y = np.asarray(samples, dtype=np.float32).reshape(-1)
    if sample_rate_hz <= 0:
        raise AudioAnalysisError("The audio sample rate must be greater than zero.")
    if y.size == 0 or not np.isfinite(y).all():
        raise AudioAnalysisError("The recording is empty or contains invalid audio samples.")
    if frame_length < 2 or hop_length < 1:
        raise AudioAnalysisError("Audio analysis frame and hop lengths must be positive.")

    duration = float(y.size / sample_rate_hz)
    rms_energy = float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))
    try:
        # Use Librosa's standard RMS framing when its optional JIT stack is usable.
        import librosa

        frame_rms = librosa.feature.rms(
            y=y, frame_length=frame_length, hop_length=hop_length, center=True
        )[0]
    except Exception:
        # The equivalent NumPy path keeps audio metrics available in restricted
        # Windows/Python environments where Numba cannot create its cache.
        half_frame = frame_length // 2
        padded = np.pad(y, (half_frame, frame_length - half_frame), mode="constant")
        frames = np.lib.stride_tricks.sliding_window_view(padded, frame_length)[::hop_length]
        frame_rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    peak_frame_rms = float(frame_rms.max(initial=0.0))
    if peak_frame_rms <= 1e-10:
        quiet_mask = np.ones(frame_rms.shape, dtype=bool)
    else:
        threshold = max(peak_frame_rms * (10.0 ** (silence_threshold_db / 20.0)), 1e-8)
        quiet_mask = frame_rms <= threshold
    pause_ratio = float(np.mean(quiet_mask) * 100.0) if quiet_mask.size else 0.0

    return {
        "duration_seconds": duration,
        "pause_ratio_percent": pause_ratio,
        "estimated_pause_seconds": duration * pause_ratio / 100.0,
        "rms_energy": rms_energy,
        "peak_amplitude": float(np.max(np.abs(y))),
        "sample_rate_hz": int(sample_rate_hz),
        "silence_threshold_db": float(silence_threshold_db),
        "frame_length_samples": int(frame_length),
        "hop_length_samples": int(hop_length),
    }


def analyze_audio_file(audio_path: str) -> dict:
    """Decode a path, measure it, and return samples for a waveform plot."""
    try:
        samples, sample_rate_hz = load_audio_mono(audio_path)
        metrics = analyze_audio_samples(samples, sample_rate_hz)
    except AudioAnalysisError:
        raise
    except AudioDecodeError as exc:
        raise AudioAnalysisError(str(exc)) from exc
    except Exception as exc:
        raise AudioAnalysisError(f"Could not analyse this recording: {exc}") from exc
    return {**metrics, "waveform": samples}
