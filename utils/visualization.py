"""Matplotlib visualization helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def create_waveform_figure(samples: np.ndarray, sample_rate_hz: int, max_points: int = 100_000):
    """Create a memory-conscious waveform figure for Streamlit."""
    samples = np.asarray(samples).reshape(-1)
    if sample_rate_hz <= 0 or samples.size == 0:
        raise ValueError("A non-empty waveform and positive sample rate are required.")
    stride = max(1, int(np.ceil(samples.size / max_points)))
    plotted = samples[::stride]
    times = np.arange(plotted.size) * stride / sample_rate_hz
    figure, axis = plt.subplots(figsize=(10, 2.8), constrained_layout=True)
    axis.plot(times, plotted, color="#176b87", linewidth=0.55)
    axis.set_title("Audio waveform")
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("Amplitude")
    axis.grid(True, alpha=0.22)
    axis.set_ylim(-1.05, 1.05)
    return figure
