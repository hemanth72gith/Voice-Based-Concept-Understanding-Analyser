import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from modules.audio_analysis import analyze_audio_file, analyze_audio_samples

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AudioAnalysisTests(unittest.TestCase):
    def test_duration_rms_and_silence_ratio_on_synthetic_wav(self):
        sample_rate = 16000
        speech = 0.2 * np.sin(2 * np.pi * 220 * np.arange(sample_rate) / sample_rate)
        silence = np.zeros(sample_rate, dtype=np.float32)
        samples = np.concatenate([speech.astype(np.float32), silence])
        with tempfile.NamedTemporaryFile(
            dir=PROJECT_ROOT / "outputs", suffix=".wav", delete=False
        ) as handle:
            path = Path(handle.name)
        try:
            sf.write(path, samples, sample_rate)
            result = analyze_audio_file(str(path))
        finally:
            path.unlink(missing_ok=True)
        self.assertAlmostEqual(result["duration_seconds"], 2.0, places=2)
        self.assertGreater(result["rms_energy"], 0.05)
        self.assertLess(result["rms_energy"], 0.2)
        self.assertGreater(result["pause_ratio_percent"], 40)
        self.assertLess(result["pause_ratio_percent"], 60)
        self.assertEqual(result["sample_rate_hz"], sample_rate)

    def test_all_silence_is_measured_as_quiet(self):
        result = analyze_audio_samples(np.zeros(8000, dtype=np.float32), 8000)
        self.assertEqual(result["rms_energy"], 0)
        self.assertEqual(result["pause_ratio_percent"], 100)


if __name__ == "__main__":
    unittest.main()
