import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from modules.transcription import transcribe_with_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeWhisperModel:
    device = "cpu"

    def __init__(self):
        self.options = None

    def transcribe(self, audio_path, **options):
        self.options = options
        return {"text": "  A sample spoken explanation.  "}


class TranscriptionTests(unittest.TestCase):
    def test_wrapper_returns_transcript_and_language_option(self):
        model = FakeWhisperModel()
        with tempfile.NamedTemporaryFile(
            dir=PROJECT_ROOT / "outputs", suffix=".wav", delete=False
        ) as handle:
            path = Path(handle.name)
        try:
            sf.write(path, np.zeros(8000, dtype=np.float32), 8000)
            result = transcribe_with_model(model, path, language="english")
        finally:
            path.unlink(missing_ok=True)
        self.assertEqual(result, "A sample spoken explanation.")
        self.assertEqual(model.options["language"], "english")
        self.assertFalse(model.options["fp16"])


if __name__ == "__main__":
    unittest.main()
