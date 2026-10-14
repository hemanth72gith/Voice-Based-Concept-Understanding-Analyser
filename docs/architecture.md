# VBCUA architecture

The Streamlit page in `app.py` collects an audio recording and a reference explanation. It saves the upload to a short-lived temporary file, decodes it with Librosa (using FFmpeg as a fallback), and displays a Matplotlib waveform.

The analysis path is:

1. `modules.transcription` runs local OpenAI Whisper. It uses `imageio-ffmpeg`'s bundled executable when a system FFmpeg is unavailable.
2. `modules.filler_analysis` finds a configured list of filler words and phrases with case-insensitive word-boundary matching. The ratio is matched filler tokens divided by transcript word tokens.
3. `modules.semantic_analysis` embeds the transcript and reference with Sentence Transformers (`all-MiniLM-L6-v2` by default) and computes cosine similarity. The UI clamps negative cosine values to 0 for a 0–100 display. This default model is English-oriented; set `VBCUA_EMBEDDING_MODEL` to a suitable multilingual model for other languages.
4. `modules.audio_analysis` measures duration and mono RMS energy. It estimates quiet frames by comparing frame RMS against a threshold 40 dB below the clip's peak frame RMS.
5. `modules.scoring` combines similarity, filler fluency, and pause fluency with editable sidebar weights. The default is 70/20/10; weights are normalized to sum to 100%.
6. `modules.feedback` produces simple threshold-based suggestions. `modules.report_generation` exports the result to PDF.

The score is intended for practice and is not a validated assessment. Semantic similarity does not establish factual correctness. Frame-energy pause detection is approximate and may classify soft speech as quiet.
