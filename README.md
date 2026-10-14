# Voice-Based Concept Understanding Analyser (VBCUA)

VBCUA is a beginner-friendly Streamlit project for practising spoken explanations. It transcribes a user's recording, compares the transcript with a reference explanation, checks configured filler phrases, estimates quiet time and basic audio energy, and creates a downloadable PDF report.

## Objectives

- Let a learner practise explaining a concept aloud.
- Compare the transcript with a reference supplied by the learner.
- Make the sources of feedback and the overall score visible.
- Keep the basic workflow local and free of API keys.

## Features

- WAV, MP3, M4A, FLAC, OGG, AAC, OPUS, and WEBM upload, subject to decoder support.
- Audio player and waveform preview.
- Local OpenAI Whisper transcription with tiny, base, and small model choices.
- Configurable filler phrase detection and filler-token ratio.
- Sentence Transformers cosine similarity against user text or a demo reference.
- Duration, mono RMS energy, and an estimated quiet-frame ratio.
- Adjustable score weights and a documented 0–100 practice score.
- Rule-based strengths and improvement suggestions.
- Downloadable PDF including the transcript, reference, measurements, score method, and feedback.
- Four example reference concepts in `data/reference_concepts.json`.

This score is a practice aid, not a scientifically validated assessment. Semantic similarity does not verify factual correctness. Pause detection is an energy-based estimate and can count soft speech as quiet.

## Architecture and technology

- **Streamlit** builds the dashboard and handles upload and downloads.
- **OpenAI Whisper** transcribes speech locally.
- **Sentence Transformers** creates text embeddings for semantic similarity.
- **Librosa** and **SoundFile** decode and measure audio; **imageio-ffmpeg** supplies FFmpeg on Windows for Whisper and formats that need conversion.
- **Matplotlib** draws the waveform.
- **ReportLab** creates the PDF.
- `modules/` contains transcription, semantic, filler, audio, scoring, feedback, and report logic. `utils/` holds audio and plotting helpers.

## Prerequisites

- Windows 10/11, 64-bit.
- Python 3.11 recommended for local and Cloud parity. The current project environment uses Python 3.11.0; dependency imports, unit tests, and Streamlit startup were checked there. Actual Whisper and Sentence Transformers model inference was not run during the deployment-readiness check.
- Internet for dependency installation and first-time model downloads. Model weights are cached locally afterward.

## Windows setup

Clone the repository and move into the project folder:

```powershell
git clone https://github.com/hemanth72gith/Voice-Based-Concept-Understanding-Analyser.git
Set-Location Voice-Based-Concept-Understanding-Analyser
```

Create and activate a virtual environment, then install the dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, allow it in the current terminal only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Optional local settings can be set by copying `.env.example` to `.env`. FFmpeg can be found on PATH, bundled by `imageio-ffmpeg`, or configured using `VBCUA_FFMPEG_PATH`. The application does not need an API key.

## Run the app

```powershell
python -m streamlit run app.py
```

Choose a sample reference or enter your own, upload a recording, review its waveform, and select **Analyse my explanation**. The default Whisper model is `tiny`; the first analysis downloads the Whisper weights and the default Sentence Transformers model.

## Deploy to Streamlit Community Cloud

The Streamlit entry point is the root-level `app.py`, and the dependencies are declared in the root-level `requirements.txt`. When creating the Cloud app, select this repository, the `main` branch, and `app.py`; choose Python 3.11 in Advanced settings to match the checked local environment. The app does not require secrets or API keys.

The first analysis downloads the Whisper and Sentence Transformers model files. Loaded models are cached for the life of the app process. Start with Whisper's `tiny` model on Community Cloud; the `base` and `small` choices use more memory and may be slow or exceed the available resources. Uploads are limited to 50 MB by the Streamlit configuration and the app's own validation.

## Run the tests

The unit tests use Python's standard `unittest` module:

```powershell
python -m unittest discover -s tests -v
```

They cover score formulas and weight normalization, filler detection, and audio metrics using a generated WAV fixture.

## Limitations and future enhancements

- Whisper and embedding model quality depends on the chosen model, recording, language, and hardware.
- Model files need an internet connection on first use and local disk space.
- Pause ratio is based on a configurable frame-energy threshold, not word-aligned speech activity detection.
- Filler detection is transcript-based and can miss disfluencies or mistake a listed word (such as “like”) used literally.
- The default `all-MiniLM-L6-v2` embedding model is English-oriented; use English reference text for the closest comparison, or configure a suitable multilingual model with `VBCUA_EMBEDDING_MODEL`.
- Similarity is not concept grading and cannot tell whether an explanation is correct.
- The score thresholds are transparent heuristics for practice, not validated educational measurements.
- Future work could add word timestamps, per-language filler lists, stronger speech activity detection, an editable rubric, and more detailed charts.

## Project layout

```text
app.py
modules/
utils/
data/reference_concepts.json
tests/
docs/
outputs/.gitkeep
```
