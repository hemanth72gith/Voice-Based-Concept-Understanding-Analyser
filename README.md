# Voice-Based Concept Understanding Analyser (VBCUA)

An AI-powered application designed to analyse spoken explanations of concepts and provide structured feedback on semantic relevance, speech fluency, pauses, and basic audio characteristics.

## Project Overview

The Voice-Based Concept Understanding Analyser (VBCUA) allows users to upload a spoken explanation and compare its transcription with a reference explanation. The application combines speech recognition, semantic similarity analysis, filler-word detection, and audio feature extraction to generate an interpretable evaluation report.

## Objectives

- Convert spoken explanations into text using automatic speech recognition.
- Compare a transcription with a reference concept using semantic similarity.
- Identify common filler words and estimate their frequency.
- Extract basic audio characteristics, including duration, pauses, and RMS energy.
- Provide an overall score and actionable feedback.
- Generate a downloadable PDF containing the analysis results.

## Features

- Streamlit-based user interface.
- Audio upload and playback.
- Speech-to-text transcription using Whisper.
- Semantic similarity analysis using Sentence Transformers.
- Filler-word analysis.
- Basic audio analysis using Librosa and SoundFile.
- Configurable evaluation weights.
- Visual presentation of analysis results.
- PDF report generation.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| User interface | Streamlit |
| Speech recognition | OpenAI Whisper |
| Semantic analysis | Sentence Transformers |
| Audio processing | Librosa, SoundFile |
| Visualization | Matplotlib |
| PDF generation | ReportLab |

## Application Workflow

1. Enter or select a reference concept.
2. Upload an audio recording of a spoken explanation.
3. Transcribe the audio.
4. Analyse semantic similarity between the transcription and reference.
5. Calculate filler-word and audio-related metrics.
6. Generate an overall score and feedback.
7. Review the results and download the PDF report.

## Project Structure

The application is organized into modules for transcription, semantic analysis, filler-word analysis, audio processing, scoring, feedback, visualization, and report generation.

Refer to the actual repository files for the current implementation and structure.

## Installation

### Prerequisites

- Python installed on your system.
- Git, if cloning the repository.
- Internet access for installing dependencies and downloading required models.

### Windows Setup

Clone the repository:

```bash
git clone https://github.com/hemant72gith/Voice-Based-Concept-Understanding-Analyser.git
cd Voice-Based-Concept-Understanding-Analyser
