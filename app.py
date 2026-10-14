"""Streamlit interface for the Voice-Based Concept Understanding Analyser."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / "outputs" / ".matplotlib"))

import streamlit as st

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from modules.audio_analysis import AudioAnalysisError, analyze_audio_file
from modules.feedback import build_feedback
from modules.filler_analysis import analyze_fillers
from modules.report_generation import generate_pdf_report
from modules.scoring import calculate_score
from modules.semantic_analysis import (
    DEFAULT_EMBEDDING_MODEL,
    SemanticAnalysisError,
    calculate_semantic_similarity,
    load_sentence_model,
)
from modules.transcription import (
    TranscriptionError,
    load_whisper_model,
    transcribe_with_model,
)
from utils.audio_utils import save_uploaded_audio
from utils.visualization import create_waveform_figure


APP_DIR = PROJECT_DIR
DATA_FILE = APP_DIR / "data" / "reference_concepts.json"


def _positive_int_setting(name: str, default: int) -> int:
    try:
        parsed = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return parsed if parsed > 0 else default


MAX_UPLOAD_MB = _positive_int_setting("VBCUA_MAX_UPLOAD_MB", 50)
DEFAULT_WHISPER_MODEL = os.getenv("VBCUA_WHISPER_MODEL", "tiny")
EMBEDDING_MODEL = os.getenv("VBCUA_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
SUPPORTED_AUDIO_TYPES = ["wav", "mp3", "m4a", "flac", "ogg", "aac", "opus", "webm"]
AUDIO_MIME_TYPES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".aac": "audio/aac",
    ".opus": "audio/ogg",
    ".webm": "audio/webm",
}

st.set_page_config(
    page_title="VBCUA | Voice-Based Concept Understanding Analyser",
    page_icon="🎙️",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def cached_whisper_model(model_name: str):
    return load_whisper_model(model_name)


@st.cache_resource(show_spinner=False)
def cached_sentence_model(model_name: str):
    return load_sentence_model(model_name)


@st.cache_data(show_spinner=False)
def load_reference_examples() -> list[dict[str, str]]:
    """Load the small, editable-by-users demonstration reference set."""
    try:
        data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def _fingerprint(
    audio_bytes: bytes,
    audio_filename: str,
    reference: str,
    concept_name: str,
    whisper_model: str,
    language: str,
    embedding_model: str,
    weights: dict,
) -> str:
    digest = hashlib.sha256(audio_bytes)
    digest.update(
        json.dumps(
            [audio_filename, reference, concept_name, whisper_model, language, embedding_model, weights],
            sort_keys=True,
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _show_analysis(result: dict) -> None:
    st.subheader("Your analysis")
    score = result["score"]["overall_score"]
    st.metric("Overall practice score", f"{score:.1f} / 100")
    st.progress(int(round(score)), text="Weighted score using the components below")
    st.caption(
        "This is a transparent practice aid, not a scientifically validated or diagnostic assessment."
    )

    metrics = result["audio_metrics"]
    fillers = result["filler_analysis"]
    left, middle, right = st.columns(3)
    left.metric("Semantic similarity", f"{result['semantic']['similarity_percent']:.1f}%")
    middle.metric("Filler ratio", f"{fillers['filler_ratio_percent']:.1f}%")
    right.metric("Estimated pause ratio", f"{metrics['pause_ratio_percent']:.1f}%")

    with st.expander("Score breakdown and formulas", expanded=True):
        st.markdown(
            "The default weights are 70% semantic similarity, 20% filler fluency, and "
            "10% pause fluency. Component scores are multiplied by their normalized weights. "
            "Filler score falls linearly to zero at a 25% filler-token ratio. Pause score is "
            "100 through a 15% pause ratio, then falls linearly to zero at 65%."
        )
        rows = []
        for name, label in (
            ("semantic", "Semantic similarity"),
            ("fillers", "Filler fluency"),
            ("pauses", "Pause fluency"),
        ):
            rows.append(
                {
                    "Component": label,
                    "Score": f"{result['score']['components'][name]:.1f} / 100",
                    "Weight": f"{result['score']['weights'][name] * 100:.0f}%",
                    "Weighted points": f"{result['score']['weighted_points'][name]:.1f}",
                }
            )
        st.table(rows)

    with st.expander("Audio and filler details"):
        st.write(
            f"**Duration:** {metrics['duration_seconds']:.2f} seconds  ·  "
            f"**Sample rate:** {metrics['sample_rate_hz']} Hz  ·  "
            f"**RMS energy:** {metrics['rms_energy']:.5f}"
        )
        st.write(
            f"**Estimated quiet time:** {metrics['estimated_pause_seconds']:.2f} seconds. "
            f"Pause detection treats frames at least {metrics['silence_threshold_db']:.0f} dB "
            "below the clip's peak frame energy as quiet; it can mistake soft speech for a pause."
        )
        st.write(
            f"**Words counted:** {fillers['total_words']}  ·  "
            f"**Filler occurrences:** {fillers['filler_count']}  ·  "
            f"**Filler tokens:** {fillers['filler_tokens']}"
        )
        if fillers["occurrences"]:
            st.write("**Detected phrases:** " + ", ".join(
                f"{item['phrase']} ({item['count']})" for item in fillers["occurrences"]
            ))
        else:
            st.write("No configured filler phrases were found in the transcript.")

    st.markdown("#### Transcript")
    st.text_area("Transcription", result["transcript"], height=150, disabled=True, label_visibility="collapsed")
    st.markdown("#### Reference explanation")
    st.text_area("Reference", result["reference"], height=120, disabled=True, label_visibility="collapsed")

    feedback = result["feedback"]
    st.markdown(f"#### {feedback['headline']}")
    for item in feedback["strengths"]:
        st.markdown(f"- {item}")
    for item in feedback["suggestions"]:
        st.markdown(f"- {item}")

    try:
        pdf_bytes = generate_pdf_report(result)
        safe_name = "".join(c.lower() if c.isalnum() else "_" for c in result["concept_name"])
        safe_name = "_".join(part for part in safe_name.split("_") if part) or "concept"
        st.download_button(
            "Download PDF report",
            data=pdf_bytes,
            file_name=f"vbcua_{safe_name}_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as exc:  # keep analysis useful if a PDF font/render issue occurs
        st.error(f"The analysis is ready, but the PDF could not be generated: {exc}")


st.title("🎙️ Voice-Based Concept Understanding Analyser")
st.write(
    "Explain a concept aloud, compare your explanation with a reference, and get practical "
    "feedback on content, filler phrases, pauses, and basic audio levels."
)
st.markdown(
    "**How to use it:** 1. Choose or write a reference explanation. 2. Upload a recording. "
    "3. Run the analysis. 4. Review the breakdown and download a PDF."
)
st.caption(
    "Audio is analysed locally. The first run downloads the selected Whisper speech model and "
    "the Sentence Transformers embedding model; no API key is needed."
)

examples = load_reference_examples()
example_map = {item["title"]: item["explanation"] for item in examples}
reference_choices = ["Write my own"] + list(example_map)

reference_col, audio_col = st.columns([1, 1], gap="large")
with reference_col:
    st.subheader("1. Reference concept")
    selected_reference = st.selectbox("Load a demonstration reference (optional)", reference_choices)
    reference_key = "reference_text_" + str(reference_choices.index(selected_reference))
    reference_default = example_map.get(selected_reference, "")
    reference_text = st.text_area(
        "Reference explanation",
        value=reference_default,
        key=reference_key,
        height=185,
        placeholder="Write a clear explanation of the concept you will describe aloud...",
        help="The reference is used only for semantic comparison. You can edit a demonstration example.",
    )
    concept_default = selected_reference if selected_reference != "Write my own" else "My concept"
    concept_name = st.text_input("Concept name for the report", value=concept_default)

with audio_col:
    st.subheader("2. Recording")
    uploaded_audio = st.file_uploader(
        "Upload your spoken explanation",
        type=SUPPORTED_AUDIO_TYPES,
        help="WAV, MP3, M4A, FLAC, OGG, AAC, OPUS, and WEBM. Maximum size is "
        f"{MAX_UPLOAD_MB} MB. MP3 and other compressed formats need FFmpeg.",
    )
    whisper_model_name = st.selectbox(
        "Whisper model",
        options=["tiny", "base", "small"],
        index=["tiny", "base", "small"].index(DEFAULT_WHISPER_MODEL)
        if DEFAULT_WHISPER_MODEL in ["tiny", "base", "small"]
        else 0,
        help="Larger models usually take longer and use more memory. Weights download on first use.",
    )
    language = st.selectbox(
        "Spoken language",
        options=["Auto-detect", "English", "Hindi", "Spanish", "French", "German", "Japanese"],
        help="Choose Auto-detect or a language you are speaking for transcription.",
    )
    st.caption(
        "The default semantic model is English-oriented. For the closest comparison, use an "
        "English reference when selecting another transcription language."
    )

st.sidebar.header("Score weights")
st.sidebar.caption("Set relative importance. The three values are normalized to add up to 100%.")
semantic_weight = st.sidebar.number_input("Semantic similarity", 0, 100, 70, 5)
filler_weight = st.sidebar.number_input("Filler fluency", 0, 100, 20, 5)
pause_weight = st.sidebar.number_input("Pause fluency", 0, 100, 10, 5)
weights = {
    "semantic": float(semantic_weight),
    "fillers": float(filler_weight),
    "pauses": float(pause_weight),
}
if sum(weights.values()) == 0:
    st.sidebar.error("At least one score weight must be greater than zero.")

if uploaded_audio is None:
    st.info("Upload a recording to preview its waveform and begin the analysis.")
else:
    audio_bytes = uploaded_audio.getvalue()
    current_fingerprint = _fingerprint(
        audio_bytes,
        uploaded_audio.name,
        reference_text,
        concept_name,
        whisper_model_name,
        language,
        EMBEDDING_MODEL,
        weights,
    )
    extension = Path(uploaded_audio.name).suffix.lower() or ".wav"

    if len(audio_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"The uploaded file is larger than the {MAX_UPLOAD_MB} MB limit.")
    else:
        st.audio(audio_bytes, format=AUDIO_MIME_TYPES.get(extension, "audio/wav"))
        audio_result: dict | None = None
        try:
            with save_uploaded_audio(audio_bytes, extension) as audio_path:
                audio_result = analyze_audio_file(audio_path)
                figure = create_waveform_figure(
                    audio_result["waveform"], audio_result["sample_rate_hz"]
                )
                st.pyplot(figure, clear_figure=True)

                columns = st.columns(3)
                columns[0].metric("Duration", f"{audio_result['duration_seconds']:.2f} s")
                columns[1].metric("Estimated pauses", f"{audio_result['pause_ratio_percent']:.1f}%")
                columns[2].metric("RMS energy", f"{audio_result['rms_energy']:.5f}")

                if not reference_text.strip():
                    st.warning("Add a reference explanation before running the analysis.")
                if st.button(
                    "Analyse my explanation",
                    type="primary",
                    disabled=not reference_text.strip() or sum(weights.values()) == 0,
                    use_container_width=True,
                ):
                    st.session_state.pop("analysis_result", None)
                    try:
                        with st.spinner(
                            f"Loading Whisper '{whisper_model_name}' and transcribing. "
                            "The first use may download model files..."
                        ):
                            whisper_model = cached_whisper_model(whisper_model_name)
                            transcript = transcribe_with_model(
                                whisper_model,
                                audio_path,
                                language=None if language == "Auto-detect" else language.lower(),
                            )
                        filler_results = analyze_fillers(transcript)
                        with st.spinner(
                            f"Loading semantic model '{EMBEDDING_MODEL}'. "
                            "The first use may download model files..."
                        ):
                            semantic_results = calculate_semantic_similarity(
                                transcript,
                                reference_text,
                                model_name=EMBEDDING_MODEL,
                                model=cached_sentence_model(EMBEDDING_MODEL),
                            )
                        score_results = calculate_score(
                            semantic_results["similarity_percent"],
                            filler_results["filler_ratio_percent"],
                            audio_result["pause_ratio_percent"],
                            weights=weights,
                        )
                        feedback = build_feedback(
                            semantic_results["similarity_percent"],
                            filler_results["filler_ratio_percent"],
                            audio_result["pause_ratio_percent"],
                            score_results["overall_score"],
                        )
                        st.session_state["analysis_result"] = {
                            "fingerprint": current_fingerprint,
                            "concept_name": concept_name.strip() or "My concept",
                            "audio_filename": uploaded_audio.name,
                            "reference": reference_text.strip(),
                            "transcript": transcript,
                            "audio_metrics": {
                                key: value for key, value in audio_result.items() if key != "waveform"
                            },
                            "filler_analysis": filler_results,
                            "semantic": semantic_results,
                            "score": score_results,
                            "feedback": feedback,
                            "whisper_model": whisper_model_name,
                            "embedding_model": EMBEDDING_MODEL,
                        }
                    except (AudioAnalysisError, SemanticAnalysisError, TranscriptionError) as exc:
                        st.error(str(exc))
                    except Exception as exc:
                        st.error("The analysis did not finish. Review the technical detail below.")
                        with st.expander("Technical detail"):
                            st.code(f"{type(exc).__name__}: {exc}")
        except AudioAnalysisError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Could not preview this audio: {exc}")

    stored_result = st.session_state.get("analysis_result")
    if stored_result and stored_result.get("fingerprint") == current_fingerprint:
        _show_analysis(stored_result)
    elif stored_result:
        st.info("The inputs changed. Run the analysis again to refresh the report.")

