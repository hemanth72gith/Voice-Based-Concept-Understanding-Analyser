# Viva preparation questions

1. **Why use Whisper?** It is a locally runnable speech recognition model, so the basic transcription workflow does not need an API key.
2. **What does semantic similarity mean here?** Sentence Transformers encodes the two texts as vectors and the app computes cosine similarity. It measures related wording and meaning, not factual accuracy.
3. **How are fillers counted?** A case-insensitive phrase list is matched with word boundaries. Multiword phrases count as one occurrence and their full number of tokens in the ratio.
4. **How does the pause estimate work?** The mono signal is split into overlapping frames. Frames at least 40 dB below the recording's peak frame RMS are treated as quiet.
5. **How is the total score calculated?** The app combines semantic similarity, a linear filler score, and a linear pause score using normalized weights. The default weights are 70%, 20%, and 10%.
6. **What are the main limitations?** Transcription can be wrong, sentence similarity can reward related phrasing without checking correctness, and quiet-frame detection can mistake soft speech for pauses.
7. **Why process locally?** Whisper and embedding inference run on the user's machine, which avoids sending the recording to an application API. The model files are downloaded from their upstream model sources on first use.
8. **How could the project be improved?** Add word-level timestamps, multilingual filler lists, noise-robust speech activity detection, concept-specific rubric checks, and accessibility support.
