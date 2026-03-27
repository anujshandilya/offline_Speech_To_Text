import streamlit as st
import whisper
from pathlib import Path
import tempfile

# Title
st.title("Audio Transcription Dashboard")

# Load Whisper model (only once)
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# Supported audio formats
AUDIO_EXTENSIONS = ["wav", "mp3", "m4a", "flac", "ogg", "webm"]

# Upload audio
uploaded_file = st.file_uploader(
    "Upload an audio recording",
    type=AUDIO_EXTENSIONS
)

if uploaded_file is not None:

    st.audio(uploaded_file)

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_audio_path = tmp_file.name

    if st.button("Transcribe Audio"):

        with st.spinner("Transcribing audio..."):
            result = model.transcribe(temp_audio_path)

        transcription = result["text"].strip()

        st.success("Transcription Completed")

        st.subheader("Transcription")
        st.text_area("Text Output", transcription, height=250)

        # Download button
        st.download_button(
            label="Download Transcription",
            data=transcription,
            file_name="transcription.txt",
            mime="text/plain"
        )