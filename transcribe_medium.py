import whisper
from pathlib import Path

# Paths
INPUT_DIR = Path("recordings")
OUTPUT_DIR = Path("mdTXT")

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)

# Load Whisper model (offline)
model = whisper.load_model("medium")

# Supported audio extensions
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}

for audio_file in INPUT_DIR.iterdir():
    if audio_file.suffix.lower() in AUDIO_EXTENSIONS:
        print(f"Transcribing: {audio_file.name}")

        # Transcribe
        result = model.transcribe(str(audio_file))

        # Output markdown text file
        output_file = OUTPUT_DIR / f"{audio_file.stem}.md"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"# Transcription: {audio_file.name}\n\n")
            f.write(result["text"].strip())

        print(f"Saved → {output_file}")
