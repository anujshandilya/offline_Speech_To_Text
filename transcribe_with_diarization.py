from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pathlib import Path

INPUT_DIR = Path("recordings")
OUTPUT_DIR = Path("dwTXT")
OUTPUT_DIR.mkdir(exist_ok=True)

whisper_model = WhisperModel(
    "large-v3",
    device="cuda",
    compute_type="int8_float16",
    cpu_threads=4,
    num_workers=1
)

diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=True
)

def find_speaker(diarization, start):
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if turn.start <= start <= turn.end:
            return speaker
    return "Unknown"

for audio_file in INPUT_DIR.iterdir():
    if audio_file.suffix.lower() not in {".wav", ".mp3", ".m4a", ".flac"}:
        continue

    out_file = OUTPUT_DIR / f"{audio_file.stem}.md"
    if out_file.exists():
        continue

    diarization = diarization_pipeline(str(audio_file))
    segments, info = whisper_model.transcribe(
        str(audio_file),
        beam_size=5,
        vad_filter=True,
        language="en"
    )

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"# Transcription with Diarization: {audio_file.name}\n\n")

        for seg in segments:
            speaker = find_speaker(diarization, seg.start)
            f.write(
                f"**{speaker}** [{seg.start:.2f} → {seg.end:.2f}]\n"
                f"{seg.text.strip()}\n\n"
            )

    print(f"Saved → {out_file}")
