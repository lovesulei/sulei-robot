from faster_whisper import WhisperModel


class Whisper:
    def __init__(self, model_size="base"):
        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
        )

    def transcribe(self, audio_file):
        segments, _ = self.model.transcribe(audio_file)

        text = " ".join(segment.text for segment in segments)

        return text.strip()