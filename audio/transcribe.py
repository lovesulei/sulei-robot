from audio.whisper import Whisper

whisper = Whisper()

text = whisper.transcribe("test.wav")

print("You:", text)