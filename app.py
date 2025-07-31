import gradio as gr
import torchaudio
import torch
from transformers import pipeline

# Loads multilingual Wav2Vec2 model with built-in tokenizer
asr = pipeline(
    "automatic-speech-recognition",
    model="jonatasgrosman/wav2vec2-large-xlsr-53-multilingual",
    tokenizer="jonatasgrosman/wav2vec2-large-xlsr-53-multilingual"
)

# Split audio into chunks with max duration 30-seconds
def chunk_audio(audio, sample_rate, max_duration=30):
    max_length = int(sample_rate * max_duration)
    return [audio[i:i+max_length] for i in range(0, len(audio), max_length)]

# Transcription function
def transcribe(audio_file):
    waveform, sample_rate = torchaudio.load(audio_file)
    waveform = waveform.mean(dim=0).numpy()  # Convert to 1D numpy array

    if len(waveform) < sample_rate * 30:
        res = asr(waveform)
        return res["text"]

    chunks = chunk_audio(waveform, sample_rate)
    full_transcription = ""

    for i, chunk in enumerate(chunks):
        try:
            res = asr(chunk)
            full_transcription += res["text"] + " "
        except Exception as e:
            full_transcription += f"\n[Error in chunk {i}: {e}]\n"

    return full_transcription.strip()

# Gradio UI
interface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath", label="Upload Audio (.wav, .mp3, .ogg, etc.)"),
    outputs="text",
    title="Multilingual Speech Transcriber",
    description=" Upload an audio file of any language , audio file can be of any format."
    allow_flagging="never"
)

interface.launch()
