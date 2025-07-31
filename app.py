from transformers import pipeline
import torchaudio
import torch
import gradio as gr
import os
from pydub import AudioSegment
import tempfile

# Load Whisper ASR pipeline 
asr = pipeline("automatic-speech-recognition", model="openai/whisper-medium", device=0 if torch.cuda.is_available() else -1)

#  convert audio to WAV format
def convert_to_wav(input_file):
    audio = AudioSegment.from_file(input_file)
    audio = audio.set_frame_rate(16000).set_channels(1)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio.export(temp_file.name, format="wav")
    return temp_file.name

#  split audio into 30-second chunks
def chunk_audio(file_path, chunk_length_ms=30000):
    audio = AudioSegment.from_wav(file_path)
    chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    temp_files = []
    for i, chunk in enumerate(chunks):
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        chunk.export(temp.name, format="wav")
        temp_files.append(temp.name)
    return temp_files

# Transcription logic
def transcribe(audio_path):
    try:
        converted_path = convert_to_wav(audio_path)
        chunk_files = chunk_audio(converted_path)
        full_text = ""
        for chunk in chunk_files:
            text = asr(chunk)["text"]
            full_text += text + " "
            os.remove(chunk)  # Clean up chunk
        os.remove(converted_path)
        return full_text.strip()
    except Exception as e:
        return f"Error during transcription: {e}"

# Gradio Interface
demo = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath", label="Upload Audio (WAV, MP3, OGG, etc.)"),
    outputs=gr.Textbox(label="Transcribed Text"),
    title="Multilingual Audio Transcriber",
    description="Upload audio in any language and format.",
    allow_flagging="never"
)

demo.launch()
