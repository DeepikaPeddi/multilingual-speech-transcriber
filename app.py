import gradio as gr
import librosa
import numpy as np
from transformers import pipeline

# this line of code loads multilingual asr model 
asr = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-large-xlsr-53", tokenizer="facebook/wav2vec2-large-xlsr-53")

# split audio into max 30 seconds chunks
def chunk_audio(audio, rate, max_duration=30):
    max_len = int(rate * max_duration)
    chunks = []
    for i in range(0, len(audio), max_len):
        chunks.append(audio[i:i + max_len])
    return chunks

# Transcription function
def transcribe(audio_file):
    #loads audio
    audio, rate = librosa.load(audio_file, sr=16000)
    
    # If audio <30 seconds then transcribes directly
    if len(audio) < rate * 30:
        res = asr(audio)
        return res["text"]
    
    # if audio>30bseconds , splits into chunks
    chunks = chunk_audio(audio, rate)
    full_text = ""
    for idx, chunk in enumerate(chunks):
        try:
            res = asr(chunk)
            full_text += res["text"] + " "
        except Exception as e:
            full_text += f"\n[Error in chunk {idx}: {e}]\n"
    
    return full_text.strip()

# Gradio interface to deploy in hugging face
interface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath", label="Upload Audio (.wav, .mp3, .ogg, etc.)"),
    outputs="text",
    title="Advanced Speech-to-Text Transcriber",
    description=" Upload an audio file of any language , audio file can be of any format."
)

# Launch the app
interface.launch()
