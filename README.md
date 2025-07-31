# multilingual-speech-transcriber

This project is a multilingual speech recognition app that converts spoken audio into text using OpenAI's Whisper model via Hugging Face Transformers. The application supports various audio formats such as .mp3, .wav, .ogg, and more, ensuring ease of use across different sources.

To handle longer audio files effectively, the app automatically:

Converts input audio to a standard 16kHz mono WAV format using pydub.

Splits the audio into 30-second chunks to prevent model memory issues.

Transcribes each chunk using the Whisper-medium model.

Combines the individual results into a complete transcript.

The interface is built using Gradio, making it user-friendly and accessible directly from a browser. The app runs seamlessly on both CPU and GPU, and is fully deployable on Hugging Face Spaces.

This tool is especially useful for transcribing interviews, lectures, podcasts, and other multilingual audio content. It was built as part of an AI learning internship to explore real-world applications of speech recognition models.
