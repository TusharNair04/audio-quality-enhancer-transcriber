
# Audio Quality Enhancer and Transcriber using OpenAI Whisper

This repository contains a Python-based solution for enhancing and transcribing low-quality audio calls. The pipeline is specifically designed to process audio files with poor clarity, focusing on improving speech intelligibility before converting the audio into text using OpenAI's Whisper Model.

## Features

- **Audio Enhancement:** Automatically filters and enhances low-quality audio calls, targeting the frequency range of human speech to reduce noise and improve clarity.
- **Speech Recognition:** Utilizes OpenAI's Whisper model, optimized for transcribing challenging audio, to produce accurate text transcriptions.
- **Modular Design:** The code is organized into separate modules for audio preprocessing and inference, making it easy to modify and extend.

## Requirements

To get started, you'll need the following dependencies installed:

- Python 3.x
- `pydub`
- `librosa`
- `numpy`
- `scipy`
- `torch`
- `transformers`
- `torchaudio`

You can install these dependencies using `pip`:

```bash
pip install pydub librosa numpy scipy torch transformers torchaudio
```

## Usage

Follow these steps to enhance and transcribe your low-quality audio calls:

1. **Prepare Your Audio File:**
   - Place your low-quality audio call file in the project directory or specify the path in the script.

2. **Run the Main Script:**
   - Execute the `main.py` script to enhance the audio quality and transcribe the audio.

   ```bash
   python main.py
   ```

   - The script processes the audio file, applying a bandpass filter to focus on voice frequencies and normalizing the audio for better clarity.
   - It then uses the Whisper model to transcribe the processed audio into text.

3. **Output:**
   - The enhanced audio file will be saved as `improved_audio.wav`.
   - The transcription will be saved as `improved_audio.txt`.

### Example

Here's how you can use the script in a typical scenario:

```python
# Load the audio file
from pydub import AudioSegment
from audio_preprocessing import enhance_audio
from inference import run_inference
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch

input_file = "path_to_your_audio_file.aiff"
audio = AudioSegment.from_file(input_file)
audio.export("temp_audio.wav", format="wav")

# Enhance the audio
enhance_audio("temp_audio.wav", "improved_audio.wav")

# Load and setup the Whisper model for speech recognition
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch_dtype = torch.float32
model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device.type if device.type == 'cpu' else "cuda:0"
)

# Run the inference to transcribe the enhanced audio
run_inference("improved_audio.wav", "en", pipe, "improved_audio.txt")
```

## Applications

This pipeline is ideal for situations where low-quality audio calls need to be transcribed, such as:

- Customer service recordings
- Interview calls
- Any audio source where the original sound quality is poor

By enhancing the audio before transcription, this tool significantly improves the accuracy and clarity of the resulting text.
Note: When working with audio data, especially in contexts such as customer service recordings or interview calls, it is crucial to handle the data with care to ensure privacy and compliance with relevant regulations, such as GDPR.
