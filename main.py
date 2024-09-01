from pydub import AudioSegment
from audio_preprocessing import enhance_audio
from inference import run_inference
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch

# Load the audio file and export to wav format
input_file = r"path_to_your_audio_file"
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
