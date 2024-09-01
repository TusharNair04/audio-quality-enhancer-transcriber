import torchaudio

def load_recorded_audio(path_audio, input_sample_rate=48000, output_sample_rate=16000):
    waveform, sample_rate = torchaudio.load(path_audio)
    waveform_resampled = torchaudio.functional.resample(waveform, orig_freq=input_sample_rate, new_freq=output_sample_rate)
    sample = waveform_resampled.numpy()[0]
    return sample

def run_inference(path_audio, output_lang, pipe, output_file):
    sample = load_recorded_audio(path_audio)
    result = pipe(sample, generate_kwargs={"language": output_lang, "task": "transcribe"})
    
    with open(output_file, 'w') as file:
        file.write(result["text"])
