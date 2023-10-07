import os
import subprocess

import torchaudio


def process_afftdn(input_file, output_file):
    # 获取输入文件的目录和文件名
    input_dir = os.path.dirname(input_file)

    # 前期音频预处理，去除低频噪音，保存为临时文件
    ffmpeg_cmd = "ffmpeg -i {} -af lowpass=f=5000,afftdn=nr=20:nf=-50 {}".format(input_file, output_file)
    print(f"process ffmepg_cmd = {ffmpeg_cmd}")
    subprocess.run(ffmpeg_cmd, shell=True)

    return output_file

def process_audio(input_file, output_file, sample_rate=16000):

    # process_wav_path = input_file.replace(".wav", "_tts_sync_p.wav")
    #
    # # 使用FFmpeg进行采样率转换
    # ffmpeg_cmd = f'ffmpeg -y -i "{input_file}" -ar {sample_rate} "{process_wav_path}"'
    # print(f"Processing with FFmpeg command: {ffmpeg_cmd}")
    # subprocess.run(ffmpeg_cmd, shell=True)
    # os.remove(input_file)

    # output_file = process_audio_noise(input_file, output_file)
    # output_file = process_audio_separation(output_file, output_file, sample_rate)
    return input_file

import librosa
import numpy as np
import soundfile as sf

def process_audio_noise(input_file, output_file):
    # Load the audio file
    audio_data, sr = librosa.load(input_file, sr=None)

    # Apply Spectral Subtraction for noise reduction
    # Parameters for Spectral Subtraction
    n_fft = 2048  # FFT window size
    hop_length = 512  # Hop length
    power = 2.0  # Exponent for the magnitude spectrogram
    margin_db = 4.0  # SNR margin in decibels

    # Compute the magnitude spectrogram
    magnitude = np.abs(librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length))

    # Compute the power spectrogram
    power_spectrogram = magnitude ** power

    # Estimate noise using minimum statistics
    noise_estimation = np.min(power_spectrogram, axis=1, keepdims=True)

    # Apply spectral subtraction
    enhanced_power = np.maximum(power_spectrogram - noise_estimation, np.finfo(float).eps)

    # Inverse STFT to obtain the enhanced audio signal
    enhanced_magnitude = np.sqrt(enhanced_power)
    enhanced_audio = librosa.istft(enhanced_magnitude * np.exp(1j * np.angle(librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length))), hop_length=hop_length)

    # Save the enhanced audio to the output file
    sf.write(output_file, enhanced_audio, sr)
    os.remove(input_file)
    return output_file

import torch
from speechbrain.pretrained import SpectralMaskEnhancement

enhance_model = SpectralMaskEnhancement.from_hparams(
    source="speechbrain/metricgan-plus-voicebank",
    savedir="pretrained_models/metricgan-plus-voicebank",
)

def process_audio_separation(input_file, output_file, sample_rate=16000):

    # Load and add fake batch dimension
    noisy = enhance_model.load_audio(
        input_file
    ).unsqueeze(0)

    # Add relative length tensor
    enhanced = enhance_model.enhance_batch(noisy, lengths=torch.tensor([1.]))

    # Saving enhanced signal on disk
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    output_file = output_file.replace(".wav", "_sp.wav")
    # enhanced_np = enhanced.detach().cpu().numpy()
    torchaudio.save(output_file, enhanced.cpu(), sample_rate)
    os.remove(input_file)
    return output_file
    # sf.write(output_file, enhanced_np, sample_rate)

# import torchaudio
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
# # 输入的音频文件为wav格式，要求使用Python代码或者模型对其进行降噪处理
# # Load the Hugging Face model and processor
# model_name = "./checkpoints/ConvTasNet_Libri1Mix_enhsingle_16k"
# processor = Wav2Vec2Processor.from_pretrained(model_name)
# model = Wav2Vec2ForCTC.from_pretrained(model_name)
# def process_audio_separation(input_file, output_file):
#
#     # Load the audio file
#     waveform, sample_rate = torchaudio.load(input_file)
#     # Tokenize the audio and obtain features
#     inputs = processor(waveform, return_tensors="pt", padding="longest")
#
#     # Perform speech separation
#     with torch.no_grad():
#         logits = model(inputs.input_values).logits
#
#     # Convert logits to waveform
#     separated_waveform = processor.decode(logits.argmax(dim=-1)[0], skip_special_tokens=True)
#
#     # Save the separated source (speech) as an audio file
#     torchaudio.save(output_file, separated_waveform.squeeze(), sample_rate)
#
#     return output_file