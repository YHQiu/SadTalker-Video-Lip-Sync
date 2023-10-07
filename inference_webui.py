# 定义推理函数
import hashlib
import os
import subprocess
from argparse import Namespace
from datetime import datetime
import random

import cv2
from pydub import AudioSegment

from tts import xtts
from inference import main
from src.utils import audio_process_tool


def run(driven_audio, source_video, enhancer, use_DAIN, time_step, result_dir):
    args = Namespace(
        driven_audio=driven_audio,
        source_video=source_video,
        checkpoint_dir='./checkpoints',
        result_dir=result_dir,
        batch_size=1,
        enhancer=enhancer,
        cpu=False,  # Assuming you don't want to run on CPU by default
        use_DAIN=use_DAIN,
        DAIN_weight='./checkpoints/DAIN_weight',
        dian_output='dian_output',
        time_step=time_step,
        remove_duplicates=False,
    )
    return main(args)

def convert_to_wav(input_file):
    # Ensure the input file is in WAV format
    if input_file.lower().endswith('.wav'):
        return input_file

    if input_file.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
        # Get the audio codec and create an AudioSegment
        # Create the output WAV file path
        output_file = os.path.splitext(input_file)[0] + '.wav'

        # Use FFmpeg to extract audio from video
        cmd = [
            'ffmpeg',
            '-y',
            '-i', input_file,  # Input video file
            '-vn',  # Disable video stream
            '-ac', '2',  # Set stereo audio
            '-ar', '44100',  # Set audio sample rate to 44100 Hz
            '-acodec', 'pcm_s16le',  # Set audio codec to PCM signed 16-bit little-endian
            output_file  # Output WAV file
        ]

        # Run the FFmpeg command
        # 将 cmd 列表转换为字符串命令
        cmd_str = ' '.join(cmd)
        # 执行命令
        subprocess.run(cmd_str, shell=True)

    else:
        # Load the audio from other audio formats
        audio = AudioSegment.from_file(input_file)
        # Create the output WAV file path
        output_file = os.path.splitext(input_file)[0] + '.wav'
        # Export the audio to WAV format
        audio.export(output_file, format='wav')

    print(f"process success ouput_audio_file = {output_file}")
    return output_file

def preprocess_audio(input_file):
    # Check if the input file is a video file (e.g., MP4)
    if input_file.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
        print(f"input file is {input_file}")
    else:
        audio = AudioSegment.from_file(input_file)
        duration_in_seconds = len(audio) / 1000  # Duration in seconds
        if duration_in_seconds < 3:
            raise ValueError("为了保证合成效果发音人语音文件不得低于3s。")

    # Call the convert_to_wav function if audio duration is >= 1 minute
    return convert_to_wav(input_file)

tmp_path = "/tmp/gradio"
if not os.path.exists(tmp_path):os.makedirs(tmp_path, exist_ok=True)

def get_audio_length(audio_file_path):
    """
    获取音频文件的长度（秒）。
    """
    audio = AudioSegment.from_file(audio_file_path)
    return len(audio) / 1000.0  # 转换为秒


def get_video_length(video_file_path):
    """
    获取视频文件的长度（秒）。
    """
    video = cv2.VideoCapture(video_file_path)

    # 检查视频是否成功打开
    if not video.isOpened():
        raise Exception("无法打开视频文件")

    # 获取视频的帧数和帧率
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(video.get(cv2.CAP_PROP_FPS))

    # 计算视频长度（秒）
    video_length = frame_count / frame_rate

    # 释放视频对象
    video.release()

    return video_length

def generate_random_filename(input):
    current_time = str(datetime.now())  # 获取当前时间戳字符串
    random_number = str(random.random())  # 生成一个随机浮点数字符串
    hash_str = "genereate_hash" + input + current_time + random_number
    return hashlib.md5(hash_str.encode()).hexdigest()

def run_audio(decoder_iterations, post_filter_magnitude_threshold, post_filter_mel_smoothing, sample_rate,
              speaker_audio, speaker_video, speed, text):
    if speaker_audio is not None:
        audio_file = preprocess_audio(speaker_audio)
    else:
        audio_file = preprocess_audio(speaker_video)
    # 对音频进行降噪处理
    process_audio_file = audio_process_tool.process_audio(audio_file, os.path.join(tmp_path, generate_random_filename(
        audio_file) + ".wav"), sample_rate)
    print(f"process audio file is {process_audio_file}")
    output_wav_path = os.path.join(tmp_path, generate_random_filename(text) + ".wav")
    xtts.tts_sync(text, output_wav_path, process_audio_file, speed, decoder_iterations, sample_rate,
                  post_filter_magnitude_threshold, post_filter_mel_smoothing)
    return output_wav_path