"""
推理界面
auth: qiuhongyu
date: 2023-09-15
"""

import sys

import xtts
from src.utils import video_concat, audio_process_tool

sys.path.append('/usr/local/lib/python3.9/site-packages')

import cv2
import hashlib
import os
from argparse import Namespace
from datetime import datetime
import random

import gradio as gr
import subprocess
import time
import numpy as np
import cv2

from pydub import AudioSegment

from inference import main


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

# 定义推理函数
def run(driven_audio, source_video, enhancer, use_DAIN, time_step, return_path):
    args = Namespace(
        driven_audio=driven_audio,
        source_video=source_video,
        checkpoint_dir='./checkpoints',
        result_dir='./results',
        batch_size=1,
        enhancer=enhancer,
        cpu=False,  # Assuming you don't want to run on CPU by default
        use_DAIN=use_DAIN,
        DAIN_weight='./checkpoints/DAIN_weight',
        dian_output='dian_output',
        time_step=time_step,
        remove_duplicates=False,
        return_path=return_path
    )
    return main(args)

def generate_random_filename(input):
    current_time = str(datetime.now())  # 获取当前时间戳字符串
    random_number = str(random.random())  # 生成一个随机浮点数字符串
    hash_str = "genereate_hash" + input + current_time + random_number
    return hashlib.md5(hash_str.encode()).hexdigest()

def run_audio_inference(submit_tts_button, text, speaker_audio, speaker_video, speed, decoder_iterations, sample_rate, post_filter_magnitude_threshold, post_filter_mel_smoothing):

    if text is None:
        raise ValueError("请先输入需要合成语音的文字。")
    if speaker_audio is None and speaker_video is None:
        raise ValueError("请选择需要模仿的发音人语音,音频或者视频格式都可以。")

    if speaker_audio is not None:
        audio_file = preprocess_audio(speaker_audio)
    else:
        audio_file = preprocess_audio(speaker_video)

    # 对音频进行降噪处理
    process_audio_file = audio_process_tool.process_audio(audio_file, os.path.join(tmp_path, generate_random_filename(audio_file)+".wav"), sample_rate)
    print(f"process audio file is {process_audio_file}")

    output_wav_path = os.path.join(tmp_path, generate_random_filename(text)+".wav")
    xtts.tts_sync(text, output_wav_path, process_audio_file, speed, decoder_iterations, sample_rate, post_filter_magnitude_threshold, post_filter_mel_smoothing)
    return output_wav_path

def run_inference(submit_button, driven_audio, source_video, enhancer:gr.Dropdown, use_DAIN:gr.Checkbox, time_step:gr.Slider, result_video_output: gr.Video):
    # 检查音频和视频是否文件都已选择
    if driven_audio is None:
        raise ValueError("请选择数字人音频文件")

    if source_video is None:
        raise ValueError("请选择数字人视频文件")

    # 比对音频driven_audio和视频source_video的长度，如果音频长度小于视频，则使用ffmpeg将视频裁剪为和音频相同的长度；如果音频长于视频，则连续多次拼接视频知道其长度等于音频长度，并裁剪视频多余的部分。
    # 检查音频和视频长度，进行裁剪或拼接

    # 获取音频和视频的长度
    audio_length = get_audio_length(driven_audio)  # Implement get_audio_length function
    video_length = get_video_length(source_video)  # Implement get_video_length function
    # 音频较短，需要裁剪视频
    trimmed_video_filename = generate_random_filename(source_video) + "_trimmed_video.mp4"
    trimmed_video_path = os.path.join(os.path.dirname(source_video), trimmed_video_filename)
    if audio_length < video_length:
        processed_video_path = source_video
    elif audio_length > video_length:
        processed_video_path = trimmed_video_path
        video_concat.process_video(source_video, processed_video_path, video_length, audio_length)
    else:
        processed_video_path = source_video  # 音频和视频长度相同，无需处理

    # 构建命令行参数
    current_timestamp = int(time.time())
    result_path = f"./results/output/{current_timestamp}.mp4"

    # 执行推理
    try:
        result_video_path = run(driven_audio, processed_video_path, enhancer, use_DAIN, time_step, result_path)
        result_message = "推理完成！"
    except subprocess.CalledProcessError as e:
        result_video_path = None
        result_message = f"推理失败：{e.stderr.decode('utf-8')}"
        raise ValueError(result_message)

    # 读取结果视频
    print(f"end video= {result_video_path}")
    return result_video_path


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

with gr.Blocks(
    title="数字人Demo",
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.blue,  # 修改主题颜色为蓝色
        font="Arial, sans-serif"  # 使用一种清晰易读的字体
    ),
) as demo:
    with gr.Column():
        with gr.Row():
            gr.Label(label="费曼智能科技", value="数字人生成DEMO", font_size=18, padding=10)  # 调整标题样式
            with gr.Accordion("使用说明", open=False):
                gr.Label(value="1、通常文字和语音样本合成语音（或者直接选择语音文件）。2、选择数字人视频。3、生成视频", font_size=14)
        with gr.Row():
            with gr.Column():
                gr.Label(value="1、文本到语音合成（输入文字，选择发音人语音或者视频文件【二选一】）",
                         font_size=18)
                text_input = gr.inputs.Textbox(label="生成语音的文字内容", placeholder="请输入需要合成语音的文本。", lines=5)
            with gr.Accordion("语音克隆高级选项", open=False):
                speed = gr.inputs.Slider(minimum=0.5, maximum=2.0, default=1, step=0.05, label="Speed")
                decoder_iterations = gr.inputs.Slider(minimum=1, maximum=1024, default=50, step=1, label="Decoder Iterations")
                sample_rate = gr.inputs.Slider(minimum=8000, maximum=48000, default=16000, step=1000, label="Sample Rate")
                post_filter_magnitude_threshold = gr.inputs.Slider(minimum=0.0, maximum=1.0, default=0.01, step=0.01,
                                 label="Post Filter Magnitude Threshold")
                post_filter_mel_smoothing = gr.inputs.Slider(minimum=0.0, maximum=1.0, default=0.80, step=0.05, label="Post Filter Mel Smoothing")
            with gr.Column():
                # with gr.Row():
                speaker_audio = gr.Audio(label="请选择发音人语音文件(音频、视频二选一) 文件大小限制在100M以内", type="filepath", preprocess=preprocess_audio)
                speaker_video = gr.Video(label="请选择发音人视频文件(音频、视频二选一) 文件大小限制在100M以内", preprocess=preprocess_audio)
                submit_tts_button = gr.Button(value="合成语音", font_size=14)
    with gr.Row():
        with gr.Column():
            gr.Label(value="2、语音+视频合成新的数字人视频", font_size=18)
            audio_input = gr.Audio(label="选择音频文件,或者通过文本合成语音，文件大小限制在100M以内", type='filepath', preprocess=convert_to_wav)
            video_input = gr.inputs.Video(label="选择数字人基础视频文件，文件大小限制在100M以内", type="mp4")
        with gr.Column():
            with gr.Column():
                result_video_output = gr.Video(label="合成结果视频", interactive=False)
            with gr.Column():
                use_DAIN_input = gr.Checkbox(label="使用DAIN", font_size=14)
                with gr.Accordion("高级选项", open=False):
                    time_step_input = gr.Slider(minimum=0.1, maximum=1.0, value=0.5, interactive=True, label="选择插帧频率")
                    enhancer_options = ["none", "lip", "face"]
                    enhancer_input = gr.Dropdown(enhancer_options, value="none", interactive=True, label="选择增强类型")
                submit_button = gr.Button(value="合成视频", font_size=14)
    # 绑定按钮事件
    submit_tts_button.click(run_audio_inference, inputs=[submit_tts_button, text_input, speaker_audio, speaker_video, speed, decoder_iterations, sample_rate, post_filter_magnitude_threshold, post_filter_mel_smoothing], outputs=[audio_input])
    submit_button.click(run_inference, inputs=[submit_button, audio_input, video_input, enhancer_input, use_DAIN_input, time_step_input], outputs=[result_video_output])


user_or_password = [["qhy", "qiu3178278"], ["fm_01", "fm_8860"], ["fm_02", "fm_8860"]]

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", share=False, server_port=8082, show_error=True, auth=user_or_password, show_api=False)
