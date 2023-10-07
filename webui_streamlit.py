import os
from inference_webui import run, preprocess_audio, tmp_path, generate_random_filename
from src.utils import audio_process_tool
from tts import xtts

# Import the necessary libraries
import streamlit as st

# Everything is accessible via the st.secrets dict:

st.write("DB username:", st.secrets["db_username"])
st.write("DB password:", st.secrets["db_password"])

# And the root-level secrets are also accessible as environment variables:

import os

st.write(
    "Has environment variables been set:",
    os.environ["db_username"] == st.secrets["db_username"],
)


def main():
    run_app()

def display_audio_preview(audio_file):
    audio_bytes = None
    if audio_file is not None:
        audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/wav')

def display_video_preview(video_file):
    video_bytes = None
    if video_file is not None:
        video_bytes = video_file.read()
    st.video(video_bytes, format='video/mp4')

def text_to_speech_section():
    # Section 1: Text-to-Speech
    st.markdown("<h2 style='font-size: 18px; font-weight: bold;'>1. 文本到语音合成</h2>",
                unsafe_allow_html=True)
    with st.expander("点击展开合成选项", expanded=False):
        # Explanation
        st.markdown(
            "<p style='font-size: 12px; color: #808080;'>"
            "通过发音人语音克隆方式合成数字人语音，发音人语音长度最好在3秒到15秒之间，且不能有任何杂音才能达到语音克隆的效果。"
            "</p>",
            unsafe_allow_html=True
        )
        # Section 0: Select speaker_audio or speaker_video
        choice = st.radio("选择发音人音频来源方式（audio或video）:", ("speaker_audio", "speaker_video"),
                          index=0)  # Default selected: speaker_audio
        text_input = st.text_area("输入需要合成语音的文字内容", "")
        speaker_audio = st.file_uploader("选择发音人语音文件(音频)",
                                         type=["wav", "mp3"]) if choice == "speaker_audio" else None
        if speaker_audio:
            # Display audio and video previews
            display_audio_preview(speaker_audio)
        speaker_video = st.file_uploader("选择发音人视频文件", type=["mp4"]) if choice == "speaker_video" else None
        if speaker_video:
            # Display audio and video previews
            display_video_preview(speaker_video)
        speed = st.slider("Speed", min_value=0.5, max_value=2.0, step=0.05, value=1.0)
        decoder_iterations = st.slider("Decoder Iterations", min_value=1, max_value=1024, step=1, value=50)
        sample_rate = st.slider("Sample Rate", min_value=8000, max_value=48000, step=1000, value=16000)
        audio_button = st.button(label="合成语音", use_container_width=True)
        return text_input, speaker_audio, speaker_video, speed, decoder_iterations, sample_rate, audio_button

def video_to_video_section():
    # Section 2: Audio + Video Synthesis
    st.markdown("<h2 style='font-size: 18px; font-weight: bold;'>2. 语音+视频合成新的数字人视频</h2>",
                unsafe_allow_html=True)
    audio_input = st.file_uploader("数字人语音，选择音频文件，或者从第一步合成", type=["wav", "mp3"])
    if audio_input:
        # Display audio and video previews
        display_audio_preview(audio_input)
    video_input = st.file_uploader("数字人视频，选择数字人基础视频文件", type=["mp4"])
    if video_input:
        # Display audio and video previews
        display_video_preview(video_input)
    use_DAIN_input = st.checkbox("使用DAIN")
    time_step_input = st.slider("选择插帧频率", min_value=0.1, max_value=1.0, step=0.01, value=0.5)
    enhancer_input = st.selectbox("选择增强类型", ["none", "lip", "face"])
    result_video = st.empty()
    result_video.write("等待视频合成...")
    video_button = st.button(label="合成视频", use_container_width=True)
    return audio_input, video_input, use_DAIN_input, time_step_input, enhancer_input, result_video, video_button

def run_app():
    # Title with 22pt bold font
    st.markdown(
        "<h1 style='font-size: 22pt; font-weight: bold;'>数字人WebUI</h1>",
        unsafe_allow_html=True
    )
    # Description with 16pt font and light gray color
    st.markdown(
        "<p style='font-size: 12pt; color: #808080;'>"
        "1. 本项目基于SadTalkers实现视频唇形合成的Wav2lip.<br>"
        "2. 通过以视频文件方式进行语音驱动生成唇形，设置面部区域可配置的增强方式进行合成唇形（人脸）区域画面增强，提高生成唇形的清晰度.<br>"
        "3. 使用DAIN 插帧的DL算法对生成视频进行补帧，补充帧间合成唇形的动作过渡，使合成的唇形更为流畅、真实以及自然."
        "</p>",
        unsafe_allow_html=True
    )
    text_input, speaker_audio, speaker_video, speed, decoder_iterations, sample_rate, audio_button = text_to_speech_section()
    audio_input, video_input, use_DAIN_input, time_step_input, enhancer_input, result_video, video_button = video_to_video_section()

    # 合成语音处理
    if audio_button:
        if not text_input:
            st.warning("请输入需要合成的文本内容。")
        elif not (speaker_audio or speaker_video):
            st.warning("请选择发音人语音文件或视频文件。")
        else:
            try:
                if speaker_audio:
                    audio_file = preprocess_audio(speaker_audio)
                elif speaker_video:
                    audio_file = preprocess_audio(speaker_video)
                else:
                    raise ValueError("请选择发音人语音文件或视频文件")

                process_audio_file = audio_process_tool.process_audio(audio_file, os.path.join(tmp_path,
                                                                                               generate_random_filename(
                                                                                                   audio_file) + ".wav"), sample_rate)
                output_wav_file = os.path.join(os.path.dirname(process_audio_file), 'output.wav')
                xtts.tts_sync(text_input, output_wav_file, process_audio_file, speed, decoder_iterations, sample_rate)
                # 将语音文件 output_wav_file 放到audio_input控件中播放，并供接下来的逻辑使用。
                audio_input = open(output_wav_file, 'rb')

                st.success("语音合成完成！")

            except Exception as e:
                st.error(f"语音合成失败：{str(e)}")

    # 合成视频处理
    if video_button:
        if not audio_input:
            st.warning("请上传音频文件或从第一步合成音频。")
        elif not video_input:
            st.warning("请选择数字人基础视频文件。")
        else:
            try:
                result_video.write('正在合成视频中......')
                result_dir = os.path.dirname(video_input)
                print(f"result_dir = {result_dir}")
                result_video_path = run(audio_input, video_input, enhancer_input, use_DAIN_input, time_step_input,
                                        result_dir)
                result_video = st.video(result_video_path)
                st.success("视频合成完成！")
            except Exception as e:
                st.error(f"视频合成失败：{str(e)}")

if __name__ == "__main__":
    main()
