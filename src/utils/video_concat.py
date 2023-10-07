import hashlib
import os
import subprocess
from datetime import datetime
import random

transition_duration = 2

def generate_random_filename(input):
    current_time = str(datetime.now())  # 获取当前时间戳字符串
    random_number = str(random.random())  # 生成一个随机浮点数字符串
    hash_str = "genereate_hash" + input + current_time + random_number
    return hashlib.md5(hash_str.encode()).hexdigest()

def process_video(source_video, repeated_video_filename, video_length, audio_length):

    process_path = os.path.join(os.path.dirname(source_video), generate_random_filename(os.path.dirname(source_video)))
    os.makedirs(process_path)
    repeated_video_path = os.path.join(process_path, repeated_video_filename)
    num_repeats = int(audio_length / video_length) + 1
    txt_file = os.path.join(process_path, "video_list.txt")
    source_video_escaped = source_video.replace(" ", r"\ ")
    with open(txt_file, "w") as f:
        for i in range(num_repeats):
            f.write(f"file '{source_video_escaped}'\n")

    # Run ffmpeg command to concatenate the videos
    ffmpeg_cmd = f"ffmpeg -f concat -safe 0 -i \"{txt_file}\"  \"{repeated_video_path}\""
    subprocess.run(ffmpeg_cmd, shell=True)
    # print(ffmpeg_cmd)

    # Remove the temporary video list file
    os.remove(txt_file)

    processed_video_path = repeated_video_path
    print(f"{processed_video_path}")
    return processed_video_path


if __name__ == "__main__":
    source_video = "C:\\Users\\63224\\Downloads\\ltj.mp4"
    repeated_video_filename = "test_repeated_video.mp4"
    video_length = 10
    audio_length = 35
    process_video(source_video, repeated_video_filename, video_length, audio_length)
