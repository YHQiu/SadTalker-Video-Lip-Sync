from TTS.api import TTS
import re
import torch

# 自定义分词器和正则表达式来检查仅包含一个标点符号的句子
def custom_tokenizer(text):
    # 使用正则表达式将文本分割成句子
    sentences = re.split(r'([。！？])', text)
    sentences = [s1 + s2 for s1, s2 in zip(sentences[::2], sentences[1::2])]
    # 过滤掉仅包含一个标点符号的句子，并移除奇数个双引号
    return [remove_odd_quotes(sentence.strip()) for sentence in sentences if not is_single_punctuation(sentence)]

def is_single_punctuation(sentence):
    # 正则表达式检查句子是否仅包含一个标点符号
    return re.match(r'^[^\w\s]+$', sentence)

def remove_odd_quotes(sentence):
    # 使用正则表达式移除奇数个双引号
    sentence = sentence.replace("”", " ")
    if sentence.count('"') % 2 == 1 or sentence.count('”') % 2 == 1:
        sentence = re.sub(r'["“]', '', sentence)
    return sentence

# 生成语音，使用自定义分词器和过滤器
def tts_sync(text: str, output_wav_path: str, speaker_wav_path: str, speed: int = 1.0, decoder_iterations: int = 30, sample_rate=22500, post_filter_magnitude_threshold=0.02, post_filter_mel_smoothing=0.85, language: str = "zh-cn"):

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v1")
    if torch.cuda.is_available():
        tts.to(device='cuda')

    # 使用自定义分词器拆分文本成句子
    sentences = custom_tokenizer(text)
    input = ""
    for i, sentence in enumerate(sentences):
        input += sentence

    tts.tts_to_file(
        text=input,
        file_path=output_wav_path,
        speaker_wav=speaker_wav_path,
        language=language,
        speed=speed,
        decoder_iterations=decoder_iterations
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output_wav_path
