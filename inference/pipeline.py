# wav --> [ASR] --> text --> [LLM] --> text --> [TTS] --> wav --> [GFPP] --> video

import sys
sys.path.append('./')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 假设只有一块显卡
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from asr import paraformer as ASR
from llm import qw2 as LLM
from tts import xtts2 as TTS
from inference import genefacepp_infer as GFPP


def infer_from_text(text, video_path='infer_outs/pipeline_demo.mp4'):
    # text --> [LLM] --> text
    response = LLM.infer(text + "（注意，回答请不要超过50个字）", max_new_tokens=50)

    if len(response)==0:
        return None, 'LLM未回答文字'

    print('reply text: ', response)

    # text --> [TTS] --> wav    
    # 中文不能超过82个字
    # Warning: The text length exceeds the character limit of 82 for language 'zh'
    TTS.text2wav(response, 
        language="zh-cn", 
        speaker_wav="data/test.wav", # 模仿的声音
        output_wav_path="infer_outs/pipeline.wav"
    )

    #wav --> [GFPP] --> video
    inp = {
            'a2m_ckpt': 'checkpoints/audio2motion_vae',
            'postnet_ckpt': '',
            'head_ckpt': '',
            'torso_ckpt': 'checkpoints/motion2video_nerf/News_torso',
            'drv_audio_name': 'infer_outs/pipeline.wav',
            'drv_pose': 'nearest',
            'blink_mode': 'period',
            'temperature': 0.2,
            'mouth_amp': 0.4,
            'lle_percent': 0.2,
            'debug': False,
            'out_name': video_path,
            'raymarching_end_threshold': 0.01,
            'low_memory_usage': False,
            }
    GFPP.GeneFace2Infer.example_run(inp)

    return response, f"video generated in {video_path}."


def infer_from_wav(wav_file):
    if not os.path.exists(wav_file):
        return None, 'wav文件不存在'

    # wav --> [ASR] --> text
    with open(wav_file, 'rb') as f: 
        text = ASR.infer.wav2text(f.read())
    if len(text)>0:
        print('text: ', text)
        return infer_from_text(text)
    else:
        return None, '语音未识别到文字'


if __name__ == '__main__':

    print("Pipeline DEMO. '@'开头可以输入wav文件")
    print('=' * 85)

    while True:
        raw_input_text = input("请输入您的问题：")
        raw_input_text = str(raw_input_text).strip()
        if len(raw_input_text.strip()) == 0:
            break
        if raw_input_text[0]=='@':
            response, mp4_path = infer_from_wav(raw_input_text[1:])
        else:
            response, mp4_path = infer_from_text(raw_input_text)

        print(f"Response: {response} ({mp4_path})")

    print("Bye!")
