# ChatFace



## TODO

- [x] 语音识别 - Paraformer
- [x] LLM对话 - QWen1.5
- [x] 语音生成 - coqui/TTS
- [x] 语音驱动视频生成 - GeneFace++
- [x] Pipeline
- [x] Demo webapp



## 环境安装

```bash
# GeneFace++ (如过已配置过 GeneFace++ 环境, 可忽略)

## 需要 pytorch 2.1 + CUDA 11.8
sudo pip3.9 install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

## pytorch3d (https://github.com/facebookresearch/pytorch3d/archive/refs/tags/v0.7.6.tar.gz)
tar xvfz pytorch3d-0.7.6.tar.gz
cd pytorch3d-0.7.6 && sudo pip3.9 install -e .

## 安装 torch-ngp
sudo bash docs/install_ext.sh

## MMCV安装
sudo pip3.9 install openmim==0.3.9
sudo mim install mmcv==2.1.0 # 使用mim来加速mmcv安装

sudo pip3.9 install mediapipe
sudo pip3.9 install pyloudnorm
sudo pip3.9 install setproctitle

# ASR
sudo pip3.9 install funasr

# TTS 环境配置 见 tts/TEST.md

# Qwen2
sudo pip3.9 install transformers==4.37.2
```



## 测试

```bash
# ASR 语音识别
python3.9 asr/paraformer.py

# Qwen2 对话
python3.9 llm/qw2.py

# TTS 生成语音
python3.9 tts/xtts2.py

# Geneface++ 生成 video
CUDA_VISIBLE_DEVICES=0 python3.9 inference/genefacepp_infer.py --head_ckpt= --torso_ckpt=checkpoints/motion2video_nerf/News_torso --drv_aud=data/raw/val_wavs/MacronSpeech.wav --out_name=infer_outs/News_demo.mp4

# Geneface++ UI界面
CUDA_VISIBLE_DEVICES=0 python3.9 inference/app_genefacepp.py --head_ckpt= --torso_ckpt=checkpoints/motion2video_nerf/News_torso

# 命令行 pipeline
python3.9 inference/pipeline.py

# pipeline webapp demo
python3.9 inference/app_pipeline.py
```
