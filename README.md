# ChatFace



## TODO

- [ ] 语音识别 - Paraformer
- [ ] LLM对话 - QWen1.5
- [ ] 语音生成 - MockingBird
- [x] 语音驱动视频生成 - GeneFace++



## 环境安装 （假设已配置过GeneFace, 见GeneFace的TEST.md）

```bash
# GeneFace++

## 需要 pytorch 2.0.1 (2.1 会报错)
sudo pip3.9 install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

## MMCV安装
sudo pip3.9 install openmim==0.3.9
sudo mim install mmcv==2.1.0 # 使用mim来加速mmcv安装

sudo pip3.9 install mediapipe
sudo pip3.9 install pyloudnorm
sudo pip3.9 install setproctitle

# ASR
sudo pip3.9 install funasr
```



## 测试

```bash
# 生成 video
CUDA_VISIBLE_DEVICES=0 python3.9 inference/genefacepp_infer.py --head_ckpt= --torso_ckpt=checkpoints/motion2video_nerf/News_torso --drv_aud=data/raw/val_wavs/MacronSpeech.wav --out_name=infer_outs/News_demo.mp4

# UI界面
CUDA_VISIBLE_DEVICES=0 python3.9 inference/app_genefacepp.py --head_ckpt= --torso_ckpt=checkpoints/motion2video_nerf/News_torso
```
