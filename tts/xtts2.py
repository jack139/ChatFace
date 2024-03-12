from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.audio.numpy_transforms import save_wav

print("load config ...")
config = XttsConfig()
config.load_json("../lm_model/XTTS-v2/config.json")

#print(config)

print("load model weights ...")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="../lm_model/XTTS-v2/", eval=True)
model.cuda()

if __name__ == '__main__':
    print("infer ...")
    outputs = model.synthesize(
        #"It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
        "我花了很长时间才形成自己的声音，现在我有了声音，我不会保持沉默。",
        config,
        speaker_wav="data/asr_example.wav",
        gpt_cond_len=3,
        #language="en", 
        language="zh-cn"
    )

    # dict_keys(['wav', 'gpt_latents', 'speaker_embedding'])
    #print(outputs.keys())

    save_wav(wav=outputs['wav'], path="infer_outs/tts_output.wav", sample_rate=config.model_args.output_sample_rate, pipe_out=None)