import json
#import os
#from pathlib import Path

#import IPython.display as ipd
from fairseq import hub_utils
#from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
#from fairseq.models.speech_to_text.hub_interface import S2THubInterface
from fairseq.models.text_to_speech import CodeHiFiGANVocoder
from fairseq.models.text_to_speech.hub_interface import VocoderHubInterface

#from huggingface_hub import snapshot_download
import torchaudio
'''
cache_dir = os.getenv("HUGGINGFACE_HUB_CACHE")

# speech synthesis           
library_name = "fairseq"
cache_dir = (
    cache_dir or (Path.home() / ".cache" / library_name).as_posix()
)
cache_dir = snapshot_download(
    f"facebook/unit_hifigan_HK_layer12.km2500_frame_TAT-TTS", cache_dir=cache_dir, library_name=library_name
)
'''

cache_dir = "../lm_model/unit_hifigan_HK_layer12.km2500_frame_TAT-TTS"

x = hub_utils.from_pretrained(
    cache_dir,
    "model.pt",
    ".",
    archive_map=CodeHiFiGANVocoder.hub_models(),
    config_yaml="config.json",
    fp16=False,
    is_vocoder=True,
)

with open(f"{x['args']['data']}/config.json") as f:
    vocoder_cfg = json.load(f)
assert (
    len(x["args"]["model_path"]) == 1
), "Too many vocoder models in the input"

vocoder = CodeHiFiGANVocoder(x["args"]["model_path"][0], vocoder_cfg)
tts_model = VocoderHubInterface(vocoder_cfg, vocoder)

if __name__ == '__main__':

    unit = "1 2 3 4"
    tts_sample = tts_model.get_model_input(unit)
    wav, sr = tts_model.get_prediction(tts_sample)

    # 保存为 wav, https://stackoverflow.com/questions/10357992/how-to-generate-audio-from-a-numpy-array
    import numpy as np
    from scipy.io.wavfile import write

    data = wav.numpy()
    scaled = np.int16(data / np.max(np.abs(data)) * 32767)
    write('test.wav', sr, scaled)

    #ipd.Audio(wav, rate=sr)
