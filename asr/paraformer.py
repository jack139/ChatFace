# 预训练模型来源：
# https://huggingface.co/funasr/paraformer-zh-streaming
# https://github.com/alibaba-damo-academy/FunASR

import os
import soundfile
from io import BytesIO
from funasr import AutoModel
from datetime import datetime

'''
Note: chunk_size is the configuration for streaming latency. 
    [0,10,5] indicates that the real-time display granularity is 10*60=600ms, 
    and the lookahead information is 5*60=300ms. Each inference input is 600ms 
    (sample points are 16000*0.6=960), and the output is the corresponding text. 
    For the last speech segment input, is_final=True needs to be set to force 
    the output of the last word.
'''
chunk_size = [0, 10, 5] #[0, 10, 5] 600ms, [0, 8, 4] 480ms
encoder_chunk_look_back = 4 #number of chunks to lookback for encoder self-attention
decoder_chunk_look_back = 1 #number of encoder chunks to lookback for decoder cross-attention


class ParaformerInfer:
    def __init__(self, model_dir="paraformer-zh-streaming", device='cpu'):
        self.model_dir = model_dir
        self.model = AutoModel(model=model_dir, model_revision="v2.0.4", device=device)

    # non-streaming
    def call_nonstreaming(self, speech): 
        start_time = datetime.now()

        result = []

        try:
            res = self.model.generate(input=speech, batch_size_s=1)
        except Exception as e:
            print(f"ERROR: {e}")
            return None

        #print(res)
        if len(res)>0:
            result.append(res[0].get('text', ''))

        print('wav2text elapsed: {!s}s'.format(datetime.now() - start_time))

        return ''.join(result)


    # streaming
    def call_streaming(self, speech): 
        chunk_stride = chunk_size[1] * 960 # 600ms

        start_time = datetime.now()

        result = []
        cache = {}
        total_chunk_num = int(len((speech)-1)/chunk_stride+1)
        print("total_chunk_num=", total_chunk_num)
        for i in range(total_chunk_num):
            speech_chunk = speech[i*chunk_stride:(i+1)*chunk_stride]
            is_final = i == total_chunk_num - 1
            try:
                res = self.model.generate(input=speech_chunk, cache=cache, is_final=is_final, chunk_size=chunk_size, 
                            encoder_chunk_look_back=encoder_chunk_look_back, decoder_chunk_look_back=decoder_chunk_look_back)
            except Exception as e:
                print(f"Chunk {i} ERROR: {e}")
                continue

            #print(i, res)
            if len(res)>0:
                result.append(res[0].get('text', ''))

        print('wav2text elapsed: {!s}s'.format(datetime.now() - start_time))

        return ''.join(result)


    def wav2text(self, wav_data, streaming=True): 
        tmp_buff = BytesIO()
        tmp_buff.write(wav_data)
        tmp_buff.seek(0)
        speech, sample_rate = soundfile.read(tmp_buff, always_2d=True)
        print("sample_rate:", sample_rate)

        if speech.shape[1]==2: # Stereo to Mono wav
            print("Stereo --> Mono wav")
            speech = speech.sum(axis=1) / 2

        if streaming:
            return self.call_streaming(speech)
        else:
            return self.call_nonstreaming(speech)


model_path = "../lm_model/paraformer-zh-streaming"
infer = ParaformerInfer(model_path, device="cuda")


if __name__ == '__main__':

    test_file = f"{model_path}/example/asr_example.wav"
    #test_file = "data/raw/val_wavs/news2_16k.wav"

    with open(test_file, 'rb') as f: 
        print(infer.wav2text(f.read()))
